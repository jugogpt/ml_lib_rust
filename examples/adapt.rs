//! Phase 1 demo: load a prior → adapt on a local slice under a budget → predict.
//!
//! ```text
//! cargo run --release --example adapt
//! ```
//!
//! Prefers an existing `checkpoints/mnist.mlckpt` from the `mnist` example;
//! otherwise trains a short prior first.

use ml_lib_rust::{
    AdaptBatch, AdaptConfig, AdaptSession, Adam, LrSchedule, MnistMmapDataset,
    ModelContext, Tensor, TrainBudget, MV_FLAG_COST, MV_FLAG_DESIRED_OUTPUT, MV_FLAG_INPUT,
    MV_FLAG_NONE, MV_FLAG_OUTPUT, MV_FLAG_PARAMETER, MV_FLAG_REQUIRES_GRAD,
};

fn main() -> std::io::Result<()> {
    let train = MnistMmapDataset::open(
        "data/train-images.idx3-ubyte",
        "data/train-labels.idx1-ubyte",
    )?;
    let test = MnistMmapDataset::open(
        "data/t10k-images.idx3-ubyte",
        "data/t10k-labels.idx1-ubyte",
    )?;

    let batch_size = 64usize;
    let ckpt = "checkpoints/mnist.mlckpt";

    let mut opt = Adam::with_schedule(LrSchedule::Constant(5e-4));
    opt.weight_decay = 1e-4;

    let mut session = AdaptSession::load_or_new(ckpt, |m| create_mnist_model(m, batch_size), opt)?;
    session.config = AdaptConfig {
        replay_ratio: 0.5,
        replay_capacity: 4096,
        grad_clip: Some(1.0),
        report_accuracy: true,
    };

    // If this is a fresh model (no prior steps), do a short warm-start on train data.
    if session.opt.global_step() == 0 {
        println!("no prior found — warm-starting on MNIST train…");
        let warm = session.adapt_dataset(&train, TrainBudget::steps(400).with_max_millis(30_000));
        println!(
            "warm-start: steps={} stop={:?} loss={:.4} acc≈{:.1}%",
            warm.steps_done,
            warm.stop,
            warm.mean_loss,
            warm.last_accuracy.unwrap_or(0.0) * 100.0
        );
        session.save(ckpt)?;
    } else {
        println!(
            "loaded prior from {ckpt} (opt step {})",
            session.opt.global_step()
        );
    }

    // Simulate "user data": a small local batch of test digits (on-device personalization).
    let mut lx = Tensor::new(&[32, 784]);
    let mut ly = Tensor::new(&[32, 10]);
    let local_ids: Vec<usize> = (0..32).collect();
    test.gather_batch(&local_ids, &mut lx, &mut ly);
    let local = AdaptBatch::new(lx, ly);

    println!(
        "adapting on {} local samples (replay mix {:.0}%)…",
        local.len(),
        session.config.replay_ratio * 100.0
    );

    let metrics = session.adapt(&local, TrainBudget::steps(50).with_max_millis(10_000));
    println!(
        "adapt: steps={} stop={:?} mean_loss={:.4} batch_acc≈{:.1}% replay={} elapsed={}ms",
        metrics.steps_done,
        metrics.stop,
        metrics.mean_loss,
        metrics.last_accuracy.unwrap_or(0.0) * 100.0,
        metrics.replay_size,
        metrics.elapsed_millis
    );

    session.save(ckpt)?;
    println!("saved {ckpt}");

    // Predict first local sample
    let mut x0 = Tensor::new(&[784]);
    x0.data
        .copy_from_slice(&local.x.data[..784]);
    let class = session.predict_class(&x0)?;
    let probs = session.predict(&x0)?;
    println!("predict class={class}  p[class]={:.4}", probs.data[class]);

    // Session totals accumulate across adapt calls
    println!(
        "session totals: steps={} samples={} last_loss={:.4}",
        session.totals.steps_done, session.totals.samples_seen, session.totals.last_loss
    );

    Ok(())
}

fn create_mnist_model(model: &mut ModelContext, batch: usize) {
    let input = model.create_var(&[batch, 784], MV_FLAG_INPUT);

    let w0 = model.create_var(&[784, 16], MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let w1 = model.create_var(&[16, 16], MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let w2 = model.create_var(&[16, 10], MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    let bound0 = (6.0f32 / (784 + 16) as f32).sqrt();
    let bound1 = (6.0f32 / (16 + 16) as f32).sqrt();
    let bound2 = (6.0f32 / (16 + 10) as f32).sqrt();

    let mut rng = ml_lib_rust::prng::Prng::new(0x853c49e6748fea9b, 0xda3e39cb94b95bdb);
    model.vars[w0].val_mut().fill_rand(&mut rng, -bound0, bound0);
    model.vars[w1].val_mut().fill_rand(&mut rng, -bound1, bound1);
    model.vars[w2].val_mut().fill_rand(&mut rng, -bound2, bound2);

    let b0 = model.create_var(&[16], MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let b1 = model.create_var(&[16], MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let b2 = model.create_var(&[10], MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    let z0 = model.matmul(input, w0, MV_FLAG_NONE).unwrap();
    let z0b = model.add(z0, b0, MV_FLAG_NONE).unwrap();
    let a0 = model.relu(z0b, MV_FLAG_NONE);

    let z1 = model.matmul(a0, w1, MV_FLAG_NONE).unwrap();
    let z1b = model.add(z1, b1, MV_FLAG_NONE).unwrap();
    let z1r = model.relu(z1b, MV_FLAG_NONE);
    let a1 = model.add(a0, z1r, MV_FLAG_NONE).unwrap();

    let z2 = model.matmul(a1, w2, MV_FLAG_NONE).unwrap();
    let z2b = model.add(z2, b2, MV_FLAG_NONE).unwrap();
    let output = model.softmax(z2b, MV_FLAG_OUTPUT);

    let y = model.create_var(&[batch, 10], MV_FLAG_DESIRED_OUTPUT);
    model.cross_entropy(y, output, MV_FLAG_COST).unwrap();
}
