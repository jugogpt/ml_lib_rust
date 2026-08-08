//! Full MNIST prior training via [`AdaptSession`].
//!
//! ```text
//! cargo run --release --example mnist
//! ```

use ml_lib_rust::{
    AdaptSession, Adam, LrSchedule, MnistMmapDataset, ModelContext, Tensor, TrainBudget,
    MV_FLAG_COST, MV_FLAG_DESIRED_OUTPUT, MV_FLAG_INPUT, MV_FLAG_NONE, MV_FLAG_OUTPUT,
    MV_FLAG_PARAMETER, MV_FLAG_REQUIRES_GRAD,
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

    {
        let mut x = Tensor::new(&[1, 784]);
        let mut y = Tensor::new(&[1, 10]);
        test.gather_batch(&[0], &mut x, &mut y);
        draw_mnist_digit(&x.data);
        for i in 0..10 {
            print!("{:.0} ", y.data[i]);
        }
        println!("\n");
    }

    let batch_size = 64usize;
    let ckpt = "checkpoints/mnist.mlckpt";

    let mut opt = Adam::with_schedule(LrSchedule::Cosine {
        initial: 1e-3,
        min: 1e-5,
        total_steps: 10_000,
    });
    opt.weight_decay = 1e-4;

    let mut session = AdaptSession::load_or_new(ckpt, |m| create_mnist_model(m, batch_size), opt)?;

    // Preview before this run's training
    {
        let mut x = Tensor::new(&[784]);
        let mut row = Tensor::new(&[1, 784]);
        let mut y = Tensor::new(&[1, 10]);
        test.gather_batch(&[0], &mut row, &mut y);
        x.data.copy_from_slice(&row.data);
        let probs = session.predict(&x)?;
        print!("pre-adapt output: ");
        for i in 0..10 {
            print!("{:.2} ", probs.data[i]);
        }
        println!();
    }

    let budget = TrainBudget::steps(937 * 5)
        .with_max_millis(120_000)
        .with_max_bytes(64 * 1024 * 1024);

    println!("training prior with AdaptSession::adapt_dataset…");
    let metrics = session.adapt_dataset(&train, budget);
    println!(
        "stopped after {} steps ({:?}), train-batch acc≈{:.1}%, mean_loss={:.4}",
        metrics.steps_done,
        metrics.stop,
        metrics.last_accuracy.unwrap_or(0.0) * 100.0,
        metrics.mean_loss
    );

    session.save(ckpt)?;
    println!("saved {ckpt}");

    {
        let mut x = Tensor::new(&[784]);
        let mut row = Tensor::new(&[1, 784]);
        let mut y = Tensor::new(&[1, 10]);
        test.gather_batch(&[0], &mut row, &mut y);
        x.data.copy_from_slice(&row.data);
        let class = session.predict_class(&x)?;
        let probs = session.predict(&x)?;
        print!("post-adapt output: ");
        for i in 0..10 {
            print!("{:.6} ", probs.data[i]);
        }
        println!("\npredicted class={class} (label one-hot argmax={})", y.argmax_rows()[0]);
    }

    Ok(())
}

fn draw_mnist_digit(data: &[f32]) {
    for y in 0..28 {
        for x in 0..28 {
            let num = data[x + y * 28];
            let col = 232 + (num * 23.0) as i32;
            print!("\x1b[48;5;{}m  ", col);
        }
        println!();
    }
    print!("\x1b[0m");
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
