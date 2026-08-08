mod model;
mod prng;
mod tensor;

use model::{
    ModelContext, MV_FLAG_COST, MV_FLAG_DESIRED_OUTPUT, MV_FLAG_INPUT, MV_FLAG_NONE,
    MV_FLAG_OUTPUT, MV_FLAG_PARAMETER, MV_FLAG_REQUIRES_GRAD,
};
use tensor::{gather_rows, Tensor};

struct ModelTrainingDesc {
    train_images: Tensor,
    train_labels: Tensor,
    test_images: Tensor,
    test_labels: Tensor,
    epochs: u32,
    batch_size: usize,
    learning_rate: f32,
}

fn main() -> std::io::Result<()> {
    let train_images = Tensor::load_idx_images("data/train-images.idx3-ubyte")?;
    let test_images = Tensor::load_idx_images("data/t10k-images.idx3-ubyte")?;
    let train_labels_file = Tensor::load_idx_labels("data/train-labels.idx1-ubyte")?;
    let test_labels_file = Tensor::load_idx_labels("data/t10k-labels.idx1-ubyte")?;

    let mut train_labels = Tensor::new(&[train_labels_file.rows(), 10]);
    let mut test_labels = Tensor::new(&[test_labels_file.rows(), 10]);

    for i in 0..train_labels_file.numel() {
        let num = train_labels_file.data[i] as usize;
        train_labels.data[i * 10 + num] = 1.0;
    }
    for i in 0..test_labels_file.numel() {
        let num = test_labels_file.data[i] as usize;
        test_labels.data[i * 10 + num] = 1.0;
    }

    draw_mnist_digit(&test_images.data[..784]);
    for i in 0..10 {
        print!("{:.0} ", test_labels.data[i]);
    }
    println!("\n");

    let batch_size = 64usize;
    let mut model = ModelContext::new();
    create_mnist_model(&mut model, batch_size);
    model.compile();

    // Single-sample preview via a batch of 1 in the first row
    {
        let input_idx = model.input.unwrap();
        let mut inp = model.vars[input_idx].val_mut();
        inp.clear();
        inp.data[..784].copy_from_slice(&test_images.data[..784]);
    }
    model.training = false;
    model.feedforward();
    print!("pre-training output: ");
    {
        let out = model.vars[model.output.unwrap()].val();
        for i in 0..10 {
            print!("{:.2} ", out.data[i]);
        }
    }
    println!();

    let training_desc = ModelTrainingDesc {
        train_images,
        train_labels,
        test_images,
        test_labels,
        epochs: 10,
        batch_size,
        learning_rate: 0.1,
    };

    model.training = true;
    model_train(&mut model, &training_desc);

    {
        let input_idx = model.input.unwrap();
        let mut inp = model.vars[input_idx].val_mut();
        inp.clear();
        inp.data[..784].copy_from_slice(&training_desc.test_images.data[..784]);
    }
    model.training = false;
    model.feedforward();
    print!("post-training output: ");
    {
        let out = model.vars[model.output.unwrap()].val();
        for i in 0..10 {
            print!("{:.6} ", out.data[i]);
        }
    }
    println!("\n");

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

/// Batched MLP: x[B,784] @ W[784,H] → ... → softmax[B,10]
fn create_mnist_model(model: &mut ModelContext, batch: usize) {
    let input = model.create_var(&[batch, 784], MV_FLAG_INPUT);

    let w0 = model.create_var(&[784, 16], MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let w1 = model.create_var(&[16, 16], MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let w2 = model.create_var(&[16, 10], MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    let bound0 = (6.0f32 / (784 + 16) as f32).sqrt();
    let bound1 = (6.0f32 / (16 + 16) as f32).sqrt();
    let bound2 = (6.0f32 / (16 + 10) as f32).sqrt();

    let mut rng = prng::Prng::new(0x853c49e6748fea9b, 0xda3e39cb94b95bdb);
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
    let a1 = model.add(a0, z1r, MV_FLAG_NONE).unwrap(); // residual

    let z2 = model.matmul(a1, w2, MV_FLAG_NONE).unwrap();
    let z2b = model.add(z2, b2, MV_FLAG_NONE).unwrap();
    let output = model.softmax(z2b, MV_FLAG_OUTPUT);

    let y = model.create_var(&[batch, 10], MV_FLAG_DESIRED_OUTPUT);
    model.cross_entropy(y, output, MV_FLAG_COST).unwrap();
}

fn model_train(model: &mut ModelContext, training_desc: &ModelTrainingDesc) {
    let train_images = &training_desc.train_images;
    let train_labels = &training_desc.train_labels;
    let test_images = &training_desc.test_images;
    let test_labels = &training_desc.test_labels;

    let num_examples = train_images.rows();
    let num_tests = test_images.rows();
    let batch_size = training_desc.batch_size;
    let num_batches = num_examples / batch_size;

    let mut training_order: Vec<usize> = (0..num_examples).collect();

    let input_idx = model.input.unwrap();
    let desired_output_idx = model.desired_output.unwrap();
    let output_idx = model.output.unwrap();
    let cost_idx = model.cost.unwrap();

    let mut batch_x = Tensor::new(&[batch_size, train_images.cols()]);
    let mut batch_y = Tensor::new(&[batch_size, train_labels.cols()]);
    let mut batch_idx = vec![0usize; batch_size];

    for epoch in 0..training_desc.epochs {
        for i in (1..num_examples).rev() {
            let j = (prng::rand() as usize) % (i + 1);
            training_order.swap(i, j);
        }

        for batch in 0..num_batches {
            for &i in &model.cost_prog.vars {
                let cur = &model.vars[i];
                if cur.flags & MV_FLAG_PARAMETER != 0 {
                    cur.grad_mut().clear();
                }
            }

            let start = batch * batch_size;
            batch_idx.copy_from_slice(&training_order[start..start + batch_size]);
            gather_rows(&mut batch_x, train_images, &batch_idx);
            gather_rows(&mut batch_y, train_labels, &batch_idx);

            model.vars[input_idx].val_mut().copy_from(&batch_x);
            model.vars[desired_output_idx].val_mut().copy_from(&batch_y);

            // One forward + backward over the whole batch
            model.prog_compute(&model.cost_prog);
            model.prog_compute_grads(&model.cost_prog);

            let avg_cost = model.vars[cost_idx].val().sum() / batch_size as f32;

            for &i in &model.cost_prog.vars {
                let cur = &model.vars[i];
                if cur.flags & MV_FLAG_PARAMETER == 0 {
                    continue;
                }
                let mut grad = cur.grad_mut();
                grad.scale(training_desc.learning_rate);
                let mut val = cur.val_mut();
                for k in 0..val.numel() {
                    val.data[k] -= grad.data[k];
                }
            }

            print!(
                "Epoch {:2} / {:2}, Batch {:4} / {:4}, Average Cost: {:.4}\r",
                epoch + 1,
                training_desc.epochs,
                batch + 1,
                num_batches,
                avg_cost
            );
            use std::io::Write;
            std::io::stdout().flush().ok();
        }
        println!();

        // Batched evaluation
        model.training = false;
        let mut num_correct = 0u32;
        let mut avg_cost = 0.0f32;
        let test_batches = num_tests / batch_size;
        for batch in 0..test_batches {
            let start = batch * batch_size;
            for i in 0..batch_size {
                batch_idx[i] = start + i;
            }
            gather_rows(&mut batch_x, test_images, &batch_idx);
            gather_rows(&mut batch_y, test_labels, &batch_idx);
            model.vars[input_idx].val_mut().copy_from(&batch_x);
            model.vars[desired_output_idx].val_mut().copy_from(&batch_y);
            model.prog_compute(&model.cost_prog);

            avg_cost += model.vars[cost_idx].val().sum();
            let preds = model.vars[output_idx].val().argmax_rows();
            let labels = model.vars[desired_output_idx].val().argmax_rows();
            for i in 0..batch_size {
                if preds[i] == labels[i] {
                    num_correct += 1;
                }
            }
        }
        model.training = true;

        let evaluated = (test_batches * batch_size) as f32;
        avg_cost /= evaluated;
        println!(
            "Test Completed. Accuracy: {:5} / {:5} ({:.1}%), Average Cost: {:.4}",
            num_correct,
            test_batches * batch_size,
            num_correct as f32 / evaluated * 100.0,
            avg_cost
        );
    }
}
