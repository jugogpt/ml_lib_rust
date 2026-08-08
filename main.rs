mod matrix;
mod model;
mod prng;

use matrix::Matrix;
use model::{
    ModelContext, MV_FLAG_COST, MV_FLAG_DESIRED_OUTPUT, MV_FLAG_INPUT, MV_FLAG_NONE,
    MV_FLAG_OUTPUT, MV_FLAG_PARAMETER, MV_FLAG_REQUIRES_GRAD,
};


struct ModelTrainingDesc {
    train_images: Matrix,
    train_labels: Matrix,
    test_images: Matrix,
    test_labels: Matrix,

    epochs: u32,
    batch_size: u32,
    learning_rate: f32,
}

fn main() -> std::io::Result<()> {
    let train_images = Matrix::load_idx_images("data/train-images.idx3-ubyte")?;
    let test_images = Matrix::load_idx_images("data/t10k-images.idx3-ubyte")?;

    let train_labels_file = Matrix::load_idx_labels("data/train-labels.idx1-ubyte")?;
    let test_labels_file = Matrix::load_idx_labels("data/t10k-labels.idx1-ubyte")?;

    let mut train_labels = Matrix::new(train_labels_file.rows, 10);
    let mut test_labels = Matrix::new(test_labels_file.rows, 10);

    for i in 0..train_labels_file.rows {
        let num = train_labels_file.data[i] as usize;
        train_labels.data[i * 10 + num] = 1.0;
    }

    for i in 0..test_labels_file.rows {
        let num = test_labels_file.data[i] as usize;
        test_labels.data[i * 10 + num] = 1.0;
    }


    draw_mnist_digit(&test_images.data);
    for i in 0..10 {
        print!("{:.0} ", test_labels.data[i]);
    }

    println!("\n");

    let mut model: ModelContext = ModelContext::new();
    create_mnist_model(&mut model);
    model.compile();

    {
        let input_idx: usize = model.input.unwrap();
        model.vars[input_idx].val_mut().data[..784].copy_from_slice(&test_images.data[..784]);
    }
    model.feedforward(); // FFNN implemented in the model impl

    print!("pre-training output: ");
    {
        let output_idx = model.output.unwrap();
        let out_val = model.vars[output_idx].val();
        for i in 0..10 {
            print!("{:.2} ", out_val.data[i]);
        }
    }
    println!();

    let training_desc: ModelTrainingDesc = ModelTrainingDesc {
        train_images,
        train_labels,
        test_images,
        test_labels,
        epochs: 10,
        batch_size: 50,
        learning_rate: 0.01,
    };

    model_train(&mut model, &training_desc);

    {
        let input_idx: usize = model.input.unwrap();
        let img = &training_desc.test_images.data[..784];
        model.vars[input_idx].val_mut().data[..784].copy_from_slice(img);
    }
    model.feedforward();
    print!("post-training output: ");
    {
        let output_idx = model.output.unwrap();
        let out_val = model.vars[output_idx].val();
        for i in 0..10 {
            print!("{:.6} ", out_val.data[i]);
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

fn create_mnist_model(model: &mut ModelContext) {
    let input = model.create_var(784, 1, MV_FLAG_INPUT);
    //layers
    let w0 = model.create_var(16, 784, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let w1 = model.create_var(16, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let w2 = model.create_var(10, 16, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    let bound0 = (6.0f32 / (784 + 16) as f32).sqrt();
    let bound1 = (6.0f32 / (16 + 16) as f32).sqrt();
    let bound2 = (6.0f32 / (16 + 10) as f32).sqrt();

    let mut rng = prng::Prng::new(0x853c49e6748fea9b, 0xda3e39cb94b95bdb);
    model.vars[w0].val_mut().fill_rand(&mut rng, -bound0, bound0);
    model.vars[w1].val_mut().fill_rand(&mut rng, -bound1, bound1);
    model.vars[w2].val_mut().fill_rand(&mut rng, -bound2, bound2);

    let b0 = model.create_var(16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let b1 = model.create_var(16, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);
    let b2 = model.create_var(10, 1, MV_FLAG_REQUIRES_GRAD | MV_FLAG_PARAMETER);

    let z0_a = model.matmul(w0, input, MV_FLAG_NONE).unwrap();
    let z0_b = model.add(z0_a, b0, MV_FLAG_NONE).unwrap();
    let a0 = model.relu(z0_b, MV_FLAG_NONE);


    let z1_a = model.matmul(w1, a0, MV_FLAG_NONE).unwrap();
    let z1_b = model.add(z1_a, b1, MV_FLAG_NONE).unwrap();
    let z1_c = model.relu(z1_b, MV_FLAG_NONE);
    let a1 = model.add(a0, z1_c, MV_FLAG_NONE).unwrap();

    let z2_a = model.matmul(w2, a1, MV_FLAG_NONE).unwrap();
    let z2_b = model.add(z2_a, b2, MV_FLAG_NONE).unwrap();
    let output = model.softmax(z2_b, MV_FLAG_OUTPUT);

    let y = model.create_var(10, 1, MV_FLAG_DESIRED_OUTPUT);

    model.cross_entropy(y, output, MV_FLAG_COST).unwrap();
}


fn model_train(model: &mut ModelContext, training_desc: &ModelTrainingDesc) {
    let train_images = &training_desc.train_images;
    let train_labels = &training_desc.train_labels;
    let test_images = &training_desc.test_images;
    let test_labels = &training_desc.test_labels;

    let num_examples = train_images.rows;
    let input_size = train_images.cols;
    let output_size = train_labels.cols;
    let num_tests = test_images.rows;

    let num_batches = num_examples / training_desc.batch_size as usize;

    let mut training_order: Vec<u32> = (0..num_examples as u32).collect();

    let input_idx = model.input.unwrap();
    let desired_output_idx = model.desired_output.unwrap();
    let output_idx = model.output.unwrap();
    let cost_idx = model.cost.unwrap();

    for epoch in 0..training_desc.epochs {
        for _ in 0..num_examples {
            let a = (prng::rand() as usize) % num_examples;
            let b = (prng::rand() as usize) % num_examples;
            training_order.swap(a, b);
        }

        for batch in 0..num_batches {
            for &i in &model.cost_prog.vars {
                let cur = &model.vars[i];
                if cur.flags & MV_FLAG_PARAMETER != 0 {
                    cur.grad_mut().clear();
                }
            }

            let mut avg_cost = 0.0f32;
            for i in 0..training_desc.batch_size as usize {
                let order_index = batch * training_desc.batch_size as usize + i;
                let index = training_order[order_index] as usize;

                {
                    let src = &train_images.data[index * input_size..(index + 1) * input_size];
                    model.vars[input_idx].val_mut().data[..input_size].copy_from_slice(src);
                }
                {
                    let src =
                        &train_labels.data[index * output_size..(index + 1) * output_size];
                    model.vars[desired_output_idx].val_mut().data[..output_size]
                        .copy_from_slice(src);
                }

                model.prog_compute(&model.cost_prog);
                model.prog_compute_grads(&model.cost_prog);

                avg_cost += model.vars[cost_idx].val().sum();
            }
            avg_cost /= training_desc.batch_size as f32;

            for &i in &model.cost_prog.vars {
                let cur = &model.vars[i];
                if cur.flags & MV_FLAG_PARAMETER == 0 {
                    continue;
                }

                let mut grad = cur.grad_mut();
                grad.scale(training_desc.learning_rate / training_desc.batch_size as f32);

                let mut val = cur.val_mut();
                for k in 0..val.data.len() {
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

        let mut num_correct = 0u32;
        let mut avg_cost = 0.0f32;
        for i in 0..num_tests {
            {
                let src = &test_images.data[i * input_size..(i + 1) * input_size];
                model.vars[input_idx].val_mut().data[..input_size].copy_from_slice(src);
            }
            {
                let src = &test_labels.data[i * output_size..(i + 1) * output_size];
                model.vars[desired_output_idx].val_mut().data[..output_size]
                    .copy_from_slice(src);
            }

            model.prog_compute(&model.cost_prog);

            avg_cost += model.vars[cost_idx].val().sum();
            num_correct += (model.vars[output_idx].val().argmax()
                == model.vars[desired_output_idx].val().argmax()) as u32;
        }

        avg_cost /= num_tests as f32;
        println!(
            "Test Completed. Accuracy: {:5} / {:5} ({:.1}%), Average Cost: {:.4}",
            num_correct,
            num_tests,
            num_correct as f32 / num_tests as f32 * 100.0,
            avg_cost
        );
    }
}




