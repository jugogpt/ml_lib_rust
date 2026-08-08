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
    train_lables: Matrix,
    test_images: Matrix, 
    test_labels: Matrix,

    epoch: u32,
    batch_size: u32, 
    learning_rate: f32,
}

fn main() -> std::io::Result<()> {
    let train_images = Matrix::load(60000, 784, "train_images.mat")?;
    let test_images = Matrix::load(10000, 784, "test_images.mat")?;

    let mut train_labels = Matrix::new(60000, 10);
    let mut test_labels = Matrix::new(10000, 10);


    {

        let train_labels_file: Matrix = Matrix::load(60000, 1, "train_labels.mat")?;
        let test_labels_file: Matrix = Matrix::load(10000, 1, "test_labels.mat")?;

        for i in 0..60000 {
            let num = train_labels_file.data[i] as usize;
            train_labels.data[i * 10 + num] = 1.0;
        }

        for i in 0..10000 {
            let num = test_labels_file.data[i] as usize;
            test_labels.data[i * 10 + num] = 1.0;
        }

    }


    draw_mnist_digit(&test_images.data);
    for i: usize 0..10 {
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

    print!("pre-training output")
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



