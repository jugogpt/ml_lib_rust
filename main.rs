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

    




}