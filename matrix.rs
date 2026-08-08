use crate::prng::Prng;
use std::fs;
use std::io;

#[derive(Clone)]
pub struct Matrix {
    pub rows: usize, 
    pub cols: usize, 
    pub data: Vec<f32>
}

impl Matrix {
    pub fn new(rows: usize, cols: usize) -> Self{
        Matrix {
            rows,
            cols,

            // data is a vector that represents a 2d matrix
            data: vec![0.0; rows * cols], 
        }
    }

    //we need a function to copy a matrix, a function to fill a matrix, and a function to clear a matrix 

    pub fn load(rows: usize, cols: usize, filename: &str) -> io::Result<Self> {

        let mut mat: Matrix = Matrix::new(rows, cols);
        let bytes: Vec<u8> = fs::read(path filename)?; // all bytes
        let want_bytes: usize = rows * cols * 4; //bytes per area, 4 bytes per float cell in 2D matrix 
        let take_bytes: usize = bytes.len().min(want_bytes)
        let take_floats: usize = take_bytes / 4; // number of floats in the matrix 

        for i: usize in 0..take_floats {
            let b: [u8; 4] = [
                bytes[i * 4],
                bytes[i * 4 + 1],
                bytes[i * 4  + 2],
                bytes[i * 4 + 3],

            ];
            mat.data[i] = f32::from_le_bytes(b);
        }

        Ok(mat) 
    }

    pub fn copy_from(&mut self, src: &Matrix) -> bool { 
        //we want to make a copy of the matrix struct that was passed into src
        if self.rows != src.rows || self.cols != src.cols {
            return false;
        }

        self.data.copy_from_slice(&src.data);
        true
    }


    pub fn clear(&mut self) {
        self.data.iter_mut().for_each(|v: &mut f32| *v = 0.0) // set the value of each v in self.data *v = 0.0, i.e. to zero
    }

    // fill the matrix 
    pub fn fill (&mut self, x: f32) {
        self.data.iter_mut().for_each(|v: &mut f32| *v = x)
    }


    //return the sum of all of the values in the matrix 
    pub fn sum (&self) -> f32 {
        self.data.iter().sum()
    }


    pub fn fill_rand (&mut self, rng: &mut Prng, lower: f32, upper: f32, upper: f32) {
        for v: &mut f32 in self.data.iter_mut() {
            *v = rng.randf() * (upper - lower) + lower;
        }
    }
    
    pub fn scale(&mut self, scale: f32) {
        self.data.iter_mut().for_each(|v: &mut f32| *v *= scale) // *v is the value, not the refernece of v
    }

    pub fn argmax(&self) -> usize {
        let mut max_i: usize = 0 //max index, default is 0, change if we find self.data[i] > self.data[max_i]

        for i: usize in 1..self.data.len(){ 
            if self.data[i] > self.data[max_i] {
                max_i = i;
            }
        }
        max_i
    }

}

//the below functions are for the purpose of matrix operations for ML


//input two matrices and add their values in place injectively
pub fn mat_add(out: &mut Matrix, a: &Matrix, b: &Matrix) -> bool {

    //if their dimensions do not line up, then you need to reject 
    // if a.rows != b.rows || a.cols != b.cols -> return false
    if a.rows != b.rows || a.cols != b.cols {
        return false;
    }
    if out.rows != a.rows || out.cols != a.cols {
        return false;
    }

    //update some given mutable matrix reference out as the addition of matrix references a, b 
    for i: usize in 0..out.data.len() {
        out.data[i] = a.data[i] + b.data[i];
    }

    true


}

//editing out so mutable
pub fn mat_sub(out: &mut Matrix, a: &Matrix, b: &Matrix) -> {
    // if dimensions do not match up, then we 
    if a.rows != b.rows || a.cols != b.cols {
        return false
    }
    if out.rows != a.rows || out.cols != a.cols {
        return false
    }

    for i in 0..out.data.len() {
        out.data[i] = a.data[i] - b.data[i];
    }

    true 
}


pub fn mat_mul_nn(out: &mut Matix, a: &Matrix, b: &Matrix) {
    for i: usize in 0..out.rows {
        for k: usize in 0..a.cols {
            for j: usize in 0..out.cols {
                out.data[j + i * out.cols] += a.data[k + i*a.cols] * b.data[j + k*b.cols];
            }
        }
    }
}

pub fn mat_mul_nt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i: usize in 0..out.rows {
        for j: usize in 0..out.cols {
            for k: usize in 0..a.cols {
                out.data[j + i * out.cols] += a.data[i + k*a.cols] * b.data[k + j*b.cols];
            }
        }
    }
}

pub fn mat_mul_tn(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for k: usize in 0..a.rows {
        for i: usize in 0..out.cols {
            for j: usize in 0..out.cols {
                out.data[j + i * out.cols] += a.data[i + k * a.cols] * b.data[j + k * b.cols];
            }
        }
    }
}

pub fn mat_mul_tt(out: &mut Matrix, a: &Matrix, b: &Matrix) {
    for i: usize in 0..out.rows {
        for j: usize in 0..out.cols {
            for k: usize in 0..a.rows {
                out.data[j + i * out.cols] += a.data[i + k * a.cols] * b.data[k + j * b.cols];
            }
        }
    }
    
}

// out = a @ b (optionally transposing a and/or b first). 
// If 'zero_out' is true, 'out' is cleared before accumulating.
pub fn mat_mul(
    out: &mut Matrix, 
    a: &Matrix,
    b: &Matrix,
    zero_out: bool,
    transpose_a: bool,
    transpose_b: bool,
) -> bool {
    let a_rows = if transpose_a { a.cols } else { a.rows };
    let a_cols = if transpose_a { a.rows } else { a.cols };
    let b_rows = if transpose_b { b.cols } else { b.rows };
    let b_cols = if transpose_b { b.rows } else { b.cols };


    if a_cols != b_rows {
        return false;
    }

    if out.rows != a_rows || out.cols != b_cols {
        return false;
    }

    if zero_out {
        out.clear();
    }

    match (transpose_a, transpose_b) {
        (false, false) => mat_mul_nn(out, a, b),
        (false, true) => mat_mul_nt(out, a, b),
        (true, false) => mat_mul_tn(out, a, b),
        (true, true) => mat_mul_tt(out, a, b),

    }

    true 
}


pub fn mat_relu(out: &mut Matrix, input: &Matrix) {
    //base case, if the dimensions don't align, then do not run 
    if out.rows != input.rows || out.cols != input.cols { 
        return false;
    }

    for i: usize in 0..out.data.len() {
        out.data[i] = input.data[i].max(0.0); //in short, set the negatives to 0
    }

    true

}

pub fn mat_softmax(out: &mut Matrix, input: &Matrix) {
    if out.rows != input.rows || out.cols != input.cols {
        return false;
    }

    let mut sum: f32 = 0.0f32;
    //norm by the .exp() sum of matrix elements
    for i: usize in 0..out.data.len() { // for each index ,
        out.data[i] = input.data[i].exp(); // classic softmax sum calculation
        sum += out.data[i];
    }

    out.scale(1.0 / sum);

    true
}


pub fn mat_cross_entropy(out: &mut Matrix, p: &Matrix, q: &Matrix) -> bool {
    //check if all dimensions match
    if p.rows != q.rows || p.cols != q.cols {
        return false;
    }

    if out.rows != p.rows || out.cols != p.cols {
        return false;
    }

    for i: usize in 0..out.data.len() {
        out.data[i] = if p.data[i] == 0.0 {
            0.0
        } else {
            p.data[i] * -q.data[i].ln()
        };
    }

    true

}


pub fn mat_relu_add_grad(out: &mut Matrix, input: &Matrix grad: &Matrix) -> bool {
    if out.rows != input.rows || out.cols != input.cols {
        return false;
    }
    if out.rows != grad.rows || out.cols != grad.cols {
        return false;
    } 

    for i: usize in 0..out.data.len() { //only set gradient if the index has not died out
        out.data[i] += if input.data[i] > 0.0 { grad.data[i] } else { 0.0 };
    }

    true 
}


pub fn mat_softmax_add_grad(out: &mut Matrix, softmax_out: &Matrix, grad: &Matrix) -> bool {
    if softmax_out.rows != 1 && softmax_out.cols != 1 {
        return false;
    }

    let size = softmax_out.rows.max(softmax_out.cols);
    let mut jacobian = Matrix::new(size, size);

    for i in 0..size {
        for j in 0..size {
            let delta = if i == j { 1.0 } else { 0.0 };
            jacobian.data[j + i * size] = softmax_out.data[i] * (delta - softmax_out.data[j]);
        }
    }

    mat_mul(out, &jacobian, grad, false, false, false);

    true
}

pub fn mat_cross_entropy_add_grad(
    p_grad: Option<&mut Matrix>,
    q_grad: Option<&mut Matrix>,
    p: &Matrix,
    q: &Matrix,
    grad: &Matrix,
) -> bool {
    if p.rows != q.rows || p.cols != q.cols {
        return false;
    }

    let size = p.rows * p.cols;

    if let Some(p_grad) = p_grad {
        if p_grad.rows != p.rows || p_grad.cols != p.cols {
            return false;
        }

        for i in 0..size {
            p_grad.data[i] += -q.data[i].ln() * grad.data[i];
        }
    }

    if let Some(q_grad) = q_grad {
        if q_grad.rows != q.rows || q_grad.cols != q.cols {
            return false;
        }

        for i in 0..size {
            q_grad.data[i] += -p.data[i] / q.data[i] * grad.data[i];
        }
    }

    true
}



