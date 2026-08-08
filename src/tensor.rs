#![allow(dead_code)]

use crate::prng::Prng;
use rayon::prelude::*;
use std::fs;
use std::io;
use std::path::Path;

fn read_be_u32(bytes: &[u8], offset: usize) -> u32 {
    u32::from_be_bytes([
        bytes[offset],
        bytes[offset + 1],
        bytes[offset + 2],
        bytes[offset + 3],
    ])
}

fn shape_numel(shape: &[usize]) -> usize {
    shape.iter().product()
}

/// Row-major N-D tensor of `f32`.
#[derive(Clone, Debug)]
pub struct Tensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn new(shape: &[usize]) -> Self {
        let n = shape_numel(shape);
        Tensor {
            shape: shape.to_vec(),
            data: vec![0.0; n],
        }
    }

    pub fn from_vec(shape: &[usize], data: Vec<f32>) -> Self {
        assert_eq!(shape_numel(shape), data.len());
        Tensor {
            shape: shape.to_vec(),
            data,
        }
    }

    pub fn zeros_like(t: &Tensor) -> Self {
        Tensor::new(&t.shape)
    }

    pub fn numel(&self) -> usize {
        self.data.len()
    }

    pub fn ndim(&self) -> usize {
        self.shape.len()
    }

    /// Leading axis length (batch / rows for 2D).
    pub fn rows(&self) -> usize {
        self.shape.first().copied().unwrap_or(1)
    }

    /// Trailing axis length (features / cols for 2D).
    pub fn cols(&self) -> usize {
        self.shape.last().copied().unwrap_or(1)
    }

    pub fn reshape(&mut self, shape: &[usize]) {
        assert_eq!(shape_numel(shape), self.numel());
        self.shape = shape.to_vec();
    }

    pub fn viewed(mut self, shape: &[usize]) -> Self {
        self.reshape(shape);
        self
    }

    pub fn copy_from(&mut self, src: &Tensor) -> bool {
        if self.shape != src.shape {
            return false;
        }
        self.data.copy_from_slice(&src.data);
        true
    }

    pub fn clear(&mut self) {
        self.data.fill(0.0);
    }

    pub fn fill(&mut self, x: f32) {
        self.data.fill(x);
    }

    pub fn sum(&self) -> f32 {
        self.data.iter().sum()
    }

    pub fn scale(&mut self, scale: f32) {
        for v in &mut self.data {
            *v *= scale;
        }
    }

    pub fn fill_rand(&mut self, rng: &mut Prng, lower: f32, upper: f32) {
        for v in self.data.iter_mut() {
            *v = rng.randf() * (upper - lower) + lower;
        }
    }

    /// Argmax over the last axis of a 1D/2D tensor. For 2D returns argmax of row 0
    /// unless you use `argmax_rows`.
    pub fn argmax(&self) -> usize {
        let mut max_i = 0;
        for i in 1..self.data.len() {
            if self.data[i] > self.data[max_i] {
                max_i = i;
            }
        }
        max_i
    }

    /// Per-row argmax for a 2D `[rows, cols]` tensor.
    pub fn argmax_rows(&self) -> Vec<usize> {
        assert_eq!(self.ndim(), 2);
        let (rows, cols) = (self.shape[0], self.shape[1]);
        let mut out = vec![0usize; rows];
        for r in 0..rows {
            let base = r * cols;
            let mut best = 0;
            for c in 1..cols {
                if self.data[base + c] > self.data[base + best] {
                    best = c;
                }
            }
            out[r] = best;
        }
        out
    }

    /// Load MNIST IDX images as `[N, H*W]` with pixels in `[0, 1]`.
    pub fn load_idx_images(filename: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(filename)?;
        if bytes.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IDX image file too short",
            ));
        }
        let magic = read_be_u32(&bytes, 0);
        let num_images = read_be_u32(&bytes, 4) as usize;
        let rows = read_be_u32(&bytes, 8) as usize;
        let cols = read_be_u32(&bytes, 12) as usize;
        if magic != 0x0000_0803 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected IDX image magic 0x00000803, got {magic:#010x}"),
            ));
        }
        let pixels = rows * cols;
        let need = 16 + num_images * pixels;
        if bytes.len() < need {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IDX image file truncated",
            ));
        }
        let mut data = vec![0.0f32; num_images * pixels];
        for i in 0..num_images * pixels {
            data[i] = bytes[16 + i] as f32 / 255.0;
        }
        Ok(Tensor::from_vec(&[num_images, pixels], data))
    }

    /// Load MNIST IDX images as NCHW `[N, 1, H, W]`.
    pub fn load_idx_images_nchw(filename: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(filename)?;
        if bytes.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IDX image file too short",
            ));
        }
        let magic = read_be_u32(&bytes, 0);
        let n = read_be_u32(&bytes, 4) as usize;
        let h = read_be_u32(&bytes, 8) as usize;
        let w = read_be_u32(&bytes, 12) as usize;
        if magic != 0x0000_0803 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected IDX image magic 0x00000803, got {magic:#010x}"),
            ));
        }
        let need = 16 + n * h * w;
        if bytes.len() < need {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IDX image file truncated",
            ));
        }
        let mut data = vec![0.0f32; n * h * w];
        for i in 0..n * h * w {
            data[i] = bytes[16 + i] as f32 / 255.0;
        }
        Ok(Tensor::from_vec(&[n, 1, h, w], data))
    }

    pub fn load_idx_labels(filename: impl AsRef<Path>) -> io::Result<Self> {
        let bytes = fs::read(filename)?;
        if bytes.len() < 8 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IDX label file too short",
            ));
        }
        let magic = read_be_u32(&bytes, 0);
        let n = read_be_u32(&bytes, 4) as usize;
        if magic != 0x0000_0801 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("expected IDX label magic 0x00000801, got {magic:#010x}"),
            ));
        }
        if bytes.len() < 8 + n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IDX label file truncated",
            ));
        }
        let mut data = vec![0.0f32; n];
        for i in 0..n {
            data[i] = bytes[8 + i] as f32;
        }
        Ok(Tensor::from_vec(&[n], data))
    }
}

// ---------------------------------------------------------------------------
// Elementwise / reductions
// ---------------------------------------------------------------------------

fn same_shape(a: &Tensor, b: &Tensor) -> bool {
    a.shape == b.shape
}

pub fn tensor_add(out: &mut Tensor, a: &Tensor, b: &Tensor) -> bool {
    if !same_shape(a, b) || out.shape != a.shape {
        // broadcast: a [B, F] + b [F] or b [1, F]
        return tensor_add_broadcast(out, a, b);
    }
    for i in 0..out.numel() {
        out.data[i] = a.data[i] + b.data[i];
    }
    true
}

fn tensor_add_broadcast(out: &mut Tensor, a: &Tensor, b: &Tensor) -> bool {
    // Support [B, F] + [F], [B, F] + [1, F], [B, F] + [F, 1] (legacy)
    if a.ndim() == 2 && out.shape == a.shape {
        let (batch, feat) = (a.shape[0], a.shape[1]);
        if b.shape == [feat] || b.shape == [1, feat] {
            for r in 0..batch {
                for c in 0..feat {
                    let bi = if b.ndim() == 1 { c } else { c };
                    out.data[r * feat + c] = a.data[r * feat + c] + b.data[bi];
                }
            }
            return true;
        }
        if b.shape == [feat, 1] {
            for r in 0..batch {
                for c in 0..feat {
                    out.data[r * feat + c] = a.data[r * feat + c] + b.data[c];
                }
            }
            return true;
        }
    }
    false
}

pub fn tensor_sub(out: &mut Tensor, a: &Tensor, b: &Tensor) -> bool {
    if !same_shape(a, b) || out.shape != a.shape {
        return false;
    }
    for i in 0..out.numel() {
        out.data[i] = a.data[i] - b.data[i];
    }
    true
}

pub fn tensor_relu(out: &mut Tensor, input: &Tensor) -> bool {
    if out.shape != input.shape {
        return false;
    }
    for i in 0..out.numel() {
        out.data[i] = input.data[i].max(0.0);
    }
    true
}

pub fn tensor_relu_add_grad(out: &mut Tensor, input: &Tensor, grad: &Tensor) -> bool {
    if out.shape != input.shape || out.shape != grad.shape {
        return false;
    }
    for i in 0..out.numel() {
        if input.data[i] > 0.0 {
            out.data[i] += grad.data[i];
        }
    }
    true
}

pub fn tensor_sigmoid(out: &mut Tensor, input: &Tensor) -> bool {
    if out.shape != input.shape {
        return false;
    }
    for i in 0..out.numel() {
        out.data[i] = 1.0 / (1.0 + (-input.data[i]).exp());
    }
    true
}

pub fn tensor_sigmoid_add_grad(out: &mut Tensor, sigmoid_out: &Tensor, grad: &Tensor) -> bool {
    if out.shape != sigmoid_out.shape || out.shape != grad.shape {
        return false;
    }
    for i in 0..out.numel() {
        let y = sigmoid_out.data[i];
        out.data[i] += grad.data[i] * y * (1.0 - y);
    }
    true
}

pub fn tensor_tanh(out: &mut Tensor, input: &Tensor) -> bool {
    if out.shape != input.shape {
        return false;
    }
    for i in 0..out.numel() {
        out.data[i] = input.data[i].tanh();
    }
    true
}

pub fn tensor_tanh_add_grad(out: &mut Tensor, tanh_out: &Tensor, grad: &Tensor) -> bool {
    if out.shape != tanh_out.shape || out.shape != grad.shape {
        return false;
    }
    for i in 0..out.numel() {
        let y = tanh_out.data[i];
        out.data[i] += grad.data[i] * (1.0 - y * y);
    }
    true
}

/// Approximate GELU (tanh form).
pub fn tensor_gelu(out: &mut Tensor, input: &Tensor) -> bool {
    if out.shape != input.shape {
        return false;
    }
    const C: f32 = 0.7978845608; // sqrt(2/pi)
    for i in 0..out.numel() {
        let x = input.data[i];
        let u = C * (x + 0.044715 * x * x * x);
        out.data[i] = 0.5 * x * (1.0 + u.tanh());
    }
    true
}

pub fn tensor_gelu_add_grad(out: &mut Tensor, input: &Tensor, grad: &Tensor) -> bool {
    if out.shape != input.shape || out.shape != grad.shape {
        return false;
    }
    const C: f32 = 0.7978845608;
    for i in 0..out.numel() {
        let x = input.data[i];
        let x2 = x * x;
        let x3 = x2 * x;
        let u = C * (x + 0.044715 * x3);
        let tanh_u = u.tanh();
        let sech2 = 1.0 - tanh_u * tanh_u;
        let du = C * (1.0 + 3.0 * 0.044715 * x2);
        let dgelu = 0.5 * (1.0 + tanh_u) + 0.5 * x * sech2 * du;
        out.data[i] += grad.data[i] * dgelu;
    }
    true
}

/// Softmax over the last axis. Works for `[C]` or `[B, C]`.
pub fn tensor_softmax(out: &mut Tensor, input: &Tensor) -> bool {
    if out.shape != input.shape {
        return false;
    }
    let cols = input.cols();
    let rows = input.numel() / cols;
    for r in 0..rows {
        let base = r * cols;
        let mut max_v = input.data[base];
        for c in 1..cols {
            max_v = max_v.max(input.data[base + c]);
        }
        let mut sum = 0.0f32;
        for c in 0..cols {
            let e = (input.data[base + c] - max_v).exp();
            out.data[base + c] = e;
            sum += e;
        }
        let inv = 1.0 / sum;
        for c in 0..cols {
            out.data[base + c] *= inv;
        }
    }
    true
}

pub fn tensor_softmax_add_grad(out: &mut Tensor, softmax_out: &Tensor, grad: &Tensor) -> bool {
    if out.shape != softmax_out.shape || out.shape != grad.shape {
        return false;
    }
    let cols = softmax_out.cols();
    let rows = softmax_out.numel() / cols;
    for r in 0..rows {
        let base = r * cols;
        let mut dot = 0.0f32;
        for c in 0..cols {
            dot += softmax_out.data[base + c] * grad.data[base + c];
        }
        for c in 0..cols {
            out.data[base + c] +=
                softmax_out.data[base + c] * (grad.data[base + c] - dot);
        }
    }
    true
}

pub fn tensor_cross_entropy(out: &mut Tensor, p: &Tensor, q: &Tensor) -> bool {
    if p.shape != q.shape || out.shape != p.shape {
        return false;
    }
    for i in 0..out.numel() {
        out.data[i] = if p.data[i] == 0.0 {
            0.0
        } else {
            p.data[i] * -q.data[i].clamp(1e-7, 1.0).ln()
        };
    }
    true
}

pub fn tensor_cross_entropy_add_grad(
    p_grad: Option<&mut Tensor>,
    q_grad: Option<&mut Tensor>,
    p: &Tensor,
    q: &Tensor,
    grad: &Tensor,
) -> bool {
    if p.shape != q.shape || p.shape != grad.shape {
        return false;
    }
    if let Some(pg) = p_grad {
        if pg.shape != p.shape {
            return false;
        }
        for i in 0..p.numel() {
            pg.data[i] += -q.data[i].clamp(1e-7, 1.0).ln() * grad.data[i];
        }
    }
    if let Some(qg) = q_grad {
        if qg.shape != q.shape {
            return false;
        }
        for i in 0..p.numel() {
            qg.data[i] += -p.data[i] / q.data[i].max(1e-7) * grad.data[i];
        }
    }
    true
}

/// LayerNorm over the last axis. `mean_inv` cache layout: `[rows * 2]` = mean | inv_std.
pub fn tensor_layer_norm(
    out: &mut Tensor,
    input: &Tensor,
    gamma: &Tensor,
    beta: &Tensor,
    eps: f32,
    mean_inv: &mut Tensor,
) -> bool {
    if out.shape != input.shape {
        return false;
    }
    let cols = input.cols();
    let rows = input.numel() / cols;
    if gamma.numel() != cols || beta.numel() != cols {
        return false;
    }
    if mean_inv.numel() < rows * 2 {
        *mean_inv = Tensor::new(&[rows * 2]);
    }
    for r in 0..rows {
        let base = r * cols;
        let mut mean = 0.0f32;
        for c in 0..cols {
            mean += input.data[base + c];
        }
        mean /= cols as f32;
        let mut var = 0.0f32;
        for c in 0..cols {
            let d = input.data[base + c] - mean;
            var += d * d;
        }
        var /= cols as f32;
        let inv = 1.0 / (var + eps).sqrt();
        mean_inv.data[r] = mean;
        mean_inv.data[rows + r] = inv;
        for c in 0..cols {
            let norm = (input.data[base + c] - mean) * inv;
            out.data[base + c] = norm * gamma.data[c] + beta.data[c];
        }
    }
    true
}

pub fn tensor_layer_norm_add_grad(
    x_grad: &mut Tensor,
    mut gamma_grad: Option<&mut Tensor>,
    mut beta_grad: Option<&mut Tensor>,
    input: &Tensor,
    gamma: &Tensor,
    grad: &Tensor,
    mean_inv: &Tensor,
) -> bool {
    let cols = input.cols();
    let rows = input.numel() / cols;
    for r in 0..rows {
        let base = r * cols;
        let mean = mean_inv.data[r];
        let inv = mean_inv.data[rows + r];
        let mut dnorm = vec![0.0f32; cols];
        for c in 0..cols {
            dnorm[c] = grad.data[base + c] * gamma.data[c];
            if let Some(ref mut bg) = beta_grad.as_deref_mut() {
                bg.data[c] += grad.data[base + c];
            }
            if let Some(ref mut gg) = gamma_grad.as_deref_mut() {
                let norm = (input.data[base + c] - mean) * inv;
                gg.data[c] += grad.data[base + c] * norm;
            }
        }
        let mut dmean = 0.0f32;
        let mut dvar_term = 0.0f32;
        for c in 0..cols {
            let xmu = input.data[base + c] - mean;
            dmean += dnorm[c] * (-inv);
            dvar_term += dnorm[c] * xmu;
        }
        // d(inv)/d(var) path folded into x grad
        let dvar = dvar_term * (-0.5) * inv * inv * inv / cols as f32;
        dmean /= cols as f32;
        // also mean depends on x
        let mut extra_mean = 0.0f32;
        for c in 0..cols {
            extra_mean += 2.0 * (input.data[base + c] - mean) * dvar;
        }
        dmean += extra_mean / cols as f32;

        for c in 0..cols {
            let xmu = input.data[base + c] - mean;
            x_grad.data[base + c] += dnorm[c] * inv + dvar * 2.0 * xmu + dmean;
        }
    }
    true
}

pub fn tensor_dropout(
    out: &mut Tensor,
    input: &Tensor,
    mask: &mut Tensor,
    p: f32,
    train: bool,
    rng: &mut Prng,
) -> bool {
    if out.shape != input.shape {
        return false;
    }
    if !train || p <= 0.0 {
        out.data.copy_from_slice(&input.data);
        mask.data.fill(1.0);
        return true;
    }
    if mask.shape != input.shape {
        *mask = Tensor::zeros_like(input);
    }
    let keep = 1.0 - p;
    let scale = 1.0 / keep;
    for i in 0..out.numel() {
        if rng.randf() < keep {
            mask.data[i] = scale;
            out.data[i] = input.data[i] * scale;
        } else {
            mask.data[i] = 0.0;
            out.data[i] = 0.0;
        }
    }
    true
}

pub fn tensor_dropout_add_grad(out: &mut Tensor, mask: &Tensor, grad: &Tensor) -> bool {
    if out.shape != mask.shape || out.shape != grad.shape {
        return false;
    }
    for i in 0..out.numel() {
        out.data[i] += grad.data[i] * mask.data[i];
    }
    true
}

// ---------------------------------------------------------------------------
// Fast matmul (rayon over rows; tight inner loop for autovec/SIMD)
// ---------------------------------------------------------------------------

/// GEMM: `out = A @ B` for 2D, or batched `out[b] = A[b] @ B` when A is `[B,M,K]` and B is `[K,N]`.
/// Also supports A `[B,K]` @ B `[K,N]` → `[B,N]`.
pub fn tensor_matmul(
    out: &mut Tensor,
    a: &Tensor,
    b: &Tensor,
    zero_out: bool,
    transpose_a: bool,
    transpose_b: bool,
) -> bool {
    // Normalize to 2D-batch form
    let (a_batch, a_rows, a_cols) = mat_shape(a, transpose_a);
    let (b_batch, b_rows, b_cols) = mat_shape(b, transpose_b);

    if a_cols != b_rows {
        return false;
    }
    let batch = a_batch.max(b_batch);
    if a_batch != 1 && b_batch != 1 && a_batch != b_batch {
        return false;
    }

    let expected = if batch == 1 {
        vec![a_rows, b_cols]
    } else if a.ndim() == 2 && !transpose_a && a.shape[0] == batch && a.shape[1] == a_cols {
        // [B,K] @ [K,N] -> [B,N]
        vec![batch, b_cols]
    } else {
        vec![batch, a_rows, b_cols]
    };

    // Accept [B, N] when a_rows == 1 folded, or exact expected
    if out.shape != expected
        && !(batch > 1 && out.shape == [batch, b_cols] && a_rows == 1)
        && !(batch > 1 && a.ndim() == 2 && out.shape == [batch, b_cols])
    {
        // allow out [B, a_rows, b_cols] already checked; also [B,N] for [B,K]@[K,N]
        if !(a.ndim() == 2 && out.shape == [a.shape[0], b_cols] && !transpose_a) {
            return false;
        }
    }

    if zero_out {
        out.clear();
    }

    let k_dim = a_cols;
    let n_dim = b_cols;
    let m_dim = if a.ndim() == 2 && out.ndim() == 2 && out.shape[0] == a.shape[0] && !transpose_a {
        // treat each row of A as a sample: effectively M=1 per batch row
        a.shape[0]
    } else {
        a_rows
    };

    if a.ndim() == 2 && b.ndim() == 2 && out.ndim() == 2 && !transpose_a && !transpose_b {
        // Standard / batched-row GEMM: A[M,K] @ B[K,N]
        gemm_nn(
            &mut out.data,
            &a.data,
            &b.data,
            a.shape[0],
            k_dim,
            n_dim,
            a.shape[1],
            b.shape[1],
            out.shape[1],
        );
        return true;
    }

    if a.ndim() == 2 && b.ndim() == 2 && out.ndim() == 2 && !transpose_a && transpose_b {
        gemm_nt(
            &mut out.data,
            &a.data,
            &b.data,
            a.shape[0],
            a.shape[1],
            b.shape[0],
            out.shape[1],
        );
        return true;
    }

    if a.ndim() == 2 && b.ndim() == 2 && out.ndim() == 2 && transpose_a && !transpose_b {
        gemm_tn(
            &mut out.data,
            &a.data,
            &b.data,
            a.shape[1],
            a.shape[0],
            b.shape[1],
            out.shape[1],
        );
        return true;
    }

    // Generic batched fallback
    for bi in 0..batch {
        let a_off = if a_batch == 1 {
            0
        } else {
            bi * a_rows * a_cols
        };
        let b_off = if b_batch == 1 {
            0
        } else {
            bi * b_rows * b_cols
        };
        let o_off = bi * a_rows * b_cols;
        for i in 0..a_rows {
            for k in 0..k_dim {
                let av = a.data[a_off
                    + if transpose_a {
                        i + k * a_rows
                    } else {
                        k + i * a_cols
                    }];
                for j in 0..n_dim {
                    let bv = b.data[b_off
                        + if transpose_b {
                            k + j * b_rows
                        } else {
                            j + k * b_cols
                        }];
                    out.data[o_off + j + i * n_dim] += av * bv;
                }
            }
        }
    }
    let _ = m_dim;
    true
}

fn mat_shape(t: &Tensor, transpose: bool) -> (usize, usize, usize) {
    match t.ndim() {
        1 => {
            if transpose {
                (1, 1, t.shape[0])
            } else {
                (1, t.shape[0], 1)
            }
        }
        2 => {
            let (r, c) = (t.shape[0], t.shape[1]);
            if transpose {
                (1, c, r)
            } else {
                (1, r, c)
            }
        }
        3 => {
            let (b, r, c) = (t.shape[0], t.shape[1], t.shape[2]);
            if transpose {
                (b, c, r)
            } else {
                (b, r, c)
            }
        }
        _ => (1, t.rows(), t.cols()),
    }
}

fn gemm_nn(
    out: &mut [f32],
    a: &[f32],
    b: &[f32],
    m: usize,
    k: usize,
    n: usize,
    lda: usize,
    ldb: usize,
    ldc: usize,
) {
    out.par_chunks_mut(ldc)
        .take(m)
        .enumerate()
        .for_each(|(i, out_row)| {
            for p in 0..k {
                let av = a[p + i * lda];
                let brow = &b[p * ldb..p * ldb + n];
                // tight loop — LLVM typically autovectorizes this
                for j in 0..n {
                    out_row[j] += av * brow[j];
                }
            }
        });
}

fn gemm_nt(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize, ldc: usize) {
    // A[M,K] @ B^T where B is [N,K] stored row-major → B^T is [K,N]
    out.par_chunks_mut(ldc)
        .take(m)
        .enumerate()
        .for_each(|(i, out_row)| {
            for j in 0..n {
                let mut s = out_row[j];
                for p in 0..k {
                    s += a[p + i * k] * b[p + j * k];
                }
                out_row[j] = s;
            }
        });
}

fn gemm_tn(out: &mut [f32], a: &[f32], b: &[f32], m: usize, k: usize, n: usize, ldc: usize) {
    // A^T @ B with A stored [K,M] → A^T is [M,K]
    out.par_chunks_mut(ldc)
        .take(m)
        .enumerate()
        .for_each(|(i, out_row)| {
            for p in 0..k {
                let av = a[i + p * m];
                for j in 0..n {
                    out_row[j] += av * b[j + p * n];
                }
            }
        });
}

// ---------------------------------------------------------------------------
// Conv2d / MaxPool / Embedding
// ---------------------------------------------------------------------------

fn conv_out_dim(in_dim: usize, kernel: usize, stride: usize, pad: usize) -> usize {
    (in_dim + 2 * pad - kernel) / stride + 1
}

/// NCHW conv2d. weight `[O, I, KH, KW]`, bias optional `[O]`.
pub fn tensor_conv2d(
    out: &mut Tensor,
    input: &Tensor,
    weight: &Tensor,
    bias: Option<&Tensor>,
    stride: usize,
    pad: usize,
) -> bool {
    if input.ndim() != 4 || weight.ndim() != 4 {
        return false;
    }
    let (n, ic, ih, iw) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let (oc, wic, kh, kw) = (
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    );
    if ic != wic {
        return false;
    }
    let oh = conv_out_dim(ih, kh, stride, pad);
    let ow = conv_out_dim(iw, kw, stride, pad);
    if out.shape != [n, oc, oh, ow] {
        return false;
    }
    out.clear();

    out.data
        .par_chunks_mut(oc * oh * ow)
        .take(n)
        .enumerate()
        .for_each(|(ni, out_n)| {
            for oc_i in 0..oc {
                for oh_i in 0..oh {
                    for ow_i in 0..ow {
                        let mut s = 0.0f32;
                        for ic_i in 0..ic {
                            for y in 0..kh {
                                let ih_i = oh_i as isize * stride as isize + y as isize - pad as isize;
                                if ih_i < 0 || ih_i >= ih as isize {
                                    continue;
                                }
                                for x in 0..kw {
                                    let iw_i = ow_i as isize * stride as isize + x as isize - pad as isize;
                                    if iw_i < 0 || iw_i >= iw as isize {
                                        continue;
                                    }
                                    let in_idx = ((ni * ic + ic_i) * ih + ih_i as usize) * iw
                                        + iw_i as usize;
                                    let w_idx =
                                        ((oc_i * ic + ic_i) * kh + y) * kw + x;
                                    s += input.data[in_idx] * weight.data[w_idx];
                                }
                            }
                        }
                        if let Some(b) = bias {
                            s += b.data[oc_i];
                        }
                        out_n[(oc_i * oh + oh_i) * ow + ow_i] = s;
                    }
                }
            }
        });
    true
}

pub fn tensor_conv2d_add_grad(
    input_grad: Option<&mut Tensor>,
    weight_grad: Option<&mut Tensor>,
    bias_grad: Option<&mut Tensor>,
    input: &Tensor,
    weight: &Tensor,
    grad: &Tensor,
    stride: usize,
    pad: usize,
) -> bool {
    let (n, ic, ih, iw) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let (oc, _, kh, kw) = (
        weight.shape[0],
        weight.shape[1],
        weight.shape[2],
        weight.shape[3],
    );
    let oh = grad.shape[2];
    let ow = grad.shape[3];

    if let Some(bg) = bias_grad {
        for ni in 0..n {
            for oc_i in 0..oc {
                for oh_i in 0..oh {
                    for ow_i in 0..ow {
                        bg.data[oc_i] +=
                            grad.data[((ni * oc + oc_i) * oh + oh_i) * ow + ow_i];
                    }
                }
            }
        }
    }

    if let Some(wg) = weight_grad {
        for ni in 0..n {
            for oc_i in 0..oc {
                for ic_i in 0..ic {
                    for y in 0..kh {
                        for x in 0..kw {
                            let mut s = 0.0f32;
                            for oh_i in 0..oh {
                                let ih_i = oh_i as isize * stride as isize + y as isize - pad as isize;
                                if ih_i < 0 || ih_i >= ih as isize {
                                    continue;
                                }
                                for ow_i in 0..ow {
                                    let iw_i = ow_i as isize * stride as isize + x as isize - pad as isize;
                                    if iw_i < 0 || iw_i >= iw as isize {
                                        continue;
                                    }
                                    let g = grad.data[((ni * oc + oc_i) * oh + oh_i) * ow + ow_i];
                                    let in_idx = ((ni * ic + ic_i) * ih + ih_i as usize) * iw
                                        + iw_i as usize;
                                    s += g * input.data[in_idx];
                                }
                            }
                            wg.data[((oc_i * ic + ic_i) * kh + y) * kw + x] += s;
                        }
                    }
                }
            }
        }
    }

    if let Some(ig) = input_grad {
        for ni in 0..n {
            for oc_i in 0..oc {
                for oh_i in 0..oh {
                    for ow_i in 0..ow {
                        let g = grad.data[((ni * oc + oc_i) * oh + oh_i) * ow + ow_i];
                        for ic_i in 0..ic {
                            for y in 0..kh {
                                let ih_i = oh_i as isize * stride as isize + y as isize - pad as isize;
                                if ih_i < 0 || ih_i >= ih as isize {
                                    continue;
                                }
                                for x in 0..kw {
                                    let iw_i = ow_i as isize * stride as isize + x as isize - pad as isize;
                                    if iw_i < 0 || iw_i >= iw as isize {
                                        continue;
                                    }
                                    let w =
                                        weight.data[((oc_i * ic + ic_i) * kh + y) * kw + x];
                                    let in_idx = ((ni * ic + ic_i) * ih + ih_i as usize) * iw
                                        + iw_i as usize;
                                    ig.data[in_idx] += g * w;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    true
}

/// Max-pool 2D over NCHW. `indices` stores flat input index per output element.
pub fn tensor_max_pool2d(
    out: &mut Tensor,
    input: &Tensor,
    indices: &mut Tensor,
    kernel: usize,
    stride: usize,
) -> bool {
    if input.ndim() != 4 {
        return false;
    }
    let (n, c, ih, iw) = (
        input.shape[0],
        input.shape[1],
        input.shape[2],
        input.shape[3],
    );
    let oh = (ih - kernel) / stride + 1;
    let ow = (iw - kernel) / stride + 1;
    if out.shape != [n, c, oh, ow] {
        return false;
    }
    if indices.shape != out.shape {
        *indices = Tensor::new(&out.shape);
    }
    for ni in 0..n {
        for ci in 0..c {
            for oh_i in 0..oh {
                for ow_i in 0..ow {
                    let mut best = f32::NEG_INFINITY;
                    let mut best_idx = 0usize;
                    for y in 0..kernel {
                        for x in 0..kernel {
                            let ih_i = oh_i * stride + y;
                            let iw_i = ow_i * stride + x;
                            let idx = ((ni * c + ci) * ih + ih_i) * iw + iw_i;
                            let v = input.data[idx];
                            if v > best {
                                best = v;
                                best_idx = idx;
                            }
                        }
                    }
                    let oidx = ((ni * c + ci) * oh + oh_i) * ow + ow_i;
                    out.data[oidx] = best;
                    indices.data[oidx] = best_idx as f32;
                }
            }
        }
    }
    true
}

pub fn tensor_max_pool2d_add_grad(
    input_grad: &mut Tensor,
    grad: &Tensor,
    indices: &Tensor,
) -> bool {
    for i in 0..grad.numel() {
        let idx = indices.data[i] as usize;
        input_grad.data[idx] += grad.data[i];
    }
    true
}

/// Embedding lookup. `indices` holds integer ids as f32, shape `[...]`.
/// weight `[V, D]` → out `[..., D]`.
pub fn tensor_embedding(out: &mut Tensor, indices: &Tensor, weight: &Tensor) -> bool {
    if weight.ndim() != 2 {
        return false;
    }
    let (v, d) = (weight.shape[0], weight.shape[1]);
    let n = indices.numel();
    let mut expected = indices.shape.clone();
    expected.push(d);
    if out.shape != expected {
        return false;
    }
    for i in 0..n {
        let id = indices.data[i] as usize;
        if id >= v {
            return false;
        }
        out.data[i * d..(i + 1) * d].copy_from_slice(&weight.data[id * d..(id + 1) * d]);
    }
    true
}

pub fn tensor_embedding_add_grad(
    weight_grad: &mut Tensor,
    indices: &Tensor,
    grad: &Tensor,
) -> bool {
    let d = weight_grad.shape[1];
    let n = indices.numel();
    for i in 0..n {
        let id = indices.data[i] as usize;
        for j in 0..d {
            weight_grad.data[id * d + j] += grad.data[i * d + j];
        }
    }
    true
}

/// Gather a contiguous batch of rows from a 2D tensor into `out` `[batch, cols]`.
pub fn gather_rows(out: &mut Tensor, src: &Tensor, indices: &[usize]) -> bool {
    if src.ndim() != 2 || out.ndim() != 2 {
        return false;
    }
    let cols = src.shape[1];
    if out.shape != [indices.len(), cols] {
        return false;
    }
    for (i, &row) in indices.iter().enumerate() {
        let src_off = row * cols;
        let dst_off = i * cols;
        out.data[dst_off..dst_off + cols].copy_from_slice(&src.data[src_off..src_off + cols]);
    }
    true
}
