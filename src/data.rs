use crate::tensor::{gather_rows, Tensor};
use memmap2::Mmap;
use std::fs::File;
use std::io;
use std::path::Path;

/// Minimal dataset interface for on-device minibatching.
pub trait Dataset {
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Write sample `index` into caller-provided row buffers (`x`: features, `y`: target).
    fn get(&self, index: usize, x: &mut Tensor, y: &mut Tensor) -> bool;

    /// Feature / label trailing shapes for buffer allocation (excluding batch).
    fn x_cols(&self) -> usize;
    fn y_cols(&self) -> usize;
}

/// In-memory pair of 2D tensors `[N, Fx]` / `[N, Fy]`.
pub struct TensorDataset {
    pub x: Tensor,
    pub y: Tensor,
}

impl TensorDataset {
    pub fn new(x: Tensor, y: Tensor) -> Self {
        assert_eq!(x.ndim(), 2);
        assert_eq!(y.ndim(), 2);
        assert_eq!(x.shape[0], y.shape[0]);
        Self { x, y }
    }

    pub fn gather_batch(&self, indices: &[usize], bx: &mut Tensor, by: &mut Tensor) -> bool {
        gather_rows(bx, &self.x, indices) && gather_rows(by, &self.y, indices)
    }
}

impl Dataset for TensorDataset {
    fn len(&self) -> usize {
        self.x.rows()
    }

    fn get(&self, index: usize, x: &mut Tensor, y: &mut Tensor) -> bool {
        if index >= self.len() {
            return false;
        }
        gather_rows(x, &self.x, &[index]) && gather_rows(y, &self.y, &[index])
    }

    fn x_cols(&self) -> usize {
        self.x.cols()
    }

    fn y_cols(&self) -> usize {
        self.y.cols()
    }
}

/// Memory-mapped MNIST images (u8 IDX) + in-memory one-hot labels.
/// Converts pixels to f32 on gather — keeps RAM low on device.
pub struct MnistMmapDataset {
    images: Mmap,
    n: usize,
    rows: usize,
    cols: usize,
    labels: Tensor, // [N, 10] one-hot
}

impl MnistMmapDataset {
    pub fn open(
        images_path: impl AsRef<Path>,
        labels_path: impl AsRef<Path>,
    ) -> io::Result<Self> {
        let file = File::open(images_path)?;
        let images = unsafe { Mmap::map(&file)? };
        if images.len() < 16 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IDX image file too short",
            ));
        }
        let magic = u32::from_be_bytes(images[0..4].try_into().unwrap());
        let n = u32::from_be_bytes(images[4..8].try_into().unwrap()) as usize;
        let rows = u32::from_be_bytes(images[8..12].try_into().unwrap()) as usize;
        let cols = u32::from_be_bytes(images[12..16].try_into().unwrap()) as usize;
        if magic != 0x0000_0803 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "bad IDX image magic",
            ));
        }
        let need = 16 + n * rows * cols;
        if images.len() < need {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "IDX image truncated",
            ));
        }

        let label_ids = Tensor::load_idx_labels(labels_path)?;
        if label_ids.numel() != n {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "label/image count mismatch",
            ));
        }
        let mut labels = Tensor::new(&[n, 10]);
        for i in 0..n {
            let cls = label_ids.data[i] as usize;
            labels.data[i * 10 + cls] = 1.0;
        }

        Ok(Self {
            images,
            n,
            rows,
            cols,
            labels,
        })
    }

    pub fn gather_batch(&self, indices: &[usize], bx: &mut Tensor, by: &mut Tensor) -> bool {
        let pix = self.rows * self.cols;
        if bx.shape != [indices.len(), pix] || by.shape != [indices.len(), 10] {
            return false;
        }
        for (row, &idx) in indices.iter().enumerate() {
            if idx >= self.n {
                return false;
            }
            let src = 16 + idx * pix;
            let dst = row * pix;
            for j in 0..pix {
                bx.data[dst + j] = self.images[src + j] as f32 / 255.0;
            }
        }
        gather_rows(by, &self.labels, indices)
    }
}

impl Dataset for MnistMmapDataset {
    fn len(&self) -> usize {
        self.n
    }

    fn get(&self, index: usize, x: &mut Tensor, y: &mut Tensor) -> bool {
        self.gather_batch(&[index], x, y)
    }

    fn x_cols(&self) -> usize {
        self.rows * self.cols
    }

    fn y_cols(&self) -> usize {
        10
    }
}

/// Shuffle helper for index order (Fisher–Yates).
pub fn shuffle_indices(order: &mut [usize], rand_u64: impl FnMut() -> u64) {
    let mut rand_u64 = rand_u64;
    for i in (1..order.len()).rev() {
        let j = (rand_u64() as usize) % (i + 1);
        order.swap(i, j);
    }
}
