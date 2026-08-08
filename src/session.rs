//! On-device adaptation API: load → adapt under budget → save → predict.

use crate::budget::{BudgetMeter, StopReason, TrainBudget};
use crate::checkpoint::{load_checkpoint, save_checkpoint};
use crate::data::Dataset;
use crate::model::ModelContext;
use crate::optim::{clip_grad_norm, Adam, Optimizer};
use crate::prng;
use crate::tensor::Tensor;
use std::io;
use std::path::Path;

/// A labeled minibatch for on-device adaptation (`x`: features, `y`: targets).
#[derive(Clone, Debug)]
pub struct AdaptBatch {
    pub x: Tensor,
    pub y: Tensor,
}

impl AdaptBatch {
    pub fn new(x: Tensor, y: Tensor) -> Self {
        assert_eq!(x.ndim(), 2);
        assert_eq!(y.ndim(), 2);
        assert_eq!(x.shape[0], y.shape[0]);
        Self { x, y }
    }

    pub fn len(&self) -> usize {
        self.x.shape[0]
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

/// Metrics surfaced to host apps after / during adaptation.
#[derive(Clone, Debug, Default)]
pub struct AdaptMetrics {
    pub steps_done: u64,
    pub samples_seen: u64,
    pub mean_loss: f32,
    pub last_loss: f32,
    pub last_accuracy: Option<f32>,
    pub stop: StopReason,
    pub elapsed_millis: u64,
    pub replay_size: usize,
}

/// Ring buffer of past samples to reduce catastrophic forgetting.
#[derive(Clone, Debug)]
pub struct ReplayBuffer {
    capacity: usize,
    x_cols: usize,
    y_cols: usize,
    x: Vec<f32>,
    y: Vec<f32>,
    len: usize,
    write: usize,
}

impl ReplayBuffer {
    pub fn new(capacity: usize, x_cols: usize, y_cols: usize) -> Self {
        Self {
            capacity,
            x_cols,
            y_cols,
            x: vec![0.0; capacity * x_cols],
            y: vec![0.0; capacity * y_cols],
            len: 0,
            write: 0,
        }
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    pub fn push_row(&mut self, x: &[f32], y: &[f32]) {
        assert_eq!(x.len(), self.x_cols);
        assert_eq!(y.len(), self.y_cols);
        if self.capacity == 0 {
            return;
        }
        let i = self.write;
        self.x[i * self.x_cols..(i + 1) * self.x_cols].copy_from_slice(x);
        self.y[i * self.y_cols..(i + 1) * self.y_cols].copy_from_slice(y);
        self.write = (self.write + 1) % self.capacity;
        self.len = (self.len + 1).min(self.capacity);
    }

    pub fn push_batch(&mut self, batch: &AdaptBatch) {
        for r in 0..batch.len() {
            let xs = r * self.x_cols;
            let ys = r * self.y_cols;
            self.push_row(
                &batch.x.data[xs..xs + self.x_cols],
                &batch.y.data[ys..ys + self.y_cols],
            );
        }
    }

    fn copy_row_into(&self, idx: usize, bx: &mut Tensor, by: &mut Tensor, row: usize) {
        let xc = self.x_cols;
        let yc = self.y_cols;
        bx.data[row * xc..(row + 1) * xc]
            .copy_from_slice(&self.x[idx * xc..(idx + 1) * xc]);
        by.data[row * yc..(row + 1) * yc]
            .copy_from_slice(&self.y[idx * yc..(idx + 1) * yc]);
    }

    pub fn sample_row_into(&self, bx: &mut Tensor, by: &mut Tensor, row: usize) {
        debug_assert!(!self.is_empty());
        let idx = (prng::rand() as usize) % self.len;
        self.copy_row_into(idx, bx, by, row);
    }
}

/// Configuration for [`AdaptSession`].
#[derive(Clone, Debug)]
pub struct AdaptConfig {
    /// Fraction of each train minibatch drawn from the replay buffer (0..1).
    pub replay_ratio: f32,
    pub replay_capacity: usize,
    pub grad_clip: Option<f32>,
    pub report_accuracy: bool,
}

impl Default for AdaptConfig {
    fn default() -> Self {
        Self {
            replay_ratio: 0.5,
            replay_capacity: 2048,
            grad_clip: Some(1.0),
            report_accuracy: true,
        }
    }
}

/// High-level on-device learning session.
///
/// Typical flow:
/// ```ignore
/// let mut session = AdaptSession::from_model(model, Adam::new(1e-3));
/// session.load("checkpoints/mnist.mlckpt")?;
/// session.adapt(&local_batch, TrainBudget::steps(50))?;
/// session.save("checkpoints/mnist.mlckpt")?;
/// let y = session.predict(&x)?;
/// ```
pub struct AdaptSession {
    pub model: ModelContext,
    pub opt: Adam,
    pub config: AdaptConfig,
    pub replay: ReplayBuffer,
    /// Cumulative metrics across adapt calls.
    pub totals: AdaptMetrics,
    batch_size: usize,
    x_cols: usize,
    y_cols: usize,
}

impl AdaptSession {
    /// Build a session around an already-compiled model.
    pub fn from_model(model: ModelContext, opt: Adam) -> Self {
        Self::from_model_with_config(model, opt, AdaptConfig::default())
    }

    pub fn from_model_with_config(
        model: ModelContext,
        opt: Adam,
        config: AdaptConfig,
    ) -> Self {
        let input = model.input.expect("model needs an INPUT var");
        let desired = model
            .desired_output
            .expect("model needs a DESIRED_OUTPUT var");
        let (batch_size, x_cols) = {
            let v = model.vars[input].val();
            assert_eq!(v.ndim(), 2, "session expects 2D batched input");
            (v.shape[0], v.shape[1])
        };
        let y_cols = {
            let v = model.vars[desired].val();
            assert_eq!(v.ndim(), 2, "session expects 2D batched labels");
            assert_eq!(v.shape[0], batch_size);
            v.shape[1]
        };

        let replay = ReplayBuffer::new(config.replay_capacity, x_cols, y_cols);
        Self {
            model,
            opt,
            config,
            replay,
            totals: AdaptMetrics::default(),
            batch_size,
            x_cols,
            y_cols,
        }
    }

    /// Construct model via `build`, compile, wrap, then load checkpoint weights.
    pub fn load(
        path: impl AsRef<Path>,
        build: impl FnOnce(&mut ModelContext),
        opt: Adam,
    ) -> io::Result<Self> {
        let mut model = ModelContext::new();
        build(&mut model);
        model.compile();
        let mut session = Self::from_model(model, opt);
        load_checkpoint(path, &session.model, Some(&mut session.opt))?;
        Ok(session)
    }

    /// Like [`load`](Self::load) but starts fresh if the file is missing.
    pub fn load_or_new(
        path: impl AsRef<Path>,
        build: impl FnOnce(&mut ModelContext),
        opt: Adam,
    ) -> io::Result<Self> {
        let path = path.as_ref();
        let mut model = ModelContext::new();
        build(&mut model);
        model.compile();
        let mut session = Self::from_model(model, opt);
        if path.exists() {
            load_checkpoint(path, &session.model, Some(&mut session.opt))?;
        }
        Ok(session)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> io::Result<()> {
        if let Some(parent) = path.as_ref().parent() {
            if !parent.as_os_str().is_empty() {
                std::fs::create_dir_all(parent)?;
            }
        }
        save_checkpoint(path, &self.model, Some(&self.opt))
    }

    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    pub fn x_cols(&self) -> usize {
        self.x_cols
    }

    pub fn y_cols(&self) -> usize {
        self.y_cols
    }

    /// Ingest samples into the replay buffer without training.
    pub fn remember(&mut self, batch: &AdaptBatch) {
        assert_eq!(batch.x.shape[1], self.x_cols);
        assert_eq!(batch.y.shape[1], self.y_cols);
        self.replay.push_batch(batch);
    }

    /// Adapt on a local batch until `budget` is exhausted.
    ///
    /// Each step builds a full model minibatch by mixing rows from `batch`
    /// with samples from the replay buffer (`config.replay_ratio`).
    pub fn adapt(&mut self, batch: &AdaptBatch, budget: TrainBudget) -> AdaptMetrics {
        assert!(!batch.is_empty(), "adapt batch must not be empty");
        assert_eq!(batch.x.shape[1], self.x_cols);
        assert_eq!(batch.y.shape[1], self.y_cols);
        self.replay.push_batch(batch);
        self.adapt_loop(Some(batch), None, budget)
    }

    /// Stream adaptation over a [`Dataset`] (shuffled online loop).
    pub fn adapt_dataset(
        &mut self,
        data: &impl Dataset,
        budget: TrainBudget,
    ) -> AdaptMetrics {
        assert_eq!(data.x_cols(), self.x_cols);
        assert_eq!(data.y_cols(), self.y_cols);
        self.adapt_loop(None, Some(data as &dyn Dataset), budget)
    }

    fn adapt_loop(
        &mut self,
        local: Option<&AdaptBatch>,
        dataset: Option<&dyn Dataset>,
        budget: TrainBudget,
    ) -> AdaptMetrics {
        let mut meter = BudgetMeter::start(budget);
        let mut loss_sum = 0.0f32;
        let mut correct = 0u32;
        let mut classified = 0u32;
        let mut samples_seen = 0u64;

        let mut bx = Tensor::new(&[self.batch_size, self.x_cols]);
        let mut by = Tensor::new(&[self.batch_size, self.y_cols]);

        let input_idx = self.model.input.unwrap();
        let y_idx = self.model.desired_output.unwrap();
        let out_idx = self.model.output.unwrap();
        let cost_idx = self.model.cost.unwrap();

        self.model.training = true;

        // Optional cursor into a dataset
        let mut order: Vec<usize> = dataset.map(|d| (0..d.len()).collect()).unwrap_or_default();
        let mut cursor = 0usize;
        if !order.is_empty() {
            crate::data::shuffle_indices(&mut order, prng::rand);
        }

        loop {
            let bytes = self.model.parameter_bytes() + self.opt.state_bytes();
            if meter.exhausted(bytes) {
                break;
            }

            self.fill_train_batch(&mut bx, &mut by, local, dataset, &mut order, &mut cursor);

            self.model.clear_parameter_grads();
            self.model.vars[input_idx].val_mut().copy_from(&bx);
            self.model.vars[y_idx].val_mut().copy_from(&by);
            self.model.prog_compute(&self.model.cost_prog);
            self.model.prog_compute_grads(&self.model.cost_prog);

            if let Some(max_norm) = self.config.grad_clip {
                clip_grad_norm(&self.model, max_norm);
            }
            self.opt.step(&self.model);

            let loss = self.model.vars[cost_idx].val().sum() / self.batch_size as f32;
            loss_sum += loss;
            samples_seen += self.batch_size as u64;
            meter.tick_step();

            if self.config.report_accuracy {
                let preds = self.model.vars[out_idx].val().argmax_rows();
                let labels = self.model.vars[y_idx].val().argmax_rows();
                for i in 0..self.batch_size {
                    if preds[i] == labels[i] {
                        correct += 1;
                    }
                }
                classified += self.batch_size as u32;
            }
        }

        let bytes = self.model.parameter_bytes() + self.opt.state_bytes();
        let stop = meter
            .stop_reason(bytes)
            .unwrap_or(StopReason::Completed);
        let steps = meter.steps_done;
        let mean_loss = if steps > 0 {
            loss_sum / steps as f32
        } else {
            0.0
        };
        let last_accuracy = if classified > 0 {
            Some(correct as f32 / classified as f32)
        } else {
            None
        };

        let metrics = AdaptMetrics {
            steps_done: steps,
            samples_seen,
            mean_loss,
            last_loss: mean_loss,
            last_accuracy,
            stop,
            elapsed_millis: meter.elapsed().as_millis() as u64,
            replay_size: self.replay.len(),
        };

        self.totals.steps_done += metrics.steps_done;
        self.totals.samples_seen += metrics.samples_seen;
        self.totals.last_loss = metrics.mean_loss;
        self.totals.last_accuracy = metrics.last_accuracy.or(self.totals.last_accuracy);
        self.totals.stop = metrics.stop;
        self.totals.elapsed_millis += metrics.elapsed_millis;
        self.totals.replay_size = self.replay.len();
        if self.totals.steps_done > 0 {
            // cheap running blend
            let t = self.totals.steps_done as f32;
            let s = metrics.steps_done as f32;
            self.totals.mean_loss =
                (self.totals.mean_loss * (t - s).max(0.0) + metrics.mean_loss * s) / t;
        }

        self.model.training = false;
        metrics
    }

    fn fill_train_batch(
        &mut self,
        bx: &mut Tensor,
        by: &mut Tensor,
        local: Option<&AdaptBatch>,
        dataset: Option<&dyn Dataset>,
        order: &mut Vec<usize>,
        cursor: &mut usize,
    ) {
        let replay_n = if self.replay.is_empty() {
            0
        } else {
            ((self.batch_size as f32) * self.config.replay_ratio).round() as usize
        }
        .min(self.batch_size);

        let fresh_n = self.batch_size - replay_n;

        // Fresh rows: prefer dataset stream, else local batch, else replay.
        for row in 0..fresh_n {
            if let Some(data) = dataset {
                if order.is_empty() {
                    self.replay.sample_row_into(bx, by, row);
                    continue;
                }
                if *cursor >= order.len() {
                    crate::data::shuffle_indices(order, prng::rand);
                    *cursor = 0;
                }
                let idx = order[*cursor];
                *cursor += 1;
                let mut rx = Tensor::new(&[1, self.x_cols]);
                let mut ry = Tensor::new(&[1, self.y_cols]);
                data.get(idx, &mut rx, &mut ry);
                bx.data[row * self.x_cols..(row + 1) * self.x_cols].copy_from_slice(&rx.data);
                by.data[row * self.y_cols..(row + 1) * self.y_cols].copy_from_slice(&ry.data);
                self.replay.push_row(&rx.data, &ry.data);
            } else if let Some(batch) = local {
                let src = (prng::rand() as usize) % batch.len();
                let xs = src * self.x_cols;
                let ys = src * self.y_cols;
                bx.data[row * self.x_cols..(row + 1) * self.x_cols]
                    .copy_from_slice(&batch.x.data[xs..xs + self.x_cols]);
                by.data[row * self.y_cols..(row + 1) * self.y_cols]
                    .copy_from_slice(&batch.y.data[ys..ys + self.y_cols]);
            } else if !self.replay.is_empty() {
                self.replay.sample_row_into(bx, by, row);
            } else {
                // should not happen
                bx.data[row * self.x_cols..(row + 1) * self.x_cols].fill(0.0);
                by.data[row * self.y_cols..(row + 1) * self.y_cols].fill(0.0);
            }
        }

        for row in fresh_n..self.batch_size {
            if !self.replay.is_empty() {
                self.replay.sample_row_into(bx, by, row);
            } else if let Some(batch) = local {
                let src = (prng::rand() as usize) % batch.len();
                let xs = src * self.x_cols;
                let ys = src * self.y_cols;
                bx.data[row * self.x_cols..(row + 1) * self.x_cols]
                    .copy_from_slice(&batch.x.data[xs..xs + self.x_cols]);
                by.data[row * self.y_cols..(row + 1) * self.y_cols]
                    .copy_from_slice(&batch.y.data[ys..ys + self.y_cols]);
            }
        }
    }

    /// Run forward pass. `x` may be `[F]`, `[1, F]`, or `[B, F]` with `B <= batch_size`.
    pub fn predict(&mut self, x: &Tensor) -> io::Result<Tensor> {
        let input_idx = self.model.input.unwrap();
        let out_idx = self.model.output.unwrap();

        let (rows, cols, src) = match x.ndim() {
            1 => {
                if x.shape[0] != self.x_cols {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("expected {} features, got {}", self.x_cols, x.shape[0]),
                    ));
                }
                (1usize, self.x_cols, x.data.as_slice())
            }
            2 => {
                if x.shape[1] != self.x_cols {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!("expected {} features, got {}", self.x_cols, x.shape[1]),
                    ));
                }
                if x.shape[0] > self.batch_size {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidInput,
                        format!(
                            "batch {} exceeds session batch_size {}",
                            x.shape[0], self.batch_size
                        ),
                    ));
                }
                (x.shape[0], self.x_cols, x.data.as_slice())
            }
            _ => {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidInput,
                    "predict expects rank-1 or rank-2 tensor",
                ))
            }
        };

        self.model.training = false;
        {
            let mut inp = self.model.vars[input_idx].val_mut();
            inp.clear();
            inp.data[..rows * cols].copy_from_slice(src);
        }
        self.model.feedforward();

        let out = self.model.vars[out_idx].val();
        let y_cols = out.shape[1];
        let mut result = Tensor::new(&[rows, y_cols]);
        result.data.copy_from_slice(&out.data[..rows * y_cols]);
        Ok(result)
    }

    /// Convenience: predicted class index for a single feature vector.
    pub fn predict_class(&mut self, x: &Tensor) -> io::Result<usize> {
        let y = self.predict(x)?;
        Ok(y.argmax_rows()[0])
    }
}
