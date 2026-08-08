//! Embeddable autodiff + tensor runtime for **on-device** learning in Rust.
//!
//! Core flow:
//! ```ignore
//! let mut session = AdaptSession::load_or_new("model.mlckpt", build_model, Adam::new(1e-3))?;
//! session.adapt(&local_batch, TrainBudget::steps(50))?;
//! session.save("model.mlckpt")?;
//! let y = session.predict(&x)?;
//! ```

pub mod budget;
pub mod checkpoint;
pub mod data;
pub mod model;
pub mod optim;
pub mod prng;
pub mod session;
pub mod tensor;

pub use budget::{BudgetMeter, StopReason, TrainBudget};
pub use checkpoint::{load_checkpoint, save_checkpoint};
pub use data::{shuffle_indices, Dataset, MnistMmapDataset, TensorDataset};
pub use model::{
    ModelContext, ModelVar, ModelVarOp, OpParams, MV_FLAG_COST, MV_FLAG_DESIRED_OUTPUT,
    MV_FLAG_INPUT, MV_FLAG_NONE, MV_FLAG_OUTPUT, MV_FLAG_PARAMETER, MV_FLAG_REQUIRES_GRAD,
};
pub use optim::{clip_grad_norm, Adam, LrSchedule, Optimizer, Sgd};
pub use session::{AdaptBatch, AdaptConfig, AdaptMetrics, AdaptSession, ReplayBuffer};
pub use tensor::Tensor;
