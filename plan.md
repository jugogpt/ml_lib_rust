# ml_lib_rust — On-device learning

## North star

Build a **safe, embeddable autodiff library** so Rust programs can **train and adapt small models on-device** — inside a binary, game, mobile/edge agent, or WASM app — with no Python runtime and no cloud round-trip for personalization.

> Train once (or ship a prior). Adapt forever on the user's machine.

That is the Rust-specific wedge: **in-process continual learning** with deterministic, testable, shippable code.

## Why Rust (not another PyTorch clone)

- **Embed**: `cargo add` a trainer into any app; single binary / WASM artifact
- **Safety**: shape checks, no GIL, fearless parallel data + kernels
- **Constraints**: path to `no_std`+alloc inference; bounded memory/CPU budgets
- **Privacy**: user data never leaves the device during fine-tunes
- **Repro**: deterministic RNG + checkpoints for CI and rollback

## Product pillars

1. **Tiny models that fit devices** — MLPs, small CNNs, tiny transformers (not 7B)
2. **Fast local adapt** — few-step fine-tune / replay buffer / LoRA-style adapters later
3. **Checkpoint format** — load prior → train N steps on local data → save; resume-safe
4. **Budgets** — max RAM, max steps, max latency APIs so apps stay responsive
5. **Inference path** — freeze graph, strip training, run hot path everywhere

## Current foundation (done)

- N-D `Tensor`, batched matmul (Rayon), graph autodiff
- Ops: relu/sigmoid/tanh/GELU, softmax, CE, layernorm, dropout, conv2d, max-pool, embedding
- MNIST demo ~96% accuracy (`cargo run --release`)

## Roadmap

### Phase 0 — Library shape (done)
- Lib crate (`src/`) + `examples/mnist.rs`
- Adam + SGD, LR schedules, gradient clip
- Native checkpoint `*.mlckpt` (params + Adam state)
- `Dataset` trait + `MnistMmapDataset` (memmap images)
- `TrainBudget` / `BudgetMeter` (`max_steps`, `max_millis`, `max_bytes`)

### Phase 1 — On-device trainer API (done)
```rust
let mut session = AdaptSession::load_or_new("model.mlckpt", build, Adam::new(1e-3))?;
session.adapt(&local_batch, TrainBudget::steps(50))?;
session.save("model.mlckpt")?;
let y = session.predict(&x)?;
```
- `AdaptSession` + `AdaptBatch` + `AdaptMetrics`
- Replay buffer with configurable mix ratio
- Budget-interrupted online loops (`adapt` / `adapt_dataset`)
- Examples: `mnist`, `adapt`

### Phase 2 — Flagship demos (prove usefulness)
1. **Personal digit pad** — ship MNIST prior; fine-tune on the user's handwritten digits
2. **Local anomaly / habit model** — tiny MLP on device sensor or log features
3. **WASM adapt** — same API in the browser (checkpoint in IndexedDB)

### Phase 3 — Device constraints
- Inference-only feature flag; shrink deps
- Quantization path (int8 weights) for storage + bandwidth
- Optional `no_std`+alloc predict
- CPU SIMD kernels; backend trait for future GPU/Metal/Vulkan later

### Phase 4 — Credibility
- Numerical gradient CI, benches vs candle/burn (CPU, small models)
- docs.rs + book: “Embed a learner in 50 lines”
- Shape-typed tensors (compile-time ranks) once the dynamic API is stable

## Non-goals (for now)

- Competing on large-scale GPU training / HF model zoo
- Distributed multi-node
- Full CUDA parity

## Success metric

A third-party Rust app can **ship a prior, adapt on local data under a budget, and keep working offline** — with tests that prove the adapt step is deterministic and bounded.
