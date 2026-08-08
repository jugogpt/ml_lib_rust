use crate::model::{ModelContext, MV_FLAG_PARAMETER};
use crate::tensor::Tensor;
use std::collections::HashMap;

/// Learning-rate schedule evaluated at a global step count.
#[derive(Clone, Debug)]
pub enum LrSchedule {
    Constant(f32),
    StepDecay {
        initial: f32,
        drop_every: u64,
        gamma: f32,
    },
    Cosine {
        initial: f32,
        min: f32,
        total_steps: u64,
    },
}

impl LrSchedule {
    pub fn lr(&self, step: u64) -> f32 {
        match self {
            LrSchedule::Constant(lr) => *lr,
            LrSchedule::StepDecay {
                initial,
                drop_every,
                gamma,
            } => {
                if *drop_every == 0 {
                    return *initial;
                }
                let drops = step / *drop_every;
                initial * gamma.powi(drops as i32)
            }
            LrSchedule::Cosine {
                initial,
                min,
                total_steps,
            } => {
                if *total_steps == 0 {
                    return *initial;
                }
                let t = (step.min(*total_steps) as f32) / (*total_steps as f32);
                let cos = (1.0 + (std::f32::consts::PI * t).cos()) * 0.5;
                min + (initial - min) * cos
            }
        }
    }
}

/// Clip all parameter gradients so total L2 norm ≤ `max_norm`. Returns pre-clip norm.
pub fn clip_grad_norm(model: &ModelContext, max_norm: f32) -> f32 {
    let mut sumsq = 0.0f32;
    for v in &model.vars {
        if v.flags & MV_FLAG_PARAMETER == 0 || !v.has_grad() {
            continue;
        }
        let g = v.grad();
        for &x in &g.data {
            sumsq += x * x;
        }
    }
    let norm = sumsq.sqrt();
    if norm > max_norm && norm > 0.0 {
        let scale = max_norm / norm;
        for v in &model.vars {
            if v.flags & MV_FLAG_PARAMETER == 0 || !v.has_grad() {
                continue;
            }
            v.grad_mut().scale(scale);
        }
    }
    norm
}

pub trait Optimizer {
    fn step(&mut self, model: &ModelContext);
    fn set_lr(&mut self, lr: f32);
    fn lr(&self) -> f32;
    fn global_step(&self) -> u64;
}

/// Plain SGD with optional weight decay (decoupled).
#[derive(Clone, Debug)]
pub struct Sgd {
    pub schedule: LrSchedule,
    pub weight_decay: f32,
    step: u64,
}

impl Sgd {
    pub fn new(lr: f32) -> Self {
        Self {
            schedule: LrSchedule::Constant(lr),
            weight_decay: 0.0,
            step: 0,
        }
    }

    pub fn with_schedule(schedule: LrSchedule) -> Self {
        Self {
            schedule,
            weight_decay: 0.0,
            step: 0,
        }
    }
}

impl Optimizer for Sgd {
    fn step(&mut self, model: &ModelContext) {
        let lr = self.schedule.lr(self.step);
        for &i in &model.parameter_indices() {
            let cur = &model.vars[i];
            if !cur.has_grad() {
                continue;
            }
            let g = cur.grad();
            let mut w = cur.val_mut();
            for k in 0..w.numel() {
                let mut gi = g.data[k];
                if self.weight_decay != 0.0 {
                    gi += self.weight_decay * w.data[k];
                }
                w.data[k] -= lr * gi;
            }
        }
        self.step += 1;
    }

    fn set_lr(&mut self, lr: f32) {
        self.schedule = LrSchedule::Constant(lr);
    }

    fn lr(&self) -> f32 {
        self.schedule.lr(self.step)
    }

    fn global_step(&self) -> u64 {
        self.step
    }
}

/// Adam with optional weight decay (AdamW-style when `weight_decay > 0`).
#[derive(Clone, Debug)]
pub struct Adam {
    pub schedule: LrSchedule,
    pub beta1: f32,
    pub beta2: f32,
    pub eps: f32,
    pub weight_decay: f32,
    step: u64,
    m: HashMap<usize, Vec<f32>>,
    v: HashMap<usize, Vec<f32>>,
}

impl Adam {
    pub fn new(lr: f32) -> Self {
        Self {
            schedule: LrSchedule::Constant(lr),
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay: 0.0,
            step: 0,
            m: HashMap::new(),
            v: HashMap::new(),
        }
    }

    pub fn with_schedule(schedule: LrSchedule) -> Self {
        let mut a = Self::new(0.0);
        a.schedule = schedule;
        a
    }

    fn ensure_state(&mut self, idx: usize, n: usize) {
        self.m.entry(idx).or_insert_with(|| vec![0.0; n]);
        self.v.entry(idx).or_insert_with(|| vec![0.0; n]);
    }

    pub fn state_bytes(&self) -> u64 {
        let mut b = 0u64;
        for s in self.m.values() {
            b += (s.len() * 4) as u64;
        }
        for s in self.v.values() {
            b += (s.len() * 4) as u64;
        }
        b
    }

    /// Accessors for checkpointing.
    pub fn export_state(&self) -> (u64, Vec<(usize, Vec<f32>, Vec<f32>)>) {
        let mut out = Vec::new();
        for (&k, m) in &self.m {
            let v = self.v.get(&k).cloned().unwrap_or_else(|| vec![0.0; m.len()]);
            out.push((k, m.clone(), v));
        }
        out.sort_by_key(|(k, _, _)| *k);
        (self.step, out)
    }

    pub fn import_state(&mut self, step: u64, slots: Vec<(usize, Vec<f32>, Vec<f32>)>) {
        self.step = step;
        self.m.clear();
        self.v.clear();
        for (k, m, v) in slots {
            self.m.insert(k, m);
            self.v.insert(k, v);
        }
    }

    pub fn global_step(&self) -> u64 {
        self.step
    }
}

impl Optimizer for Adam {
    fn step(&mut self, model: &ModelContext) {
        self.step += 1;
        let t = self.step as f32;
        let lr = self.schedule.lr(self.step.saturating_sub(1));
        let b1 = self.beta1;
        let b2 = self.beta2;
        let eps = self.eps;
        let wd = self.weight_decay;

        for &i in &model.parameter_indices() {
            let cur = &model.vars[i];
            if !cur.has_grad() {
                continue;
            }
            let n = cur.val().numel();
            self.ensure_state(i, n);

            let g = cur.grad();
            let m = self.m.get_mut(&i).unwrap();
            let v = self.v.get_mut(&i).unwrap();
            let mut w = cur.val_mut();

            for k in 0..n {
                let gi = g.data[k];
                // AdamW: decay weights directly
                if wd != 0.0 {
                    w.data[k] -= lr * wd * w.data[k];
                }
                m[k] = b1 * m[k] + (1.0 - b1) * gi;
                v[k] = b2 * v[k] + (1.0 - b2) * gi * gi;
                let mhat = m[k] / (1.0 - b1.powf(t));
                let vhat = v[k] / (1.0 - b2.powf(t));
                w.data[k] -= lr * mhat / (vhat.sqrt() + eps);
            }
        }
    }

    fn set_lr(&mut self, lr: f32) {
        self.schedule = LrSchedule::Constant(lr);
    }

    fn lr(&self) -> f32 {
        self.schedule.lr(self.step)
    }

    fn global_step(&self) -> u64 {
        self.step
    }
}

/// Zero a tensor in-place helper used by tests/examples.
pub fn zero_like(t: &Tensor) -> Tensor {
    Tensor::zeros_like(t)
}
