use std::time::{Duration, Instant};

/// Hard limits so on-device training can stay responsive.
#[derive(Clone, Debug)]
pub struct TrainBudget {
    pub max_steps: Option<u64>,
    pub max_millis: Option<u64>,
    /// Soft/hard cap on resident parameter+optimizer bytes the caller cares about.
    pub max_bytes: Option<u64>,
}

impl TrainBudget {
    pub fn unlimited() -> Self {
        Self {
            max_steps: None,
            max_millis: None,
            max_bytes: None,
        }
    }

    pub fn steps(n: u64) -> Self {
        Self {
            max_steps: Some(n),
            max_millis: None,
            max_bytes: None,
        }
    }

    pub fn millis(ms: u64) -> Self {
        Self {
            max_steps: None,
            max_millis: Some(ms),
            max_bytes: None,
        }
    }

    pub fn with_max_bytes(mut self, bytes: u64) -> Self {
        self.max_bytes = Some(bytes);
        self
    }

    pub fn with_max_steps(mut self, steps: u64) -> Self {
        self.max_steps = Some(steps);
        self
    }

    pub fn with_max_millis(mut self, ms: u64) -> Self {
        self.max_millis = Some(ms);
        self
    }
}

/// Tracks consumption of a [`TrainBudget`] during a training session.
#[derive(Debug)]
pub struct BudgetMeter {
    budget: TrainBudget,
    start: Instant,
    pub steps_done: u64,
}

impl BudgetMeter {
    pub fn start(budget: TrainBudget) -> Self {
        Self {
            budget,
            start: Instant::now(),
            steps_done: 0,
        }
    }

    pub fn elapsed(&self) -> Duration {
        self.start.elapsed()
    }

    pub fn tick_step(&mut self) {
        self.steps_done += 1;
    }

    pub fn exhausted(&self, bytes_used: u64) -> bool {
        if let Some(max) = self.budget.max_steps {
            if self.steps_done >= max {
                return true;
            }
        }
        if let Some(ms) = self.budget.max_millis {
            if self.elapsed().as_millis() as u64 >= ms {
                return true;
            }
        }
        if let Some(max_b) = self.budget.max_bytes {
            if bytes_used > max_b {
                return true;
            }
        }
        false
    }

    pub fn budget(&self) -> &TrainBudget {
        &self.budget
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Default)]
pub enum StopReason {
    #[default]
    Completed,
    MaxSteps,
    MaxMillis,
    MaxBytes,
}

impl BudgetMeter {
    pub fn stop_reason(&self, bytes_used: u64) -> Option<StopReason> {
        if let Some(max) = self.budget.max_steps {
            if self.steps_done >= max {
                return Some(StopReason::MaxSteps);
            }
        }
        if let Some(ms) = self.budget.max_millis {
            if self.elapsed().as_millis() as u64 >= ms {
                return Some(StopReason::MaxMillis);
            }
        }
        if let Some(max_b) = self.budget.max_bytes {
            if bytes_used > max_b {
                return Some(StopReason::MaxBytes);
            }
        }
        None
    }
}
