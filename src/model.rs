#![allow(dead_code)]

use crate::prng::Prng;
use crate::tensor::*;
use std::cell::{Ref, RefCell, RefMut};

pub const MV_FLAG_NONE: u32 = 0;
pub const MV_FLAG_REQUIRES_GRAD: u32 = 1 << 0;
pub const MV_FLAG_PARAMETER: u32 = 1 << 1;
pub const MV_FLAG_INPUT: u32 = 1 << 2;
pub const MV_FLAG_OUTPUT: u32 = 1 << 3;
pub const MV_FLAG_DESIRED_OUTPUT: u32 = 1 << 4;
pub const MV_FLAG_COST: u32 = 1 << 5;

#[derive(Clone, Copy, PartialEq, Debug)]
pub enum OpParams {
    None,
    Matmul {
        transpose_a: bool,
        transpose_b: bool,
    },
    Dropout {
        p: f32,
    },
    LayerNorm {
        eps: f32,
    },
    Conv2d {
        stride: usize,
        padding: usize,
    },
    MaxPool2d {
        kernel: usize,
        stride: usize,
    },
}

impl Default for OpParams {
    fn default() -> Self {
        OpParams::None
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelVarOp {
    Null,
    Create,
    Relu,
    Sigmoid,
    Tanh,
    Gelu,
    Softmax,
    Add,
    Sub,
    Matmul,
    CrossEntropy,
    LayerNorm,
    Dropout,
    Conv2d,
    MaxPool2d,
    Embedding,
}

impl ModelVarOp {
    fn num_inputs(self) -> usize {
        use ModelVarOp::*;
        match self {
            Null | Create => 0,
            Relu | Sigmoid | Tanh | Gelu | Softmax | Dropout | MaxPool2d => 1,
            Add | Sub | Matmul | CrossEntropy | Embedding => 2,
            LayerNorm => 3, // x, gamma, beta
            Conv2d => 3,    // x, weight, bias (bias may be unused if None input)
        }
    }
}

pub struct ModelVar {
    pub index: usize,
    pub flags: u32,
    val: RefCell<Tensor>,
    grad: Option<RefCell<Tensor>>,
    pub op: ModelVarOp,
    pub inputs: [Option<usize>; 3],
    pub params: OpParams,
    /// Scratch for dropout masks, layernorm stats, maxpool indices.
    cache: RefCell<Option<Tensor>>,
}

impl ModelVar {
    pub fn val(&self) -> Ref<'_, Tensor> {
        self.val.borrow()
    }

    pub fn val_mut(&self) -> RefMut<'_, Tensor> {
        self.val.borrow_mut()
    }

    pub fn grad(&self) -> Ref<'_, Tensor> {
        self.grad.as_ref().expect("var has no grad").borrow()
    }

    pub fn grad_mut(&self) -> RefMut<'_, Tensor> {
        self.grad.as_ref().expect("var has no grad").borrow_mut()
    }

    pub fn has_grad(&self) -> bool {
        self.grad.is_some()
    }

    fn requires_grad(&self) -> bool {
        self.flags & MV_FLAG_REQUIRES_GRAD != 0
    }
}

pub struct ModelProgram {
    pub vars: Vec<usize>,
}

pub struct ModelContext {
    pub vars: Vec<ModelVar>,
    pub input: Option<usize>,
    pub output: Option<usize>,
    pub desired_output: Option<usize>,
    pub cost: Option<usize>,
    pub forward_prog: ModelProgram,
    pub cost_prog: ModelProgram,
    /// Controls dropout behavior.
    pub training: bool,
    rng: RefCell<Prng>,
}

impl ModelContext {
    pub fn new() -> Self {
        ModelContext {
            vars: Vec::new(),
            input: None,
            output: None,
            desired_output: None,
            cost: None,
            forward_prog: ModelProgram { vars: Vec::new() },
            cost_prog: ModelProgram { vars: Vec::new() },
            training: true,
            rng: RefCell::new(Prng::new(0x853c49e6748fea9b, 0xda3e39cb94b95bdb)),
        }
    }

    pub fn create_var(&mut self, shape: &[usize], flags: u32) -> usize {
        let index = self.vars.len();
        let grad = if flags & MV_FLAG_REQUIRES_GRAD != 0 {
            Some(RefCell::new(Tensor::new(shape)))
        } else {
            None
        };

        self.vars.push(ModelVar {
            index,
            flags,
            val: RefCell::new(Tensor::new(shape)),
            grad,
            op: ModelVarOp::Create,
            inputs: [None, None, None],
            params: OpParams::None,
            cache: RefCell::new(None),
        });

        if flags & MV_FLAG_INPUT != 0 {
            self.input = Some(index);
        }
        if flags & MV_FLAG_OUTPUT != 0 {
            self.output = Some(index);
        }
        if flags & MV_FLAG_DESIRED_OUTPUT != 0 {
            self.desired_output = Some(index);
        }
        if flags & MV_FLAG_COST != 0 {
            self.cost = Some(index);
        }
        index
    }

    fn wire(
        &mut self,
        out: usize,
        op: ModelVarOp,
        inputs: &[Option<usize>],
        params: OpParams,
    ) -> usize {
        self.vars[out].op = op;
        self.vars[out].params = params;
        for (i, inp) in inputs.iter().enumerate() {
            self.vars[out].inputs[i] = *inp;
        }
        out
    }

    fn maybe_grad_flag(&self, inputs: &[Option<usize>], mut flags: u32) -> u32 {
        for inp in inputs.iter().flatten() {
            if self.vars[*inp].requires_grad() {
                flags |= MV_FLAG_REQUIRES_GRAD;
                break;
            }
        }
        flags
    }

    pub fn relu(&mut self, input: usize, flags: u32) -> usize {
        let shape = self.vars[input].val().shape.clone();
        let flags = self.maybe_grad_flag(&[Some(input)], flags);
        let out = self.create_var(&shape, flags);
        self.wire(out, ModelVarOp::Relu, &[Some(input)], OpParams::None)
    }

    pub fn sigmoid(&mut self, input: usize, flags: u32) -> usize {
        let shape = self.vars[input].val().shape.clone();
        let flags = self.maybe_grad_flag(&[Some(input)], flags);
        let out = self.create_var(&shape, flags);
        self.wire(out, ModelVarOp::Sigmoid, &[Some(input)], OpParams::None)
    }

    pub fn tanh(&mut self, input: usize, flags: u32) -> usize {
        let shape = self.vars[input].val().shape.clone();
        let flags = self.maybe_grad_flag(&[Some(input)], flags);
        let out = self.create_var(&shape, flags);
        self.wire(out, ModelVarOp::Tanh, &[Some(input)], OpParams::None)
    }

    pub fn gelu(&mut self, input: usize, flags: u32) -> usize {
        let shape = self.vars[input].val().shape.clone();
        let flags = self.maybe_grad_flag(&[Some(input)], flags);
        let out = self.create_var(&shape, flags);
        self.wire(out, ModelVarOp::Gelu, &[Some(input)], OpParams::None)
    }

    pub fn softmax(&mut self, input: usize, flags: u32) -> usize {
        let shape = self.vars[input].val().shape.clone();
        let flags = self.maybe_grad_flag(&[Some(input)], flags);
        let out = self.create_var(&shape, flags);
        self.wire(out, ModelVarOp::Softmax, &[Some(input)], OpParams::None)
    }

    pub fn dropout(&mut self, input: usize, p: f32, flags: u32) -> usize {
        let shape = self.vars[input].val().shape.clone();
        let flags = self.maybe_grad_flag(&[Some(input)], flags);
        let out = self.create_var(&shape, flags);
        self.wire(
            out,
            ModelVarOp::Dropout,
            &[Some(input)],
            OpParams::Dropout { p },
        )
    }

    pub fn add(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let shape = self.vars[a].val().shape.clone();
        // allow broadcast bias
        let flags = self.maybe_grad_flag(&[Some(a), Some(b)], flags);
        let out = self.create_var(&shape, flags);
        Some(self.wire(out, ModelVarOp::Add, &[Some(a), Some(b)], OpParams::None))
    }

    pub fn sub(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let sa = self.vars[a].val().shape.clone();
        let sb = self.vars[b].val().shape.clone();
        if sa != sb {
            return None;
        }
        let flags = self.maybe_grad_flag(&[Some(a), Some(b)], flags);
        let out = self.create_var(&sa, flags);
        Some(self.wire(out, ModelVarOp::Sub, &[Some(a), Some(b)], OpParams::None))
    }

    /// Batched matmul. Common case: `x[B,K] @ w[K,N] -> [B,N]`.
    pub fn matmul(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        self.matmul_ex(a, b, false, false, flags)
    }

    pub fn matmul_ex(
        &mut self,
        a: usize,
        b: usize,
        transpose_a: bool,
        transpose_b: bool,
        flags: u32,
    ) -> Option<usize> {
        let (a_shape, b_shape) = {
            let av = self.vars[a].val();
            let bv = self.vars[b].val();
            (av.shape.clone(), bv.shape.clone())
        };

        let out_shape = matmul_out_shape(&a_shape, &b_shape, transpose_a, transpose_b)?;
        let flags = self.maybe_grad_flag(&[Some(a), Some(b)], flags);
        let out = self.create_var(&out_shape, flags);
        Some(self.wire(
            out,
            ModelVarOp::Matmul,
            &[Some(a), Some(b)],
            OpParams::Matmul {
                transpose_a,
                transpose_b,
            },
        ))
    }

    pub fn cross_entropy(&mut self, p: usize, q: usize, flags: u32) -> Option<usize> {
        let shape = self.vars[p].val().shape.clone();
        if shape != self.vars[q].val().shape {
            return None;
        }
        let flags = self.maybe_grad_flag(&[Some(p), Some(q)], flags);
        let out = self.create_var(&shape, flags);
        Some(self.wire(
            out,
            ModelVarOp::CrossEntropy,
            &[Some(p), Some(q)],
            OpParams::None,
        ))
    }

    pub fn layer_norm(
        &mut self,
        x: usize,
        gamma: usize,
        beta: usize,
        eps: f32,
        flags: u32,
    ) -> usize {
        let shape = self.vars[x].val().shape.clone();
        let flags = self.maybe_grad_flag(&[Some(x), Some(gamma), Some(beta)], flags);
        let out = self.create_var(&shape, flags);
        self.wire(
            out,
            ModelVarOp::LayerNorm,
            &[Some(x), Some(gamma), Some(beta)],
            OpParams::LayerNorm { eps },
        )
    }

    pub fn conv2d(
        &mut self,
        x: usize,
        weight: usize,
        bias: Option<usize>,
        stride: usize,
        padding: usize,
        flags: u32,
    ) -> Option<usize> {
        let (n, _ic, ih, iw) = {
            let v = self.vars[x].val();
            if v.ndim() != 4 {
                return None;
            }
            (v.shape[0], v.shape[1], v.shape[2], v.shape[3])
        };
        let (oc, kh, kw) = {
            let w = self.vars[weight].val();
            if w.ndim() != 4 {
                return None;
            }
            (w.shape[0], w.shape[2], w.shape[3])
        };
        let oh = (ih + 2 * padding - kh) / stride + 1;
        let ow = (iw + 2 * padding - kw) / stride + 1;
        let flags = self.maybe_grad_flag(&[Some(x), Some(weight), bias], flags);
        let out = self.create_var(&[n, oc, oh, ow], flags);
        Some(self.wire(
            out,
            ModelVarOp::Conv2d,
            &[Some(x), Some(weight), bias],
            OpParams::Conv2d { stride, padding },
        ))
    }

    pub fn max_pool2d(
        &mut self,
        x: usize,
        kernel: usize,
        stride: usize,
        flags: u32,
    ) -> Option<usize> {
        let (n, c, ih, iw) = {
            let v = self.vars[x].val();
            if v.ndim() != 4 {
                return None;
            }
            (v.shape[0], v.shape[1], v.shape[2], v.shape[3])
        };
        let oh = (ih - kernel) / stride + 1;
        let ow = (iw - kernel) / stride + 1;
        let flags = self.maybe_grad_flag(&[Some(x)], flags);
        let out = self.create_var(&[n, c, oh, ow], flags);
        Some(self.wire(
            out,
            ModelVarOp::MaxPool2d,
            &[Some(x)],
            OpParams::MaxPool2d { kernel, stride },
        ))
    }

    pub fn embedding(&mut self, indices: usize, weight: usize, flags: u32) -> Option<usize> {
        let mut shape = self.vars[indices].val().shape.clone();
        let d = {
            let w = self.vars[weight].val();
            if w.ndim() != 2 {
                return None;
            }
            w.shape[1]
        };
        shape.push(d);
        let flags = self.maybe_grad_flag(&[Some(indices), Some(weight)], flags);
        let out = self.create_var(&shape, flags);
        Some(self.wire(
            out,
            ModelVarOp::Embedding,
            &[Some(indices), Some(weight)],
            OpParams::None,
        ))
    }

    pub fn prog_create(&self, out_var: usize) -> ModelProgram {
        let n = self.vars.len();
        let mut visited = vec![false; n];
        let mut stack: Vec<usize> = Vec::with_capacity(n);
        let mut out: Vec<usize> = Vec::with_capacity(n);
        stack.push(out_var);

        while let Some(cur) = stack.pop() {
            if cur >= n {
                continue;
            }
            if visited[cur] {
                out.push(cur);
                continue;
            }
            visited[cur] = true;
            stack.push(cur);
            let num_inputs = self.vars[cur].op.num_inputs();
            for i in 0..num_inputs {
                if let Some(input) = self.vars[cur].inputs[i] {
                    if input >= n || visited[input] {
                        continue;
                    }
                    if let Some(pos) = stack.iter().position(|&x| x == input) {
                        stack.remove(pos);
                    }
                    stack.push(input);
                }
            }
        }
        ModelProgram { vars: out }
    }

    pub fn compile(&mut self) {
        if let Some(output) = self.output {
            self.forward_prog = self.prog_create(output);
        }
        if let Some(cost) = self.cost {
            self.cost_prog = self.prog_create(cost);
        }
    }

    pub fn feedforward(&self) {
        self.prog_compute(&self.forward_prog);
    }

    pub fn prog_compute(&self, prog: &ModelProgram) {
        for &i in &prog.vars {
            let cur = &self.vars[i];
            let a = cur.inputs[0];
            let b = cur.inputs[1];
            let c = cur.inputs[2];

            match cur.op {
                ModelVarOp::Null | ModelVarOp::Create => {}
                ModelVarOp::Relu => {
                    let av = self.vars[a.unwrap()].val();
                    tensor_relu(&mut cur.val_mut(), &av);
                }
                ModelVarOp::Sigmoid => {
                    let av = self.vars[a.unwrap()].val();
                    tensor_sigmoid(&mut cur.val_mut(), &av);
                }
                ModelVarOp::Tanh => {
                    let av = self.vars[a.unwrap()].val();
                    tensor_tanh(&mut cur.val_mut(), &av);
                }
                ModelVarOp::Gelu => {
                    let av = self.vars[a.unwrap()].val();
                    tensor_gelu(&mut cur.val_mut(), &av);
                }
                ModelVarOp::Softmax => {
                    let av = self.vars[a.unwrap()].val();
                    tensor_softmax(&mut cur.val_mut(), &av);
                }
                ModelVarOp::Add => {
                    let av = self.vars[a.unwrap()].val();
                    let bv = self.vars[b.unwrap()].val();
                    tensor_add(&mut cur.val_mut(), &av, &bv);
                }
                ModelVarOp::Sub => {
                    let av = self.vars[a.unwrap()].val();
                    let bv = self.vars[b.unwrap()].val();
                    tensor_sub(&mut cur.val_mut(), &av, &bv);
                }
                ModelVarOp::Matmul => {
                    let av = self.vars[a.unwrap()].val();
                    let bv = self.vars[b.unwrap()].val();
                    let (ta, tb) = match cur.params {
                        OpParams::Matmul {
                            transpose_a,
                            transpose_b,
                        } => (transpose_a, transpose_b),
                        _ => (false, false),
                    };
                    tensor_matmul(&mut cur.val_mut(), &av, &bv, true, ta, tb);
                }
                ModelVarOp::CrossEntropy => {
                    let av = self.vars[a.unwrap()].val();
                    let bv = self.vars[b.unwrap()].val();
                    tensor_cross_entropy(&mut cur.val_mut(), &av, &bv);
                }
                ModelVarOp::LayerNorm => {
                    let eps = match cur.params {
                        OpParams::LayerNorm { eps } => eps,
                        _ => 1e-5,
                    };
                    let xv = self.vars[a.unwrap()].val();
                    let gv = self.vars[b.unwrap()].val();
                    let bv = self.vars[c.unwrap()].val();
                    let mut cache = cur.cache.borrow_mut();
                    if cache.is_none() {
                        *cache = Some(Tensor::new(&[xv.rows() * 2]));
                    }
                    tensor_layer_norm(
                        &mut cur.val_mut(),
                        &xv,
                        &gv,
                        &bv,
                        eps,
                        cache.as_mut().unwrap(),
                    );
                }
                ModelVarOp::Dropout => {
                    let p = match cur.params {
                        OpParams::Dropout { p } => p,
                        _ => 0.0,
                    };
                    let av = self.vars[a.unwrap()].val();
                    let mut cache = cur.cache.borrow_mut();
                    if cache.is_none() {
                        *cache = Some(Tensor::zeros_like(&av));
                    }
                    let mut rng = self.rng.borrow_mut();
                    tensor_dropout(
                        &mut cur.val_mut(),
                        &av,
                        cache.as_mut().unwrap(),
                        p,
                        self.training,
                        &mut rng,
                    );
                }
                ModelVarOp::Conv2d => {
                    let (stride, padding) = match cur.params {
                        OpParams::Conv2d { stride, padding } => (stride, padding),
                        _ => (1, 0),
                    };
                    let xv = self.vars[a.unwrap()].val();
                    let wv = self.vars[b.unwrap()].val();
                    let bias = c.map(|i| self.vars[i].val());
                    tensor_conv2d(
                        &mut cur.val_mut(),
                        &xv,
                        &wv,
                        bias.as_deref(),
                        stride,
                        padding,
                    );
                }
                ModelVarOp::MaxPool2d => {
                    let (kernel, stride) = match cur.params {
                        OpParams::MaxPool2d { kernel, stride } => (kernel, stride),
                        _ => (2, 2),
                    };
                    let xv = self.vars[a.unwrap()].val();
                    let mut cache = cur.cache.borrow_mut();
                    if cache.is_none() {
                        *cache = Some(Tensor::zeros_like(&cur.val()));
                    }
                    tensor_max_pool2d(
                        &mut cur.val_mut(),
                        &xv,
                        cache.as_mut().unwrap(),
                        kernel,
                        stride,
                    );
                }
                ModelVarOp::Embedding => {
                    let iv = self.vars[a.unwrap()].val();
                    let wv = self.vars[b.unwrap()].val();
                    tensor_embedding(&mut cur.val_mut(), &iv, &wv);
                }
            }
        }
    }

    pub fn prog_compute_grads(&self, prog: &ModelProgram) {
        for &i in &prog.vars {
            let cur = &self.vars[i];
            if !cur.requires_grad() {
                continue;
            }
            if cur.flags & MV_FLAG_PARAMETER != 0 {
                continue;
            }
            cur.grad_mut().clear();
        }

        // Mean over batch: scale seed grad by 1/batch when cost is [B, C]
        {
            let last = *prog.vars.last().unwrap();
            let mut g = self.vars[last].grad_mut();
            let batch = g.rows().max(1);
            g.fill(1.0 / batch as f32);
        }

        for &i in prog.vars.iter().rev() {
            let cur = &self.vars[i];
            if !cur.requires_grad() {
                continue;
            }

            let a = cur.inputs[0];
            let b = cur.inputs[1];
            let c = cur.inputs[2];

            match cur.op {
                ModelVarOp::Null | ModelVarOp::Create => {}
                ModelVarOp::Relu => {
                    let av = self.vars[a.unwrap()].val();
                    let cg = cur.grad();
                    let mut ag = self.vars[a.unwrap()].grad_mut();
                    tensor_relu_add_grad(&mut ag, &av, &cg);
                }
                ModelVarOp::Sigmoid => {
                    let cv = cur.val();
                    let cg = cur.grad();
                    let mut ag = self.vars[a.unwrap()].grad_mut();
                    tensor_sigmoid_add_grad(&mut ag, &cv, &cg);
                }
                ModelVarOp::Tanh => {
                    let cv = cur.val();
                    let cg = cur.grad();
                    let mut ag = self.vars[a.unwrap()].grad_mut();
                    tensor_tanh_add_grad(&mut ag, &cv, &cg);
                }
                ModelVarOp::Gelu => {
                    let av = self.vars[a.unwrap()].val();
                    let cg = cur.grad();
                    let mut ag = self.vars[a.unwrap()].grad_mut();
                    tensor_gelu_add_grad(&mut ag, &av, &cg);
                }
                ModelVarOp::Softmax => {
                    let cv = cur.val();
                    let cg = cur.grad();
                    let mut ag = self.vars[a.unwrap()].grad_mut();
                    tensor_softmax_add_grad(&mut ag, &cv, &cg);
                }
                ModelVarOp::Add => {
                    let cg = cur.grad();
                    if self.vars[a.unwrap()].requires_grad() {
                        let mut ag = self.vars[a.unwrap()].grad_mut();
                        for i in 0..ag.numel().min(cg.numel()) {
                            ag.data[i] += cg.data[i];
                        }
                    }
                    if self.vars[b.unwrap()].requires_grad() {
                        let mut bg = self.vars[b.unwrap()].grad_mut();
                        // broadcast-add grad: sum over batch if bias is [F]
                        if bg.ndim() == 1 && cg.ndim() == 2 && bg.shape[0] == cg.shape[1] {
                            let (batch, feat) = (cg.shape[0], cg.shape[1]);
                            for r in 0..batch {
                                for f in 0..feat {
                                    bg.data[f] += cg.data[r * feat + f];
                                }
                            }
                        } else if bg.shape == [bg.cols(), 1] && cg.ndim() == 2 {
                            let (batch, feat) = (cg.shape[0], cg.shape[1]);
                            for r in 0..batch {
                                for f in 0..feat {
                                    bg.data[f] += cg.data[r * feat + f];
                                }
                            }
                        } else {
                            for i in 0..bg.numel().min(cg.numel()) {
                                bg.data[i] += cg.data[i];
                            }
                        }
                    }
                }
                ModelVarOp::Sub => {
                    let cg = cur.grad();
                    if self.vars[a.unwrap()].requires_grad() {
                        let mut ag = self.vars[a.unwrap()].grad_mut();
                        for i in 0..ag.numel() {
                            ag.data[i] += cg.data[i];
                        }
                    }
                    if self.vars[b.unwrap()].requires_grad() {
                        let mut bg = self.vars[b.unwrap()].grad_mut();
                        for i in 0..bg.numel() {
                            bg.data[i] -= cg.data[i];
                        }
                    }
                }
                ModelVarOp::Matmul => {
                    let (ta, tb) = match cur.params {
                        OpParams::Matmul {
                            transpose_a,
                            transpose_b,
                        } => (transpose_a, transpose_b),
                        _ => (false, false),
                    };
                    let cg = cur.grad();
                    // For plain x[B,K] @ w[K,N]: dX = dY @ W^T, dW = X^T @ dY
                    if self.vars[a.unwrap()].requires_grad() {
                        let bv = self.vars[b.unwrap()].val();
                        let mut ag = self.vars[a.unwrap()].grad_mut();
                        // dA = dOut @ B^T  (when !ta && !tb)
                        tensor_matmul(&mut ag, &cg, &bv, false, false, !tb);
                        let _ = ta;
                    }
                    if self.vars[b.unwrap()].requires_grad() {
                        let av = self.vars[a.unwrap()].val();
                        let mut bg = self.vars[b.unwrap()].grad_mut();
                        // dB = A^T @ dOut
                        tensor_matmul(&mut bg, &av, &cg, false, !ta, false);
                    }
                }
                ModelVarOp::CrossEntropy => {
                    let (p_idx, q_idx) = (a.unwrap(), b.unwrap());
                    let cg = cur.grad();
                    let pv = self.vars[p_idx].val();
                    let qv = self.vars[q_idx].val();
                    let mut p_grad_ref = if self.vars[p_idx].requires_grad() {
                        Some(self.vars[p_idx].grad_mut())
                    } else {
                        None
                    };
                    let mut q_grad_ref = if self.vars[q_idx].requires_grad() {
                        Some(self.vars[q_idx].grad_mut())
                    } else {
                        None
                    };
                    tensor_cross_entropy_add_grad(
                        p_grad_ref.as_deref_mut(),
                        q_grad_ref.as_deref_mut(),
                        &pv,
                        &qv,
                        &cg,
                    );
                }
                ModelVarOp::LayerNorm => {
                    let xv = self.vars[a.unwrap()].val();
                    let gv = self.vars[b.unwrap()].val();
                    let cg = cur.grad();
                    let cache = cur.cache.borrow();
                    let mean_inv = cache.as_ref().unwrap();
                    let mut xg = self.vars[a.unwrap()].grad_mut();
                    let mut gg = if self.vars[b.unwrap()].requires_grad() {
                        Some(self.vars[b.unwrap()].grad_mut())
                    } else {
                        None
                    };
                    let mut bg = if self.vars[c.unwrap()].requires_grad() {
                        Some(self.vars[c.unwrap()].grad_mut())
                    } else {
                        None
                    };
                    tensor_layer_norm_add_grad(
                        &mut xg,
                        gg.as_deref_mut(),
                        bg.as_deref_mut(),
                        &xv,
                        &gv,
                        &cg,
                        mean_inv,
                    );
                }
                ModelVarOp::Dropout => {
                    let cache = cur.cache.borrow();
                    let mask = cache.as_ref().unwrap();
                    let cg = cur.grad();
                    let mut ag = self.vars[a.unwrap()].grad_mut();
                    tensor_dropout_add_grad(&mut ag, mask, &cg);
                }
                ModelVarOp::Conv2d => {
                    let (stride, padding) = match cur.params {
                        OpParams::Conv2d { stride, padding } => (stride, padding),
                        _ => (1, 0),
                    };
                    let xv = self.vars[a.unwrap()].val();
                    let wv = self.vars[b.unwrap()].val();
                    let cg = cur.grad();
                    let mut ig = if self.vars[a.unwrap()].requires_grad() {
                        Some(self.vars[a.unwrap()].grad_mut())
                    } else {
                        None
                    };
                    let mut wg = if self.vars[b.unwrap()].requires_grad() {
                        Some(self.vars[b.unwrap()].grad_mut())
                    } else {
                        None
                    };
                    let mut bg = if let Some(bi) = c {
                        if self.vars[bi].requires_grad() {
                            Some(self.vars[bi].grad_mut())
                        } else {
                            None
                        }
                    } else {
                        None
                    };
                    tensor_conv2d_add_grad(
                        ig.as_deref_mut(),
                        wg.as_deref_mut(),
                        bg.as_deref_mut(),
                        &xv,
                        &wv,
                        &cg,
                        stride,
                        padding,
                    );
                }
                ModelVarOp::MaxPool2d => {
                    let cache = cur.cache.borrow();
                    let indices = cache.as_ref().unwrap();
                    let cg = cur.grad();
                    let mut ag = self.vars[a.unwrap()].grad_mut();
                    tensor_max_pool2d_add_grad(&mut ag, &cg, indices);
                }
                ModelVarOp::Embedding => {
                    // typically only weight is trained
                    if self.vars[b.unwrap()].requires_grad() {
                        let iv = self.vars[a.unwrap()].val();
                        let cg = cur.grad();
                        let mut wg = self.vars[b.unwrap()].grad_mut();
                        tensor_embedding_add_grad(&mut wg, &iv, &cg);
                    }
                }
            }
        }
    }

    /// Indices of variables marked `MV_FLAG_PARAMETER`, in creation order.
    pub fn parameter_indices(&self) -> Vec<usize> {
        self.vars
            .iter()
            .filter(|v| v.flags & MV_FLAG_PARAMETER != 0)
            .map(|v| v.index)
            .collect()
    }

    pub fn clear_parameter_grads(&self) {
        for &i in &self.parameter_indices() {
            if self.vars[i].has_grad() {
                self.vars[i].grad_mut().clear();
            }
        }
    }

    /// Approximate parameter storage in bytes (f32 weights only).
    pub fn parameter_bytes(&self) -> u64 {
        self.parameter_indices()
            .iter()
            .map(|&i| (self.vars[i].val().numel() * 4) as u64)
            .sum()
    }
}

fn matmul_out_shape(
    a: &[usize],
    b: &[usize],
    transpose_a: bool,
    transpose_b: bool,
) -> Option<Vec<usize>> {
    let (a_r, a_c) = match a.len() {
        2 => {
            if transpose_a {
                (a[1], a[0])
            } else {
                (a[0], a[1])
            }
        }
        _ => return None,
    };
    let (b_r, b_c) = match b.len() {
        2 => {
            if transpose_b {
                (b[1], b[0])
            } else {
                (b[0], b[1])
            }
        }
        _ => return None,
    };
    if a_c != b_r {
        return None;
    }
    Some(vec![a_r, b_c])
}
