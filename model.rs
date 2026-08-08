

use crate::matrix::{
    mat_add, mat_cross_entropy, mat_cross_entropy_add_grad, mat_mul, mat_relu, mat_relu_add_grad,
    mat_softmax, mat_softmax_add_grad, mat_sub, Matrix,
};
use std::cell::{Ref, RefCell, RefMut};



//conditionals to calls  the functions below
pub const MV_FLAG_NONE: u32 = 0; 
pub const MV_FLAG_REQUIRES_GRAD: u32 = 1 <<0;
pub const MV_FLAG_PARAMETER: u32 = 1 << 1;
pub const MV_FLAG_INPUT: u32 = 1 << 2;
pub const MV_FLAG_OUTPUT: u32 = 1 << 3;
pub const MV_FLAG_DESIRED_OUTPUT: u32 = 1 << 4;
pub const MV_FLAG_COST: u32 = 1 << 5;


#[derive(Clone, Copy, PartialEq, Eq, Debug)]
pub enum ModelVarOp { //types of ModelVarOp (operatiors that we can use to act on the model's variables)
    Null, 
    Create,
    Relu,
    Softmax, 
    Add,
    Sub,
    Matmul,
    CrossEntropy,
}

impl ModelVarOp {
    fn num_inputs(self) -> usize {
        use ModelVarOp::*;
        match self {
            Null | Create => 0,
            Relu | Softmax => 1, 
            Add | Sub | Matmul | CrossEntropy => 2,
        }
    }
}


// RefCell wrappings intend to solve the conflict that arises bc rust's borrow checker won't allow
//the pattern where 'model_var' nodes reference each otehr via raw pointers, which would be possible in C++ or C
// to exist directly (mutably updating one node's matrix while reading its neigbor's matrices all living in the same backing store); 
//so each node's matrices are wrapped in this 'RefCell' wrap.

// NOdes are stored in a Vec and referred to by index (in our case usize) instead of by pointer
// which would lead to ownership errors
pub struct ModelVar {
    pub index: usize, 
    pub flags: u32,

    val: RefCell<Matrix>,
    grad: Option<RefCell<Matrix>>,
    pub op: ModelVarOp,
    pub inputs: [Option<usize>; 2],
}

impl ModelVar {

    pub fn val(&self) -> Ref<'_, Matrix> {
        self.val.borrow()
    }

    pub fn val_mut(&self) -> RefMut<'_, Matrix> {
        self.val.borrow_mut()
    }

    pub fn grad(&self) -> Ref<'_, Matrix> {
        self.grad.as_ref().expect("var has no grad").borrow()
    }

    pub fn grad_mut(&self) -> RefMut<'_, Matrix> {
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
            cost_prog: ModelProgram {vars: Vec::new() },
        }
    }


    pub fn create_var(&mut self, rows: usize, cols: usize, flags: u32) -> usize {
        let index: usize = self.vars.len();

        let grad: Option<RefCell<Matrix>> = if flags & MV_FLAG_REQUIRES_GRAD != 0 {
            Some(RefCell::new(Matrix::new(rows, cols)))
        }else {
            None
        };

        self.vars.push(ModelVar {
            index,
            flags,
            val: RefCell::new(Matrix::new(rows, cols)),
            grad,
            op: ModelVarOp::Create,
            inputs: [None, None],
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

    fn unary(
        &mut self,
        input: usize,
        rows: usize,
        cols: usize,
        mut flags: u32,
        op: ModelVarOp,
    ) -> usize {
        if self.vars[input].requires_grad() {
            flags |= MV_FLAG_REQUIRES_GRAD;
        }

        let out = self.create_var(rows, cols, flags);
        self.vars[out].op = op;
        self.vars[out].inputs[0] = Some(input);
        out
    }

    fn binary(
        &mut self,
        a: usize,
        b: usize,
        rows: usize,
        cols: usize,
        mut flags: u32,
        op: ModelVarOp,
    ) -> usize {
        if self.vars[a].requires_grad() || self.vars[b].requires_grad() {
            flags |= MV_FLAG_REQUIRES_GRAD;
        }

        let out = self.create_var(rows, cols, flags);
        self.vars[out].op = op;
        self.vars[out].inputs[0] = Some(a);
        self.vars[out].inputs[1] = Some(b);
        out
    }

    pub fn relu(&mut self, input: usize, flags: u32) -> usize {
        let (rows, cols) = {
            let v = self.vars[input].val();
            (v.rows, v.cols)
        };
        self.unary(input, rows, cols, flags, ModelVarOp::Relu)
    }

    pub fn softmax(&mut self, input: usize, flags: u32) -> usize {
        let (rows, cols) = {
            let v = self.vars[input].val();
            (v.rows, v.cols)
        };
        self.unary(input, rows, cols, flags, ModelVarOp::Softmax)
    }

    pub fn add(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (a_rows, a_cols) = {
            let v = self.vars[a].val();
            (v.rows, v.cols)
        };
        let (b_rows, b_cols) = {
            let v = self.vars[b].val();
            (v.rows, v.cols)
        };

        if a_rows != b_rows || a_cols != b_cols {
            return None;
        }

        Some(self.binary(a, b, a_rows, a_cols, flags, ModelVarOp::Add))
    }

    #[allow(dead_code)]
    pub fn sub(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (a_rows, a_cols) = {
            let v = self.vars[a].val();
            (v.rows, v.cols)
        };
        let (b_rows, b_cols) = {
            let v = self.vars[b].val();
            (v.rows, v.cols)
        };

        if a_rows != b_rows || a_cols != b_cols {
            return None;
        }

        Some(self.binary(a, b, a_rows, a_cols, flags, ModelVarOp::Sub))
    }

    pub fn matmul(&mut self, a: usize, b: usize, flags: u32) -> Option<usize> {
        let (a_rows, a_cols) = {
            let v = self.vars[a].val();
            (v.rows, v.cols)
        };
        let (b_rows, b_cols) = {
            let v = self.vars[b].val();
            (v.rows, v.cols)
        };

        if a_cols != b_rows {
            return None;
        }

        Some(self.binary(a, b, a_rows, b_cols, flags, ModelVarOp::Matmul))
    }

    pub fn cross_entropy(&mut self, p: usize, q: usize, flags: u32) -> Option<usize> {
        let (p_rows, p_cols) = {
            let v = self.vars[p].val();
            (v.rows, v.cols)
        };
        let (q_rows, q_cols) = {
            let v = self.vars[q].val();
            (v.rows, v.cols)
        };

        if p_rows != q_rows || p_cols != q_cols {
            return None;
        }

        Some(self.binary(p, q, p_rows, p_cols, flags, ModelVarOp::CrossEntropy))
    }


    //Builds a topologically sorted evaluation order ending in 'out_var', via the same 
    //via the same iterative stack based traversal as model_prog_create in 

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

            match cur.op {
                ModelVarOp::Null | ModelVarOp::Create => {}

                ModelVarOp::Relu => {
                    let av = self.vars[a.unwrap()].val();
                    mat_relu(&mut cur.val_mut(), &av);
                }
                ModelVarOp::Softmax => {
                    let av = self.vars[a.unwrap()].val();
                    mat_softmax(&mut cur.val_mut(), &av);
                }

                ModelVarOp::Add => {
                    let av = self.vars[a.unwrap()].val();
                    let bv = self.vars[b.unwrap()].val();
                    mat_add(&mut cur.val_mut(), &av, &bv);
                }
                ModelVarOp::Sub => {
                    let av = self.vars[a.unwrap()].val();
                    let bv = self.vars[b.unwrap()].val();
                    mat_sub(&mut cur.val_mut(), &av, &bv);
                }
                ModelVarOp::Matmul => {
                    let av = self.vars[a.unwrap()].val();
                    let bv = self.vars[b.unwrap()].val();
                    mat_mul(&mut cur.val_mut(), &av, &bv, true, false, false);
                }
                ModelVarOp::CrossEntropy => {
                    let av = self.vars[a.unwrap()].val();
                    let bv = self.vars[b.unwrap()].val();
                    mat_cross_entropy(&mut cur.val_mut(), &av, &bv);
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

        self.vars[*prog.vars.last().unwrap()].grad_mut().fill(1.0);

        for &i in prog.vars.iter().rev() {
            let cur = &self.vars[i];

            if !cur.requires_grad() {
                continue;
            }

            let a = cur.inputs[0];
            let b = cur.inputs[1];
            let num_inputs = cur.op.num_inputs();

            if num_inputs == 1 && !self.vars[a.unwrap()].requires_grad() {
                continue;
            }
            if num_inputs == 2
                && !self.vars[a.unwrap()].requires_grad()
                && !self.vars[b.unwrap()].requires_grad()
            {
                continue;
            }

            match cur.op {
                ModelVarOp::Null | ModelVarOp::Create => {}

                ModelVarOp::Relu => {
                    let av = self.vars[a.unwrap()].val();
                    let cg = cur.grad();
                    let mut ag = self.vars[a.unwrap()].grad_mut();
                    mat_relu_add_grad(&mut ag, &av, &cg);
                }
                ModelVarOp::Softmax => {
                    let cv = cur.val();
                    let cg = cur.grad();
                    let mut ag = self.vars[a.unwrap()].grad_mut();
                    mat_softmax_add_grad(&mut ag, &cv, &cg);
                }

                ModelVarOp::Add => {
                    // a.grad += cur.grad; b.grad += cur.grad
                    // (element-wise, since out/a/b of an accumulating add
                    // can't all be borrowed via mat_add when out aliases a
                    // or b's storage the way the C pointers did)
                    let cg = cur.grad();
                    if self.vars[a.unwrap()].requires_grad() {
                        let mut ag = self.vars[a.unwrap()].grad_mut();
                        for i in 0..ag.data.len() {
                            ag.data[i] += cg.data[i];
                        }
                    }
                    if self.vars[b.unwrap()].requires_grad() {
                        let mut bg = self.vars[b.unwrap()].grad_mut();
                        for i in 0..bg.data.len() {
                            bg.data[i] += cg.data[i];
                        }
                    }
                }
                ModelVarOp::Sub => {
                    // a.grad += cur.grad; b.grad -= cur.grad
                    let cg = cur.grad();
                    if self.vars[a.unwrap()].requires_grad() {
                        let mut ag = self.vars[a.unwrap()].grad_mut();
                        for i in 0..ag.data.len() {
                            ag.data[i] += cg.data[i];
                        }
                    }
                    if self.vars[b.unwrap()].requires_grad() {
                        let mut bg = self.vars[b.unwrap()].grad_mut();
                        for i in 0..bg.data.len() {
                            bg.data[i] -= cg.data[i];
                        }
                    }
                }

                ModelVarOp::Matmul => {
                    let cg = cur.grad();
                    if self.vars[a.unwrap()].requires_grad() {
                        let bv = self.vars[b.unwrap()].val();
                        let mut ag = self.vars[a.unwrap()].grad_mut();
                        mat_mul(&mut ag, &cg, &bv, false, false, true);
                    }
                    if self.vars[b.unwrap()].requires_grad() {
                        let av = self.vars[a.unwrap()].val();
                        let mut bg = self.vars[b.unwrap()].grad_mut();
                        mat_mul(&mut bg, &av, &cg, false, true, false);
                    }
                }

                ModelVarOp::CrossEntropy => {
                    let (p_idx, q_idx) = (a.unwrap(), b.unwrap());
                    let cg = cur.grad();
                    let pv = self.vars[p_idx].val();
                    let qv = self.vars[q_idx].val();

                    let p_requires = self.vars[p_idx].requires_grad();
                    let q_requires = self.vars[q_idx].requires_grad();

                    // p and q are always distinct nodes here, so it's safe
                    // to hold both grad borrows at once.
                    let mut p_grad_ref = if p_requires {
                        Some(self.vars[p_idx].grad_mut())
                    } else {
                        None
                    };
                    let mut q_grad_ref = if q_requires {
                        Some(self.vars[q_idx].grad_mut())
                    } else {
                        None
                    };

                    mat_cross_entropy_add_grad(
                        p_grad_ref.as_deref_mut(),
                        q_grad_ref.as_deref_mut(),
                        &pv,
                        &qv,
                        &cg,
                    );
                }
            }
        }
    }

}





