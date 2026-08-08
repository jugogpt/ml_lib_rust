

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
        self.flag & MV_FLAG_REQUIRES_GRAD != 0
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
        ModelConext {
            vars: Vec::new(),
            input: None,
            output: None,
            desired_output: None,
            cost: None,
            forward_prog: ModelProgram { vars: Vec::new() },
            cost_prog: ModelProgram {vars: Vec::new() },
        }
    }
}



