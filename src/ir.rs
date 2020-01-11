#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum UnaryOperator {
    Neg,
    Not,
    SExt,
    ZExt,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum BinaryOperator {
    Add,
    Sub,
    Mul,
    UDiv,
    URem,
    SRem,
    SMod,
    And,
    Or,
    Xor,
    Shl,
    AShr,
    LShr,
    ULe,
    ULt,
    UGe,
    UGt,
    SLe,
    SLt,
    SGe,
    SGt,
    r#Eq,
    Neq,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Expression {
    NumberLiteral(u64),
    RegisterRef(String),
    UnaryExpression {
        op: UnaryOperator,
        expr: Box<Expression>,
    },
    BinaryExpression {
        op: BinaryOperator,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },
    ConditionalExpression {
        cond: Box<Expression>,
        then: Box<Expression>,
        r#else: Box<Expression>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum BranchKind {
    IfZero,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Operation {
    Skip,
    Barrier,
    Assignment {
        reg: String,
        expr: Box<Expression>,
    },
    ConditionalAssignment {
        reg: String,
        expr: Box<Expression>,
        cond: Box<Expression>,
    },
    Load {
        reg: String,
        addr: Box<Expression>,
    },
    Store {
        reg: String,
        addr: Box<Expression>,
    },
    Jump {
        target: Box<Expression>,
    },
    Branch {
        kind: BranchKind,
        reg: String,
        target: Box<Expression>,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Instruction {
    operation: Operation,
}

impl Instruction {
    pub fn new(operation: Operation) -> Instruction {
        Instruction {
            operation: operation,
        }
    }

    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn skip() -> Instruction {
        Instruction::new(Operation::Skip)
    }

    pub fn barrier() -> Instruction {
        Instruction::new(Operation::Barrier)
    }

    pub fn assign(reg: String, expr: Box<Expression>) -> Instruction {
        Instruction::new(Operation::Assignment {
            reg: reg,
            expr: expr,
        })
    }

    pub fn assign_if(cond: Box<Expression>, reg: String, expr: Box<Expression>) -> Instruction {
        Instruction::new(Operation::ConditionalAssignment {
            cond: cond,
            reg: reg,
            expr: expr,
        })
    }

    pub fn load(reg: String, addr: Box<Expression>) -> Instruction {
        Instruction::new(Operation::Load {
            reg: reg,
            addr: addr,
        })
    }

    pub fn store(reg: String, addr: Box<Expression>) -> Instruction {
        Instruction::new(Operation::Store {
            reg: reg,
            addr: addr,
        })
    }

    pub fn jump(target: Box<Expression>) -> Instruction {
        Instruction::new(Operation::Jump { target: target })
    }

    pub fn branch_if_zero(reg: String, target: Box<Expression>) -> Instruction {
        Instruction::new(Operation::Branch {
            kind: BranchKind::IfZero,
            reg: reg,
            target: target,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Program {
    instructions: Vec<Box<Instruction>>,
}

impl Program {
    pub fn new(instructions: Vec<Box<Instruction>>) -> Program {
        Program {
            instructions: instructions,
        }
    }

    pub fn instructions(&self) -> &Vec<Box<Instruction>> {
        &self.instructions
    }
}
