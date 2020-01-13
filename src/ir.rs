#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Register {
    name: String,
}

impl Register {
    pub fn new(name: String) -> Register {
        Register { name }
    }

    pub fn name(&self) -> &String {
        &self.name
    }
}

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
    RegisterRef(Register),
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
pub enum Target {
    Location(u64),
    Label(String),
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Operation {
    Skip,
    Barrier,
    Assignment {
        reg: Register,
        expr: Expression,
    },
    ConditionalAssignment {
        reg: Register,
        expr: Expression,
        cond: Expression,
    },
    Load {
        reg: Register,
        addr: Expression,
    },
    Store {
        reg: Register,
        addr: Expression,
    },
    Jump {
        target: Target,
    },
    Branch {
        kind: BranchKind,
        reg: Register,
        target: Target,
    },
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Instruction {
    operation: Operation,
    label: Option<String>,
}

impl Instruction {
    pub fn new(operation: Operation) -> Instruction {
        Instruction {
            operation,
            label: None,
        }
    }

    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn set_label(&mut self, label: String) {
        self.label = Some(label);
    }

    pub fn label(&self) -> &Option<String> {
        &self.label
    }

    pub fn skip() -> Instruction {
        Instruction::new(Operation::Skip)
    }

    pub fn barrier() -> Instruction {
        Instruction::new(Operation::Barrier)
    }

    pub fn assign(reg: Register, expr: Expression) -> Instruction {
        Instruction::new(Operation::Assignment { reg, expr })
    }

    pub fn assign_if(cond: Expression, reg: Register, expr: Expression) -> Instruction {
        Instruction::new(Operation::ConditionalAssignment { cond, reg, expr })
    }

    pub fn load(reg: Register, addr: Expression) -> Instruction {
        Instruction::new(Operation::Load { reg, addr })
    }

    pub fn store(reg: Register, addr: Expression) -> Instruction {
        Instruction::new(Operation::Store { reg, addr })
    }

    pub fn jump(target: Target) -> Instruction {
        Instruction::new(Operation::Jump { target })
    }

    pub fn branch_if_zero(reg: Register, target: Target) -> Instruction {
        Instruction::new(Operation::Branch {
            kind: BranchKind::IfZero,
            reg,
            target,
        })
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Program {
    instructions: Vec<Instruction>,
    end_label: Option<String>,
}

impl Program {
    pub fn new(instructions: Vec<Instruction>) -> Program {
        Program {
            instructions,
            end_label: None,
        }
    }

    pub fn instructions(&self) -> &Vec<Instruction> {
        &self.instructions
    }

    pub fn set_end_label(&mut self, label: String) {
        self.end_label = Some(label);
    }

    pub fn end_label(&self) -> &Option<String> {
        &self.end_label
    }
}
