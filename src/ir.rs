#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Register {
    name: String,
}

impl Register {
    #[must_use]
    pub fn new(name: String) -> Self {
        Self { name }
    }

    #[must_use]
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
    Unary {
        op: UnaryOperator,
        expr: Box<Expression>,
    },
    Binary {
        op: BinaryOperator,
        lhs: Box<Expression>,
        rhs: Box<Expression>,
    },
    Conditional {
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
    #[must_use]
    pub fn new(operation: Operation) -> Self {
        Self {
            operation,
            label: None,
        }
    }

    #[must_use]
    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn set_label(&mut self, label: String) {
        self.label = Some(label);
    }

    #[must_use]
    pub fn label(&self) -> &Option<String> {
        &self.label
    }

    #[must_use]
    pub fn skip() -> Self {
        Self::new(Operation::Skip)
    }

    #[must_use]
    pub fn barrier() -> Self {
        Self::new(Operation::Barrier)
    }

    #[must_use]
    pub fn assign(reg: Register, expr: Expression) -> Self {
        Self::new(Operation::Assignment { reg, expr })
    }

    #[must_use]
    pub fn assign_if(cond: Expression, reg: Register, expr: Expression) -> Self {
        Self::new(Operation::ConditionalAssignment { cond, reg, expr })
    }

    #[must_use]
    pub fn load(reg: Register, addr: Expression) -> Self {
        Self::new(Operation::Load { reg, addr })
    }

    #[must_use]
    pub fn store(reg: Register, addr: Expression) -> Self {
        Self::new(Operation::Store { reg, addr })
    }

    #[must_use]
    pub fn jump(target: Target) -> Self {
        Self::new(Operation::Jump { target })
    }

    #[must_use]
    pub fn branch_if_zero(reg: Register, target: Target) -> Self {
        Self::new(Operation::Branch {
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
    #[must_use]
    pub fn new(instructions: Vec<Instruction>) -> Self {
        Self {
            instructions,
            end_label: None,
        }
    }

    #[must_use]
    pub fn instructions(&self) -> &Vec<Instruction> {
        &self.instructions
    }

    pub fn set_end_label(&mut self, label: String) {
        self.end_label = Some(label);
    }

    #[must_use]
    pub fn end_label(&self) -> &Option<String> {
        &self.end_label
    }
}
