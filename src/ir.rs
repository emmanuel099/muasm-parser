use fmt::Display;
use std::fmt;

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

impl fmt::Display for Register {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "%{}", self.name)
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Display)]
pub enum UnaryOperator {
    Neg,
    Not,
    SExt,
    ZExt,
}

#[derive(Clone, Debug, Eq, PartialEq, Hash, Display)]
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
    Eq,
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

impl fmt::Display for Expression {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::NumberLiteral(num) => write!(f, "{}", num),
            Self::RegisterRef(reg) => write!(f, "{}", reg),
            Self::Unary { op, expr } => write!(f, "({} {})", op, expr),
            Self::Binary { op, lhs, rhs } => write!(f, "({} {} {})", op, lhs, rhs),
            Self::Conditional { cond, then, r#else } => {
                write!(f, "(Ite {} {} {})", cond, then, r#else)
            }
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Target {
    Location(u64),
    Label(String),
}

impl fmt::Display for Target {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Location(num) => write!(f, "{}", num),
            Self::Label(lbl) => write!(f, "@{}", lbl),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub enum Operation {
    Skip,
    Barrier,
    Flush,
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
    BranchIfZero {
        reg: Register,
        target: Target,
    },
}

impl fmt::Display for Operation {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            Self::Skip => write!(f, "skip"),
            Self::Barrier => write!(f, "barrier"),
            Self::Flush => write!(f, "flush"),
            Self::Assignment { reg, expr } => write!(f, "{} = {}", reg, expr),
            Self::ConditionalAssignment { reg, expr, cond } => {
                write!(f, "{} = {} if {}", reg, expr, cond)
            }
            Self::Load { reg, addr } => write!(f, "load {}, {}", reg, addr),
            Self::Store { reg, addr } => write!(f, "store {}, {}", reg, addr),
            Self::Jump { target } => write!(f, "jump {}", target),
            Self::BranchIfZero { reg, target } => write!(f, "beqz {}, {}", reg, target),
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Instruction {
    operation: Operation,
    address: u64,
    label: Option<String>,
}

impl Instruction {
    #[must_use]
    pub fn new(operation: Operation) -> Self {
        Self {
            operation,
            address: 0,
            label: None,
        }
    }

    #[must_use]
    pub fn operation(&self) -> &Operation {
        &self.operation
    }

    pub fn set_address(&mut self, address: u64) {
        self.address = address;
    }

    #[must_use]
    pub fn address(&self) -> u64 {
        self.address
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
    pub fn flush() -> Self {
        Self::new(Operation::Flush)
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
        Self::new(Operation::BranchIfZero { reg, target })
    }

    #[must_use]
    pub fn is_skip(&self) -> bool {
        self.operation == Operation::Skip
    }

    #[must_use]
    pub fn is_barrier(&self) -> bool {
        self.operation == Operation::Barrier
    }

    #[must_use]
    pub fn is_assign(&self) -> bool {
        if let Operation::Assignment { .. } = self.operation {
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn is_assign_if(&self) -> bool {
        if let Operation::ConditionalAssignment { .. } = self.operation {
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn is_load(&self) -> bool {
        if let Operation::Load { .. } = self.operation {
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn is_store(&self) -> bool {
        if let Operation::Store { .. } = self.operation {
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn is_jump(&self) -> bool {
        if let Operation::Jump { .. } = self.operation {
            true
        } else {
            false
        }
    }

    #[must_use]
    pub fn is_branch_if_zero(&self) -> bool {
        if let Operation::BranchIfZero { .. } = self.operation {
            true
        } else {
            false
        }
    }
}

impl fmt::Display for Instruction {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        if let Some(lbl) = &self.label {
            write!(f, "{}: {}", lbl, self.operation)
        } else {
            write!(f, "{}", self.operation)
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct Program {
    instructions: Vec<Instruction>,
}

impl Program {
    #[must_use]
    pub fn new(instructions: Vec<Instruction>) -> Self {
        Self { instructions }
    }

    #[must_use]
    pub fn instructions(&self) -> &Vec<Instruction> {
        &self.instructions
    }
}

impl fmt::Display for Program {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let formatted_instructions = self
            .instructions
            .iter()
            .fold(String::new(), |acc, inst| format!("{}{}\n", acc, inst));
        write!(f, "{}", formatted_instructions)
    }
}
