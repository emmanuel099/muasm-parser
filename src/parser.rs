use crate::ir;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{digit1, hex_digit1, multispace0, space0},
    combinator::{all_consuming, map, map_res, opt, value},
    multi::fold_many0,
    sequence::{preceded, terminated, tuple},
    IResult,
};
use std::str::FromStr;

pub fn parse_program(input: &str) -> Result<ir::Program, &str> {
    let result = all_consuming(tuple((
        instructions,
        opt(preceded(multispace0, label)),
        multispace0,
    )))(input);
    match result {
        Ok((_, (instructions, opt_end_label, _))) => {
            let mut program = ir::Program::new(instructions);
            if let Some(lbl) = opt_end_label {
                program.set_end_label(lbl)
            };
            Ok(program)
        }
        Err(_) => Err("Failed to parse program!"),
    }
}

fn instructions(input: &str) -> IResult<&str, Vec<ir::Instruction>> {
    fold_many0(
        preceded(multispace0, alt((labeled_instruction, instruction))),
        Vec::new(),
        |mut instructions: Vec<_>, inst| {
            instructions.push(inst);
            instructions
        },
    )(input)
}

fn identifier(input: &str) -> IResult<&str, String> {
    let alpha_underscore = |c: char| c.is_alphabetic() || c == '_';
    let alphanumeric_underscore = |c: char| c.is_alphanumeric() || c == '_';
    map(
        tuple((
            take_while1(alpha_underscore),
            take_while(alphanumeric_underscore),
        )),
        |(s1, s2): (&str, &str)| s1.to_string() + s2,
    )(input)
}

fn dec_num(input: &str) -> IResult<&str, u64> {
    map_res(digit1, FromStr::from_str)(input)
}

fn hex_num(input: &str) -> IResult<&str, u64> {
    let from_str = |s: &str| u64::from_str_radix(s, 16);
    map_res(preceded(tag("0x"), hex_digit1), from_str)(input)
}

fn numeric(input: &str) -> IResult<&str, u64> {
    alt((hex_num, dec_num))(input)
}

fn number_literal(input: &str) -> IResult<&str, Box<ir::Expression>> {
    map(numeric, |n| Box::new(ir::Expression::NumberLiteral(n)))(input)
}

fn register_ref(input: &str) -> IResult<&str, Box<ir::Expression>> {
    map(identifier, |r| Box::new(ir::Expression::RegisterRef(r)))(input)
}

fn unary_expression(input: &str) -> IResult<&str, Box<ir::Expression>> {
    let operator = alt((
        value(ir::UnaryOperator::Neg, tag("-")),
        value(ir::UnaryOperator::Not, tag("~")),
    ));
    map(
        tuple((
            preceded(space0, operator),
            preceded(space0, simple_expression),
        )),
        |(op, expr)| Box::new(ir::Expression::UnaryExpression { op, expr }),
    )(input)
}

macro_rules! binary_expression {
    ($func:ident, $operator:expr, $operand:expr) => {
        fn $func(input: &str) -> IResult<&str, Box<ir::Expression>> {
            map(
                tuple((
                    preceded(space0, $operand),
                    preceded(space0, $operator),
                    preceded(space0, $operand),
                )),
                |(lhs, op, rhs)| Box::new(ir::Expression::BinaryExpression { op, lhs, rhs }),
            )(input)
        }
    };
}

binary_expression!(
    bitwise_expression,
    alt((
        value(ir::BinaryOperator::And, tag("/\\")),
        value(ir::BinaryOperator::Or, tag("\\/")),
        value(ir::BinaryOperator::Xor, tag("#")),
        value(ir::BinaryOperator::Shl, tag("<<")),
        value(ir::BinaryOperator::AShr, tag(">>>")),
        value(ir::BinaryOperator::LShr, tag(">>")),
    )),
    simple_expression
);

binary_expression!(
    mul_div_expression,
    alt((
        value(ir::BinaryOperator::Mul, tag("*")),
        value(ir::BinaryOperator::UDiv, tag("/")),
    )),
    alt((bitwise_expression, simple_expression))
);

binary_expression!(
    add_sub_expression,
    alt((
        value(ir::BinaryOperator::Add, tag("+")),
        value(ir::BinaryOperator::Sub, tag("-")),
    )),
    alt((mul_div_expression, bitwise_expression, simple_expression))
);

binary_expression!(
    compare_expression,
    alt((
        value(ir::BinaryOperator::r#Eq, tag("=")),
        value(ir::BinaryOperator::Neq, tag("\\=")),
        value(ir::BinaryOperator::SLe, tag("<=")),
        value(ir::BinaryOperator::SLt, tag("<")),
        value(ir::BinaryOperator::SGe, tag(">=")),
        value(ir::BinaryOperator::SGt, tag(">")),
    )),
    alt((
        add_sub_expression,
        mul_div_expression,
        bitwise_expression,
        simple_expression,
    ))
);

fn binary_function(input: &str) -> IResult<&str, Box<ir::Expression>> {
    let function = alt((
        value(ir::BinaryOperator::ULe, tag("ule")),
        value(ir::BinaryOperator::ULt, tag("ult")),
        value(ir::BinaryOperator::UGe, tag("uge")),
        value(ir::BinaryOperator::UGt, tag("ugt")),
        value(ir::BinaryOperator::And, tag("and")),
        value(ir::BinaryOperator::Or, tag("or")),
        value(ir::BinaryOperator::Xor, tag("xor")),
        value(ir::BinaryOperator::URem, tag("rem")),
        value(ir::BinaryOperator::SRem, tag("srem")),
        value(ir::BinaryOperator::SMod, tag("mod")),
    ));
    map(
        tuple((
            preceded(space0, function),
            tag("("),
            preceded(space0, expression),
            preceded(space0, tag(",")),
            preceded(space0, expression),
            preceded(space0, tag(")")),
        )),
        |(op, _, lhs, _, rhs, _)| Box::new(ir::Expression::BinaryExpression { op, lhs, rhs }),
    )(input)
}

fn clasped_expression(input: &str) -> IResult<&str, Box<ir::Expression>> {
    map(
        tuple((
            preceded(space0, tag("(")),
            preceded(space0, expression),
            preceded(space0, tag(")")),
        )),
        |(_, e, _)| e,
    )(input)
}

fn conditional_expression(input: &str) -> IResult<&str, Box<ir::Expression>> {
    map(
        tuple((
            preceded(space0, tag("ite(")),
            preceded(space0, expression),
            preceded(space0, tag(",")),
            preceded(space0, expression),
            preceded(space0, tag(",")),
            preceded(space0, expression),
            preceded(space0, tag(")")),
        )),
        |(_, e1, _, e2, _, e3, _)| {
            Box::new(ir::Expression::ConditionalExpression {
                cond: e1,
                then: e2,
                r#else: e3,
            })
        },
    )(input)
}

fn simple_expression(input: &str) -> IResult<&str, Box<ir::Expression>> {
    alt((
        clasped_expression,
        conditional_expression,
        binary_function,
        unary_expression,
        register_ref,
        number_literal,
    ))(input)
}

fn expression(input: &str) -> IResult<&str, Box<ir::Expression>> {
    alt((
        compare_expression,
        add_sub_expression,
        mul_div_expression,
        bitwise_expression,
        simple_expression,
    ))(input)
}

fn label(input: &str) -> IResult<&str, String> {
    terminated(identifier, tag(":"))(input)
}

fn target_location(input: &str) -> IResult<&str, ir::Target> {
    map(identifier, ir::Target::Label)(input)
}

fn target_label(input: &str) -> IResult<&str, ir::Target> {
    map(numeric, ir::Target::Location)(input)
}

fn target(input: &str) -> IResult<&str, ir::Target> {
    alt((target_location, target_label))(input)
}

fn skip_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    value(ir::Instruction::skip(), tag("skip"))(input)
}

fn barrier_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    value(ir::Instruction::barrier(), tag("spbarr"))(input)
}

fn assignment_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((
            preceded(space0, identifier),
            preceded(space0, tag("<-")),
            preceded(space0, expression),
        )),
        |(reg, _, expr)| ir::Instruction::assign(reg, expr),
    )(input)
}

fn conditional_assignment_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((
            preceded(space0, tag("cmov")),
            preceded(space0, expression),
            preceded(space0, tag(",")),
            preceded(space0, identifier),
            preceded(space0, tag("<-")),
            preceded(space0, expression),
        )),
        |(_, cond, _, reg, _, expr)| ir::Instruction::assign_if(cond, reg, expr),
    )(input)
}

fn load_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((
            preceded(space0, tag("load")),
            preceded(space0, identifier),
            preceded(space0, tag(",")),
            preceded(space0, expression),
        )),
        |(_, reg, _, addr)| ir::Instruction::load(reg, addr),
    )(input)
}

fn store_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((
            preceded(space0, tag("store")),
            preceded(space0, identifier),
            preceded(space0, tag(",")),
            preceded(space0, expression),
        )),
        |(_, reg, _, addr)| ir::Instruction::store(reg, addr),
    )(input)
}

fn jump_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((preceded(space0, tag("jmp")), preceded(space0, target))),
        |(_, target)| ir::Instruction::jump(target),
    )(input)
}

fn branch_if_zero_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((
            preceded(space0, tag("beqz")),
            preceded(space0, identifier),
            preceded(space0, tag(",")),
            preceded(space0, target),
        )),
        |(_, reg, _, target)| ir::Instruction::branch_if_zero(reg, target),
    )(input)
}

fn instruction(input: &str) -> IResult<&str, ir::Instruction> {
    alt((
        skip_instruction,
        barrier_instruction,
        assignment_instruction,
        conditional_assignment_instruction,
        load_instruction,
        store_instruction,
        jump_instruction,
        branch_if_zero_instruction,
    ))(input)
}

fn labeled_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((label, preceded(multispace0, instruction))),
        |(lbl, mut inst)| {
            inst.set_label(lbl);
            inst
        },
    )(input)
}

#[cfg(test)]
mod tests {
    #[test]
    fn parse_identifier() {
        assert_eq!(super::identifier("rax"), Ok(("", "rax".to_string())));
        assert_eq!(super::identifier("rax%"), Ok(("%", "rax".to_string())));
        assert_eq!(super::identifier("r3"), Ok(("", "r3".to_string())));
        assert_eq!(super::identifier("r_3"), Ok(("", "r_3".to_string())));
        assert_eq!(super::identifier("r3&"), Ok(("&", "r3".to_string())));
    }

    #[test]
    fn parse_dec_num() {
        assert_eq!(super::dec_num("0"), Ok(("", 0)));
        assert_eq!(super::dec_num("3"), Ok(("", 3)));
        assert_eq!(super::dec_num("42"), Ok(("", 42)));
    }

    #[test]
    fn parse_hex_num() {
        assert_eq!(super::hex_num("0x0"), Ok(("", 0)));
        assert_eq!(super::hex_num("0xC0ffee"), Ok(("", 12648430)));
    }

    #[test]
    fn parse_number_literal() {
        assert_eq!(
            super::expression("042"),
            Ok(("", Box::new(super::ir::Expression::NumberLiteral(42))))
        );
        assert_eq!(
            super::expression("0x42"),
            Ok(("", Box::new(super::ir::Expression::NumberLiteral(66))))
        );
    }

    #[test]
    fn parse_register_ref() {
        assert_eq!(
            super::expression("rax"),
            Ok((
                "",
                Box::new(super::ir::Expression::RegisterRef("rax".to_string()))
            ))
        );
    }

    macro_rules! parse_unary_expressions {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (op_text, op_type) = $value;
                let input = format!("{} 42", op_text);
                assert_eq!(
                    super::expression(&input),
                    Ok((
                        "",
                        Box::new(super::ir::Expression::UnaryExpression {
                            expr: Box::new(super::ir::Expression::NumberLiteral(42)),
                            op: op_type,
                        })
                    ))
                );
            }
        )*
        }
    }
    parse_unary_expressions! {
        parse_unary_expression_neg: ("-", super::ir::UnaryOperator::Neg),
        parse_unary_expression_not: ("~", super::ir::UnaryOperator::Not),
    }

    macro_rules! parse_binary_expressions {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (op_text, op_type) = $value;
                let input = format!("42 {} x", op_text);
                assert_eq!(
                    super::expression(&input),
                    Ok((
                        "",
                        Box::new(super::ir::Expression::BinaryExpression {
                            lhs: Box::new(super::ir::Expression::NumberLiteral(42)),
                            rhs: Box::new(super::ir::Expression::RegisterRef(String::from("x"))),
                            op: op_type,
                        })
                    ))
                );
            }
        )*
        }
    }
    parse_binary_expressions! {
        parse_compare_expression_eq: ("=", super::ir::BinaryOperator::r#Eq),
        parse_compare_expression_neq: ("\\=", super::ir::BinaryOperator::Neq),
        parse_compare_expression_sle: ("<=", super::ir::BinaryOperator::SLe),
        parse_compare_expression_slt: ("<", super::ir::BinaryOperator::SLt),
        parse_compare_expression_sge: (">=", super::ir::BinaryOperator::SGe),
        parse_compare_expression_sgt: (">", super::ir::BinaryOperator::SGt),

        parse_bitwise_expression_and: ("/\\", super::ir::BinaryOperator::And),
        parse_bitwise_expression_or: ("\\/", super::ir::BinaryOperator::Or),
        parse_bitwise_expression_xor: ("#", super::ir::BinaryOperator::Xor),
        parse_bitwise_expression_shl: ("<<", super::ir::BinaryOperator::Shl),
        parse_bitwise_expression_ashr: (">>>", super::ir::BinaryOperator::AShr),
        parse_bitwise_expression_lshr: (">>", super::ir::BinaryOperator::LShr),

        parse_arithmetic_expression_add: ("+", super::ir::BinaryOperator::Add),
        parse_arithmetic_expression_sub: ("-", super::ir::BinaryOperator::Sub),
        parse_arithmetic_expression_mul: ("*", super::ir::BinaryOperator::Mul),
        parse_arithmetic_expression_udiv: ("/", super::ir::BinaryOperator::UDiv),
    }

    macro_rules! parse_binary_functions {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (op_text, op_type) = $value;
                let input = format!("{}(42, x)", op_text);
                assert_eq!(
                    super::expression(&input),
                    Ok((
                        "",
                        Box::new(super::ir::Expression::BinaryExpression {
                            lhs: Box::new(super::ir::Expression::NumberLiteral(42)),
                            rhs: Box::new(super::ir::Expression::RegisterRef(String::from("x"))),
                            op: op_type,
                        })
                    ))
                );
            }
        )*
        }
    }
    parse_binary_functions! {
        parse_binary_function_and: ("and", super::ir::BinaryOperator::And),
        parse_binary_function_or: ("or", super::ir::BinaryOperator::Or),
        parse_binary_function_xor: ("xor", super::ir::BinaryOperator::Xor),
        parse_binary_function_urem: ("rem", super::ir::BinaryOperator::URem),
        parse_binary_function_srem: ("srem", super::ir::BinaryOperator::SRem),
        parse_binary_function_smod: ("mod", super::ir::BinaryOperator::SMod),
        parse_binary_function_ule: ("ule", super::ir::BinaryOperator::ULe),
        parse_binary_function_ult: ("ult", super::ir::BinaryOperator::ULt),
        parse_binary_function_uge: ("uge", super::ir::BinaryOperator::UGe),
        parse_binary_function_ugt: ("ugt", super::ir::BinaryOperator::UGt),
    }

    #[test]
    fn parse_clasped_expression() {
        assert_eq!(
            super::expression("(1)"),
            Ok(("", Box::new(super::ir::Expression::NumberLiteral(1))))
        );
    }

    #[test]
    fn parse_conditional_expression() {
        assert_eq!(
            super::expression("ite(1, 2, 3)"),
            Ok((
                "",
                Box::new(super::ir::Expression::ConditionalExpression {
                    cond: Box::new(super::ir::Expression::NumberLiteral(1)),
                    then: Box::new(super::ir::Expression::NumberLiteral(2)),
                    r#else: Box::new(super::ir::Expression::NumberLiteral(3)),
                })
            ))
        );
    }

    #[test]
    fn check_operator_precedence_bitwise_mul() {
        assert_eq!(
            super::expression("1 << 2 * 3 << 4"),
            Ok((
                "",
                Box::new(super::ir::Expression::BinaryExpression {
                    lhs: Box::new(super::ir::Expression::BinaryExpression {
                        lhs: Box::new(super::ir::Expression::NumberLiteral(1)),
                        rhs: Box::new(super::ir::Expression::NumberLiteral(2)),
                        op: super::ir::BinaryOperator::Shl,
                    }),
                    rhs: Box::new(super::ir::Expression::BinaryExpression {
                        lhs: Box::new(super::ir::Expression::NumberLiteral(3)),
                        rhs: Box::new(super::ir::Expression::NumberLiteral(4)),
                        op: super::ir::BinaryOperator::Shl,
                    }),
                    op: super::ir::BinaryOperator::Mul,
                })
            ))
        );
    }

    #[test]
    fn check_operator_precedence_mul_add() {
        assert_eq!(
            super::expression("1 * 2 + 3 * 4"),
            Ok((
                "",
                Box::new(super::ir::Expression::BinaryExpression {
                    lhs: Box::new(super::ir::Expression::BinaryExpression {
                        lhs: Box::new(super::ir::Expression::NumberLiteral(1)),
                        rhs: Box::new(super::ir::Expression::NumberLiteral(2)),
                        op: super::ir::BinaryOperator::Mul,
                    }),
                    rhs: Box::new(super::ir::Expression::BinaryExpression {
                        lhs: Box::new(super::ir::Expression::NumberLiteral(3)),
                        rhs: Box::new(super::ir::Expression::NumberLiteral(4)),
                        op: super::ir::BinaryOperator::Mul,
                    }),
                    op: super::ir::BinaryOperator::Add,
                })
            ))
        );
    }

    #[test]
    fn check_operator_precedence_add_compare() {
        assert_eq!(
            super::expression("1 + 2 = 3 + 4"),
            Ok((
                "",
                Box::new(super::ir::Expression::BinaryExpression {
                    lhs: Box::new(super::ir::Expression::BinaryExpression {
                        lhs: Box::new(super::ir::Expression::NumberLiteral(1)),
                        rhs: Box::new(super::ir::Expression::NumberLiteral(2)),
                        op: super::ir::BinaryOperator::Add,
                    }),
                    rhs: Box::new(super::ir::Expression::BinaryExpression {
                        lhs: Box::new(super::ir::Expression::NumberLiteral(3)),
                        rhs: Box::new(super::ir::Expression::NumberLiteral(4)),
                        op: super::ir::BinaryOperator::Add,
                    }),
                    op: super::ir::BinaryOperator::r#Eq,
                })
            ))
        );
    }

    #[test]
    fn parse_skip_instruction() {
        assert_eq!(
            super::instruction("skip"),
            Ok(("", super::ir::Instruction::skip()))
        );
    }

    #[test]
    fn parse_barrier_instruction() {
        assert_eq!(
            super::instruction("spbarr"),
            Ok(("", super::ir::Instruction::barrier()))
        );
    }

    #[test]
    fn parse_assignment_instruction() {
        assert_eq!(
            super::instruction("x <- 42"),
            Ok((
                "",
                super::ir::Instruction::assign(
                    "x".to_string(),
                    Box::new(super::ir::Expression::NumberLiteral(42))
                )
            ))
        );
    }

    #[test]
    fn parse_conditional_assignment_instruction() {
        assert_eq!(
            super::instruction("cmov 0, x <- 42"),
            Ok((
                "",
                super::ir::Instruction::assign_if(
                    Box::new(super::ir::Expression::NumberLiteral(0)),
                    "x".to_string(),
                    Box::new(super::ir::Expression::NumberLiteral(42))
                )
            ))
        );
    }

    #[test]
    fn parse_load_instruction() {
        assert_eq!(
            super::instruction("load x, 42"),
            Ok((
                "",
                super::ir::Instruction::load(
                    "x".to_string(),
                    Box::new(super::ir::Expression::NumberLiteral(42))
                )
            ))
        );
    }

    #[test]
    fn parse_store_instruction() {
        assert_eq!(
            super::instruction("store x, 42"),
            Ok((
                "",
                super::ir::Instruction::store(
                    "x".to_string(),
                    Box::new(super::ir::Expression::NumberLiteral(42))
                )
            ))
        );
    }

    #[test]
    fn parse_jump_instruction() {
        assert_eq!(
            super::instruction("jmp 42"),
            Ok((
                "",
                super::ir::Instruction::jump(super::ir::Target::Location(42))
            ))
        );
        assert_eq!(
            super::instruction("jmp lbl"),
            Ok((
                "",
                super::ir::Instruction::jump(super::ir::Target::Label("lbl".to_string()))
            ))
        );
    }

    #[test]
    fn parse_branch_if_zero_instruction() {
        assert_eq!(
            super::instruction("beqz x, 42"),
            Ok((
                "",
                super::ir::Instruction::branch_if_zero(
                    "x".to_string(),
                    super::ir::Target::Location(42)
                )
            ))
        );
        assert_eq!(
            super::instruction("beqz x, lbl"),
            Ok((
                "",
                super::ir::Instruction::branch_if_zero(
                    "x".to_string(),
                    super::ir::Target::Label("lbl".to_string())
                )
            ))
        );
    }

    #[test]
    fn parse_label() {
        assert_eq!(super::label("end:"), Ok(("", "end".to_string())));
    }

    #[test]
    fn parse_labeled_instruction_with_space_between() {
        let mut expected_inst = super::ir::Instruction::skip();
        expected_inst.set_label("Then".to_string());
        assert_eq!(
            super::labeled_instruction("Then: skip"),
            Ok(("", expected_inst))
        );
    }

    #[test]
    fn parse_labeled_instruction_with_newline_between() {
        let mut expected_inst = super::ir::Instruction::skip();
        expected_inst.set_label("Then".to_string());
        assert_eq!(
            super::labeled_instruction("Then:\n skip"),
            Ok(("", expected_inst))
        );
    }

    #[test]
    fn parse_well_formatted_program_with_single_instruction() {
        assert_eq!(
            super::parse_program("beqz x, 42\nstore x, 42"),
            Ok(super::ir::Program::new(vec![
                super::ir::Instruction::branch_if_zero(
                    "x".to_string(),
                    super::ir::Target::Location(42)
                ),
                super::ir::Instruction::store(
                    "x".to_string(),
                    Box::new(super::ir::Expression::NumberLiteral(42))
                ),
            ]))
        );
    }

    #[test]
    fn parse_well_formatted_program_with_two_instructions() {
        assert_eq!(
            super::parse_program("beqz x, 42\nstore x, 42"),
            Ok(super::ir::Program::new(vec![
                super::ir::Instruction::branch_if_zero(
                    "x".to_string(),
                    super::ir::Target::Location(42)
                ),
                super::ir::Instruction::store(
                    "x".to_string(),
                    Box::new(super::ir::Expression::NumberLiteral(42))
                ),
            ]))
        );
    }

    #[test]
    fn parse_program_with_multiple_newlines() {
        assert_eq!(
            super::parse_program("beqz x, 42\n\n\nstore x, 42"),
            Ok(super::ir::Program::new(vec![
                super::ir::Instruction::branch_if_zero(
                    "x".to_string(),
                    super::ir::Target::Location(42)
                ),
                super::ir::Instruction::store(
                    "x".to_string(),
                    Box::new(super::ir::Expression::NumberLiteral(42))
                ),
            ]))
        );
    }

    #[test]
    fn parse_program_with_leading_newline() {
        assert_eq!(
            super::parse_program("\nbeqz x, 42"),
            Ok(super::ir::Program::new(vec![
                super::ir::Instruction::branch_if_zero(
                    "x".to_string(),
                    super::ir::Target::Location(42)
                )
            ]))
        );
    }

    #[test]
    fn parse_program_with_trailing_newline() {
        assert_eq!(
            super::parse_program("beqz x, 42\n"),
            Ok(super::ir::Program::new(vec![
                super::ir::Instruction::branch_if_zero(
                    "x".to_string(),
                    super::ir::Target::Location(42)
                )
            ]))
        );
    }

    #[test]
    fn parse_program_with_spaces_and_tabs() {
        assert_eq!(
            super::parse_program("   \tbeqz x, 42\t\n \n\n store x, 42  "),
            Ok(super::ir::Program::new(vec![
                super::ir::Instruction::branch_if_zero(
                    "x".to_string(),
                    super::ir::Target::Location(42)
                ),
                super::ir::Instruction::store(
                    "x".to_string(),
                    Box::new(super::ir::Expression::NumberLiteral(42))
                ),
            ]))
        );
    }

    #[test]
    fn parse_test_program() {
        let src = r#"
            cond <- x < array1_len
            beqz cond, 5
            load v, array1 + x
            load tmp, array2 + v << 8
        "#;

        assert_eq!(
            super::parse_program(src),
            Ok(super::ir::Program::new(vec![
                super::ir::Instruction::assign(
                    "cond".to_string(),
                    Box::new(super::ir::Expression::BinaryExpression {
                        op: super::ir::BinaryOperator::SLt,
                        lhs: Box::new(super::ir::Expression::RegisterRef("x".to_string())),
                        rhs: Box::new(super::ir::Expression::RegisterRef("array1_len".to_string())),
                    })
                ),
                super::ir::Instruction::branch_if_zero(
                    "cond".to_string(),
                    super::ir::Target::Location(5)
                ),
                super::ir::Instruction::load(
                    "v".to_string(),
                    Box::new(super::ir::Expression::BinaryExpression {
                        op: super::ir::BinaryOperator::Add,
                        lhs: Box::new(super::ir::Expression::RegisterRef("array1".to_string())),
                        rhs: Box::new(super::ir::Expression::RegisterRef("x".to_string())),
                    })
                ),
                super::ir::Instruction::load(
                    "tmp".to_string(),
                    Box::new(super::ir::Expression::BinaryExpression {
                        op: super::ir::BinaryOperator::Add,
                        lhs: Box::new(super::ir::Expression::RegisterRef("array2".to_string())),
                        rhs: Box::new(super::ir::Expression::BinaryExpression {
                            op: super::ir::BinaryOperator::Shl,
                            lhs: Box::new(super::ir::Expression::RegisterRef("v".to_string())),
                            rhs: Box::new(super::ir::Expression::NumberLiteral(8)),
                        }),
                    })
                ),
            ]))
        );
    }

    #[test]
    fn parse_test_program_with_labels() {
        let src = r#"
            cond <- x < array1_len
            beqz cond, EndIf
        Then:
            load v, array1 + x
            load tmp, array2 + v << 8
        EndIf:
        "#;

        let mut labeled_load = super::ir::Instruction::load(
            "v".to_string(),
            Box::new(super::ir::Expression::BinaryExpression {
                op: super::ir::BinaryOperator::Add,
                lhs: Box::new(super::ir::Expression::RegisterRef("array1".to_string())),
                rhs: Box::new(super::ir::Expression::RegisterRef("x".to_string())),
            }),
        );
        labeled_load.set_label("Then".to_string());

        let mut program = super::ir::Program::new(vec![
            super::ir::Instruction::assign(
                "cond".to_string(),
                Box::new(super::ir::Expression::BinaryExpression {
                    op: super::ir::BinaryOperator::SLt,
                    lhs: Box::new(super::ir::Expression::RegisterRef("x".to_string())),
                    rhs: Box::new(super::ir::Expression::RegisterRef("array1_len".to_string())),
                }),
            ),
            super::ir::Instruction::branch_if_zero(
                "cond".to_string(),
                super::ir::Target::Label("EndIf".to_string()),
            ),
            labeled_load,
            super::ir::Instruction::load(
                "tmp".to_string(),
                Box::new(super::ir::Expression::BinaryExpression {
                    op: super::ir::BinaryOperator::Add,
                    lhs: Box::new(super::ir::Expression::RegisterRef("array2".to_string())),
                    rhs: Box::new(super::ir::Expression::BinaryExpression {
                        op: super::ir::BinaryOperator::Shl,
                        lhs: Box::new(super::ir::Expression::RegisterRef("v".to_string())),
                        rhs: Box::new(super::ir::Expression::NumberLiteral(8)),
                    }),
                }),
            ),
        ]);
        program.set_end_label("EndIf".to_string());

        assert_eq!(super::parse_program(src), Ok(program));
    }

    #[test]
    fn parse_erroneous_program() {
        let src = r#"
            unknowninstruction

        "#;

        assert_eq!(super::parse_program(src), Err("Failed to parse program!"));
    }
}
