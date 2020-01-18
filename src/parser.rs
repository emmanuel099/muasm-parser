use crate::ir;
use nom::{
    branch::alt,
    bytes::complete::{tag, take_while, take_while1},
    character::complete::{char, digit1, hex_digit1, multispace1, not_line_ending, space0, space1},
    combinator::{all_consuming, map, map_res, opt, value},
    multi::{fold_many0, many0},
    sequence::{preceded, terminated, tuple},
    IResult,
};
use std::str::FromStr;

pub fn parse_program(input: &str) -> Result<ir::Program, &'static str> {
    let result = all_consuming(tuple((
        instructions,
        opt(preceded(whitespaces_or_comment, label)),
        whitespaces_or_comment,
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
        preceded(
            whitespaces_or_comment,
            alt((labeled_instruction, instruction)),
        ),
        Vec::new(),
        |mut instructions: Vec<_>, inst| {
            instructions.push(inst);
            instructions
        },
    )(input)
}

fn comment(input: &str) -> IResult<&str, &str> {
    preceded(char('%'), not_line_ending)(input)
}

fn whitespaces_or_comment(input: &str) -> IResult<&str, Vec<&str>> {
    many0(alt((multispace1, comment)))(input)
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

fn register(input: &str) -> IResult<&str, ir::Register> {
    map(identifier, ir::Register::new)(input)
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

fn number_literal(input: &str) -> IResult<&str, ir::Expression> {
    map(numeric, ir::Expression::NumberLiteral)(input)
}

fn register_ref(input: &str) -> IResult<&str, ir::Expression> {
    map(register, ir::Expression::RegisterRef)(input)
}

fn unary_expression(input: &str) -> IResult<&str, ir::Expression> {
    let operator = alt((
        value(ir::UnaryOperator::Neg, char('-')),
        value(ir::UnaryOperator::Not, char('~')),
    ));
    map(
        tuple((operator, preceded(space0, simple_expression))),
        |(op, expr)| ir::Expression::Unary {
            op,
            expr: Box::new(expr),
        },
    )(input)
}

macro_rules! binary_expression {
    ($func:ident, $operator:expr, $operand:expr) => {
        fn $func(input: &str) -> IResult<&str, ir::Expression> {
            map(
                tuple((
                    $operand,
                    preceded(space0, $operator),
                    preceded(space0, $operand),
                )),
                |(lhs, op, rhs)| ir::Expression::Binary {
                    op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            )(input)
        }
    };
}

binary_expression!(
    bitwise_expression,
    alt((
        value(ir::BinaryOperator::And, tag("/\\")),
        value(ir::BinaryOperator::Or, tag("\\/")),
        value(ir::BinaryOperator::Xor, char('#')),
        value(ir::BinaryOperator::Shl, tag("<<")),
        value(ir::BinaryOperator::AShr, tag(">>>")),
        value(ir::BinaryOperator::LShr, tag(">>")),
    )),
    simple_expression
);

binary_expression!(
    mul_div_expression,
    alt((
        value(ir::BinaryOperator::Mul, char('*')),
        value(ir::BinaryOperator::UDiv, char('/')),
    )),
    alt((bitwise_expression, simple_expression))
);

binary_expression!(
    add_sub_expression,
    alt((
        value(ir::BinaryOperator::Add, char('+')),
        value(ir::BinaryOperator::Sub, char('-')),
    )),
    alt((mul_div_expression, bitwise_expression, simple_expression))
);

binary_expression!(
    compare_expression,
    alt((
        value(ir::BinaryOperator::r#Eq, char('=')),
        value(ir::BinaryOperator::Neq, tag("\\=")),
        value(ir::BinaryOperator::SLe, tag("<=")),
        value(ir::BinaryOperator::SLt, char('<')),
        value(ir::BinaryOperator::SGe, tag(">=")),
        value(ir::BinaryOperator::SGt, char('>')),
    )),
    alt((
        add_sub_expression,
        mul_div_expression,
        bitwise_expression,
        simple_expression,
    ))
);

fn binary_function(input: &str) -> IResult<&str, ir::Expression> {
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
            function,
            char('('),
            preceded(space0, expression),
            preceded(space0, char(',')),
            preceded(space0, expression),
            preceded(space0, char(')')),
        )),
        |(op, _, lhs, _, rhs, _)| ir::Expression::Binary {
            op,
            lhs: Box::new(lhs),
            rhs: Box::new(rhs),
        },
    )(input)
}

fn clasped_expression(input: &str) -> IResult<&str, ir::Expression> {
    map(
        tuple((
            char('('),
            preceded(space0, expression),
            preceded(space0, char(')')),
        )),
        |(_, e, _)| e,
    )(input)
}

fn conditional_expression(input: &str) -> IResult<&str, ir::Expression> {
    map(
        tuple((
            tag("ite("),
            preceded(space0, expression),
            preceded(space0, char(',')),
            preceded(space0, expression),
            preceded(space0, char(',')),
            preceded(space0, expression),
            preceded(space0, char(')')),
        )),
        |(_, cond, _, then, _, r#else, _)| ir::Expression::Conditional {
            cond: Box::new(cond),
            then: Box::new(then),
            r#else: Box::new(r#else),
        },
    )(input)
}

fn simple_expression(input: &str) -> IResult<&str, ir::Expression> {
    alt((
        clasped_expression,
        conditional_expression,
        binary_function,
        unary_expression,
        register_ref,
        number_literal,
    ))(input)
}

fn expression(input: &str) -> IResult<&str, ir::Expression> {
    alt((
        compare_expression,
        add_sub_expression,
        mul_div_expression,
        bitwise_expression,
        simple_expression,
    ))(input)
}

fn label(input: &str) -> IResult<&str, String> {
    terminated(identifier, char(':'))(input)
}

fn target(input: &str) -> IResult<&str, ir::Target> {
    alt((
        map(numeric, ir::Target::Location),
        map(identifier, ir::Target::Label),
    ))(input)
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
            register,
            preceded(space0, tag("<-")),
            preceded(space0, expression),
        )),
        |(reg, _, expr)| ir::Instruction::assign(reg, expr),
    )(input)
}

fn conditional_assignment_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((
            tag("cmov"),
            preceded(space1, expression),
            preceded(space0, char(',')),
            preceded(space0, register),
            preceded(space0, tag("<-")),
            preceded(space0, expression),
        )),
        |(_, cond, _, reg, _, expr)| ir::Instruction::assign_if(cond, reg, expr),
    )(input)
}

fn load_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((
            tag("load"),
            preceded(space1, register),
            preceded(space0, char(',')),
            preceded(space0, expression),
        )),
        |(_, reg, _, addr)| ir::Instruction::load(reg, addr),
    )(input)
}

fn store_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((
            tag("store"),
            preceded(space1, register),
            preceded(space0, char(',')),
            preceded(space0, expression),
        )),
        |(_, reg, _, addr)| ir::Instruction::store(reg, addr),
    )(input)
}

fn jump_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((tag("jmp"), preceded(space1, target))),
        |(_, target)| ir::Instruction::jump(target),
    )(input)
}

fn branch_if_zero_instruction(input: &str) -> IResult<&str, ir::Instruction> {
    map(
        tuple((
            tag("beqz"),
            preceded(space1, register),
            preceded(space0, char(',')),
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
        tuple((label, preceded(whitespaces_or_comment, instruction))),
        |(lbl, mut inst)| {
            inst.set_label(lbl);
            inst
        },
    )(input)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_identifier() {
        assert_eq!(identifier("rax"), Ok(("", "rax".to_string())));
        assert_eq!(identifier("rax%"), Ok(("%", "rax".to_string())));
        assert_eq!(identifier("r3"), Ok(("", "r3".to_string())));
        assert_eq!(identifier("r_3"), Ok(("", "r_3".to_string())));
        assert_eq!(identifier("r3&"), Ok(("&", "r3".to_string())));
    }

    #[test]
    fn parse_dec_num() {
        assert_eq!(dec_num("0"), Ok(("", 0)));
        assert_eq!(dec_num("3"), Ok(("", 3)));
        assert_eq!(dec_num("42"), Ok(("", 42)));
    }

    #[test]
    fn parse_hex_num() {
        assert_eq!(hex_num("0x0"), Ok(("", 0)));
        assert_eq!(hex_num("0xC0ffee"), Ok(("", 12_648_430)));
    }

    #[test]
    fn parse_number_literal() {
        assert_eq!(
            expression("042"),
            Ok(("", ir::Expression::NumberLiteral(42)))
        );
        assert_eq!(
            expression("0x42"),
            Ok(("", ir::Expression::NumberLiteral(66)))
        );
    }

    #[test]
    fn parse_register_ref() {
        assert_eq!(
            expression("rax"),
            Ok((
                "",
                ir::Expression::RegisterRef(ir::Register::new("rax".to_string()))
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
                    expression(&input),
                    Ok((
                        "",
                        ir::Expression::Unary {
                            expr: Box::new(ir::Expression::NumberLiteral(42)),
                            op: op_type,
                        }
                    ))
                );
            }
        )*
        }
    }
    parse_unary_expressions! {
        parse_unary_expression_neg: ("-", ir::UnaryOperator::Neg),
        parse_unary_expression_not: ("~", ir::UnaryOperator::Not),
    }

    macro_rules! parse_binary_expressions {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (op_text, op_type) = $value;
                let input = format!("42 {} x", op_text);
                assert_eq!(
                    expression(&input),
                    Ok((
                        "",
                        ir::Expression::Binary {
                            lhs: Box::new(ir::Expression::NumberLiteral(42)),
                            rhs: Box::new(ir::Expression::RegisterRef(ir::Register::new("x".to_string()))),
                            op: op_type,
                        }
                    ))
                );
            }
        )*
        }
    }
    parse_binary_expressions! {
        parse_compare_expression_eq: ("=", ir::BinaryOperator::r#Eq),
        parse_compare_expression_neq: ("\\=", ir::BinaryOperator::Neq),
        parse_compare_expression_sle: ("<=", ir::BinaryOperator::SLe),
        parse_compare_expression_slt: ("<", ir::BinaryOperator::SLt),
        parse_compare_expression_sge: (">=", ir::BinaryOperator::SGe),
        parse_compare_expression_sgt: (">", ir::BinaryOperator::SGt),

        parse_bitwise_expression_and: ("/\\", ir::BinaryOperator::And),
        parse_bitwise_expression_or: ("\\/", ir::BinaryOperator::Or),
        parse_bitwise_expression_xor: ("#", ir::BinaryOperator::Xor),
        parse_bitwise_expression_shl: ("<<", ir::BinaryOperator::Shl),
        parse_bitwise_expression_ashr: (">>>", ir::BinaryOperator::AShr),
        parse_bitwise_expression_lshr: (">>", ir::BinaryOperator::LShr),

        parse_arithmetic_expression_add: ("+", ir::BinaryOperator::Add),
        parse_arithmetic_expression_sub: ("-", ir::BinaryOperator::Sub),
        parse_arithmetic_expression_mul: ("*", ir::BinaryOperator::Mul),
        parse_arithmetic_expression_udiv: ("/", ir::BinaryOperator::UDiv),
    }

    macro_rules! parse_binary_functions {
        ($($name:ident: $value:expr,)*) => {
        $(
            #[test]
            fn $name() {
                let (op_text, op_type) = $value;
                let input = format!("{}(42, x)", op_text);
                assert_eq!(
                    expression(&input),
                    Ok((
                        "",
                        ir::Expression::Binary {
                            lhs: Box::new(ir::Expression::NumberLiteral(42)),
                            rhs: Box::new(ir::Expression::RegisterRef(ir::Register::new("x".to_string()))),
                            op: op_type,
                        }
                    ))
                );
            }
        )*
        }
    }
    parse_binary_functions! {
        parse_binary_function_and: ("and", ir::BinaryOperator::And),
        parse_binary_function_or: ("or", ir::BinaryOperator::Or),
        parse_binary_function_xor: ("xor", ir::BinaryOperator::Xor),
        parse_binary_function_urem: ("rem", ir::BinaryOperator::URem),
        parse_binary_function_srem: ("srem", ir::BinaryOperator::SRem),
        parse_binary_function_smod: ("mod", ir::BinaryOperator::SMod),
        parse_binary_function_ule: ("ule", ir::BinaryOperator::ULe),
        parse_binary_function_ult: ("ult", ir::BinaryOperator::ULt),
        parse_binary_function_uge: ("uge", ir::BinaryOperator::UGe),
        parse_binary_function_ugt: ("ugt", ir::BinaryOperator::UGt),
    }

    #[test]
    fn parse_clasped_expression() {
        assert_eq!(
            expression("(1)"),
            Ok(("", ir::Expression::NumberLiteral(1)))
        );
    }

    #[test]
    fn parse_conditional_expression() {
        assert_eq!(
            expression("ite(1, 2, 3)"),
            Ok((
                "",
                ir::Expression::Conditional {
                    cond: Box::new(ir::Expression::NumberLiteral(1)),
                    then: Box::new(ir::Expression::NumberLiteral(2)),
                    r#else: Box::new(ir::Expression::NumberLiteral(3)),
                }
            ))
        );
    }

    #[test]
    fn check_operator_precedence_bitwise_mul() {
        assert_eq!(
            expression("1 << 2 * 3 << 4"),
            Ok((
                "",
                ir::Expression::Binary {
                    lhs: Box::new(ir::Expression::Binary {
                        lhs: Box::new(ir::Expression::NumberLiteral(1)),
                        rhs: Box::new(ir::Expression::NumberLiteral(2)),
                        op: ir::BinaryOperator::Shl,
                    }),
                    rhs: Box::new(ir::Expression::Binary {
                        lhs: Box::new(ir::Expression::NumberLiteral(3)),
                        rhs: Box::new(ir::Expression::NumberLiteral(4)),
                        op: ir::BinaryOperator::Shl,
                    }),
                    op: ir::BinaryOperator::Mul,
                }
            ))
        );
    }

    #[test]
    fn check_operator_precedence_mul_add() {
        assert_eq!(
            expression("1 * 2 + 3 * 4"),
            Ok((
                "",
                ir::Expression::Binary {
                    lhs: Box::new(ir::Expression::Binary {
                        lhs: Box::new(ir::Expression::NumberLiteral(1)),
                        rhs: Box::new(ir::Expression::NumberLiteral(2)),
                        op: ir::BinaryOperator::Mul,
                    }),
                    rhs: Box::new(ir::Expression::Binary {
                        lhs: Box::new(ir::Expression::NumberLiteral(3)),
                        rhs: Box::new(ir::Expression::NumberLiteral(4)),
                        op: ir::BinaryOperator::Mul,
                    }),
                    op: ir::BinaryOperator::Add,
                }
            ))
        );
    }

    #[test]
    fn check_operator_precedence_add_compare() {
        assert_eq!(
            expression("1 + 2 = 3 + 4"),
            Ok((
                "",
                ir::Expression::Binary {
                    lhs: Box::new(ir::Expression::Binary {
                        lhs: Box::new(ir::Expression::NumberLiteral(1)),
                        rhs: Box::new(ir::Expression::NumberLiteral(2)),
                        op: ir::BinaryOperator::Add,
                    }),
                    rhs: Box::new(ir::Expression::Binary {
                        lhs: Box::new(ir::Expression::NumberLiteral(3)),
                        rhs: Box::new(ir::Expression::NumberLiteral(4)),
                        op: ir::BinaryOperator::Add,
                    }),
                    op: ir::BinaryOperator::r#Eq,
                }
            ))
        );
    }

    #[test]
    fn parse_skip_instruction() {
        assert_eq!(instruction("skip"), Ok(("", ir::Instruction::skip())));
    }

    #[test]
    fn parse_barrier_instruction() {
        assert_eq!(instruction("spbarr"), Ok(("", ir::Instruction::barrier())));
    }

    #[test]
    fn parse_assignment_instruction() {
        assert_eq!(
            instruction("x <- 42"),
            Ok((
                "",
                ir::Instruction::assign(
                    ir::Register::new("x".to_string()),
                    ir::Expression::NumberLiteral(42)
                )
            ))
        );
    }

    #[test]
    fn parse_conditional_assignment_instruction() {
        assert_eq!(
            instruction("cmov 0, x <- 42"),
            Ok((
                "",
                ir::Instruction::assign_if(
                    ir::Expression::NumberLiteral(0),
                    ir::Register::new("x".to_string()),
                    ir::Expression::NumberLiteral(42)
                )
            ))
        );
    }

    #[test]
    fn parse_load_instruction() {
        assert_eq!(
            instruction("load x, 42"),
            Ok((
                "",
                ir::Instruction::load(
                    ir::Register::new("x".to_string()),
                    ir::Expression::NumberLiteral(42)
                )
            ))
        );
    }

    #[test]
    fn parse_store_instruction() {
        assert_eq!(
            instruction("store x, 42"),
            Ok((
                "",
                ir::Instruction::store(
                    ir::Register::new("x".to_string()),
                    ir::Expression::NumberLiteral(42)
                )
            ))
        );
    }

    #[test]
    fn parse_jump_instruction() {
        assert_eq!(
            instruction("jmp 42"),
            Ok(("", ir::Instruction::jump(ir::Target::Location(42))))
        );
        assert_eq!(
            instruction("jmp lbl"),
            Ok((
                "",
                ir::Instruction::jump(ir::Target::Label("lbl".to_string()))
            ))
        );
    }

    #[test]
    fn parse_branch_if_zero_instruction() {
        assert_eq!(
            instruction("beqz x, 42"),
            Ok((
                "",
                ir::Instruction::branch_if_zero(
                    ir::Register::new("x".to_string()),
                    ir::Target::Location(42)
                )
            ))
        );
        assert_eq!(
            instruction("beqz x, lbl"),
            Ok((
                "",
                ir::Instruction::branch_if_zero(
                    ir::Register::new("x".to_string()),
                    ir::Target::Label("lbl".to_string())
                )
            ))
        );
    }

    #[test]
    fn parse_label() {
        assert_eq!(label("end:"), Ok(("", "end".to_string())));
    }

    #[test]
    fn parse_labeled_instruction_with_space_between() {
        let mut expected_inst = ir::Instruction::skip();
        expected_inst.set_label("Then".to_string());
        assert_eq!(labeled_instruction("Then: skip"), Ok(("", expected_inst)));
    }

    #[test]
    fn parse_labeled_instruction_with_newline_between() {
        let mut expected_inst = ir::Instruction::skip();
        expected_inst.set_label("Then".to_string());
        assert_eq!(labeled_instruction("Then:\n skip"), Ok(("", expected_inst)));
    }

    #[test]
    fn parse_well_formatted_program_with_single_instruction() {
        assert_eq!(
            parse_program("beqz x, 42\nstore x, 42"),
            Ok(ir::Program::new(vec![
                ir::Instruction::branch_if_zero(
                    ir::Register::new("x".to_string()),
                    ir::Target::Location(42)
                ),
                ir::Instruction::store(
                    ir::Register::new("x".to_string()),
                    ir::Expression::NumberLiteral(42)
                ),
            ]))
        );
    }

    #[test]
    fn parse_well_formatted_program_with_two_instructions() {
        assert_eq!(
            parse_program("beqz x, 42\nstore x, 42"),
            Ok(ir::Program::new(vec![
                ir::Instruction::branch_if_zero(
                    ir::Register::new("x".to_string()),
                    ir::Target::Location(42)
                ),
                ir::Instruction::store(
                    ir::Register::new("x".to_string()),
                    ir::Expression::NumberLiteral(42)
                ),
            ]))
        );
    }

    #[test]
    fn parse_program_with_multiple_newlines() {
        assert_eq!(
            parse_program("beqz x, 42\n\n\nstore x, 42"),
            Ok(ir::Program::new(vec![
                ir::Instruction::branch_if_zero(
                    ir::Register::new("x".to_string()),
                    ir::Target::Location(42)
                ),
                ir::Instruction::store(
                    ir::Register::new("x".to_string()),
                    ir::Expression::NumberLiteral(42)
                ),
            ]))
        );
    }

    #[test]
    fn parse_program_with_leading_newline() {
        assert_eq!(
            parse_program("\nbeqz x, 42"),
            Ok(ir::Program::new(vec![ir::Instruction::branch_if_zero(
                ir::Register::new("x".to_string()),
                ir::Target::Location(42)
            )]))
        );
    }

    #[test]
    fn parse_program_with_trailing_newline() {
        assert_eq!(
            parse_program("beqz x, 42\n"),
            Ok(ir::Program::new(vec![ir::Instruction::branch_if_zero(
                ir::Register::new("x".to_string()),
                ir::Target::Location(42)
            )]))
        );
    }

    #[test]
    fn parse_program_with_spaces_and_tabs() {
        assert_eq!(
            parse_program("   \tbeqz x, 42\t\n \n\n store x, 42  "),
            Ok(ir::Program::new(vec![
                ir::Instruction::branch_if_zero(
                    ir::Register::new("x".to_string()),
                    ir::Target::Location(42)
                ),
                ir::Instruction::store(
                    ir::Register::new("x".to_string()),
                    ir::Expression::NumberLiteral(42)
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
            parse_program(src),
            Ok(ir::Program::new(vec![
                ir::Instruction::assign(
                    ir::Register::new("cond".to_string()),
                    ir::Expression::Binary {
                        op: ir::BinaryOperator::SLt,
                        lhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                            "x".to_string()
                        ))),
                        rhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                            "array1_len".to_string()
                        ))),
                    }
                ),
                ir::Instruction::branch_if_zero(
                    ir::Register::new("cond".to_string()),
                    ir::Target::Location(5)
                ),
                ir::Instruction::load(
                    ir::Register::new("v".to_string()),
                    ir::Expression::Binary {
                        op: ir::BinaryOperator::Add,
                        lhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                            "array1".to_string()
                        ))),
                        rhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                            "x".to_string()
                        ))),
                    }
                ),
                ir::Instruction::load(
                    ir::Register::new("tmp".to_string()),
                    ir::Expression::Binary {
                        op: ir::BinaryOperator::Add,
                        lhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                            "array2".to_string()
                        ))),
                        rhs: Box::new(ir::Expression::Binary {
                            op: ir::BinaryOperator::Shl,
                            lhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                                "v".to_string()
                            ))),
                            rhs: Box::new(ir::Expression::NumberLiteral(8)),
                        }),
                    }
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

        let mut labeled_load = ir::Instruction::load(
            ir::Register::new("v".to_string()),
            ir::Expression::Binary {
                op: ir::BinaryOperator::Add,
                lhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                    "array1".to_string(),
                ))),
                rhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                    "x".to_string(),
                ))),
            },
        );
        labeled_load.set_label("Then".to_string());

        let mut program = ir::Program::new(vec![
            ir::Instruction::assign(
                ir::Register::new("cond".to_string()),
                ir::Expression::Binary {
                    op: ir::BinaryOperator::SLt,
                    lhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                        "x".to_string(),
                    ))),
                    rhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                        "array1_len".to_string(),
                    ))),
                },
            ),
            ir::Instruction::branch_if_zero(
                ir::Register::new("cond".to_string()),
                ir::Target::Label("EndIf".to_string()),
            ),
            labeled_load,
            ir::Instruction::load(
                ir::Register::new("tmp".to_string()),
                ir::Expression::Binary {
                    op: ir::BinaryOperator::Add,
                    lhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                        "array2".to_string(),
                    ))),
                    rhs: Box::new(ir::Expression::Binary {
                        op: ir::BinaryOperator::Shl,
                        lhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                            "v".to_string(),
                        ))),
                        rhs: Box::new(ir::Expression::NumberLiteral(8)),
                    }),
                },
            ),
        ]);
        program.set_end_label("EndIf".to_string());

        assert_eq!(parse_program(src), Ok(program));
    }

    #[test]
    fn parse_erroneous_program() {
        let src = r#"
            unknowninstruction

        "#;

        assert_eq!(parse_program(src), Err("Failed to parse program!"));
    }

    #[test]
    fn parse_empty_program() {
        assert_eq!(parse_program(""), Ok(ir::Program::new(vec![])));
    }

    #[test]
    fn parse_program_with_single_comment() {
        assert_eq!(parse_program("% comment"), Ok(ir::Program::new(vec![])));
    }

    #[test]
    fn parse_test_program_with_comments() {
        let src = r#"
            % test program
            % start
            c <- x < y
            beqz c, 3 % jump to end
            skip
            % end
        "#;

        assert_eq!(
            parse_program(src),
            Ok(ir::Program::new(vec![
                ir::Instruction::assign(
                    ir::Register::new("c".to_string()),
                    ir::Expression::Binary {
                        op: ir::BinaryOperator::SLt,
                        lhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                            "x".to_string()
                        ))),
                        rhs: Box::new(ir::Expression::RegisterRef(ir::Register::new(
                            "y".to_string()
                        ))),
                    }
                ),
                ir::Instruction::branch_if_zero(
                    ir::Register::new("c".to_string()),
                    ir::Target::Location(3)
                ),
                ir::Instruction::skip(),
            ]))
        );
    }
}
