use muasm_parser::parser;

fn main() {
    let src = r#"
        cond <- x < array1_len
        beqz cond, EndIf
    Then:
        load v, array1 + x
        load tmp, array2 + v << 8
    EndIf:"#;

    let program = parser::parse_program(src).unwrap();
    println!("{}", program);
}
