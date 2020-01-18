use muasm_parser::parser;
use std::env;
use std::fs;

fn main() {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        panic!("Expected a filename as input");
    }
    let filename = &args[1];
    let src = fs::read_to_string(filename).expect("Could not read file");
    let program = parser::parse_program(&src).expect("Could not parse file");
    println!("{}", program);
}
