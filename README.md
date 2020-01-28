# muasm-parser

µASM parser written in Rust using the nom parser combinator library.

## Example

Let `input` be the following µASM program:
```
    cond <- x < array1_len
    beqz cond, EndIf
Then:
    load v, array1 + x
    load tmp, array2 + v << 8
EndIf:
    skip
```

`input` can be parsed as follows:
```rust
let program = parser::parse_program(input).unwrap();
```

This will give the following intermediate representation:
```rust
Program {
    instructions: [
        Instruction {
            operation: Assignment {
                reg: Register { name: "cond" },
                expr: Binary {
                    op: SLt,
                    lhs: RegisterRef(Register { name: "x" }),
                    rhs: RegisterRef(Register { name: "array1_len" }),
                },
            },
            label: None,
        },
        Instruction {
            operation: BranchIfZero {
                reg: Register { name: "cond" },
                target: Label("EndIf"),
            },
            label: None,
        },
        Instruction {
            operation: Load {
                reg: Register { name: "v" },
                addr: Binary {
                    op: Add,
                    lhs: RegisterRef(Register { name: "array1" }),
                    rhs: RegisterRef(Register { name: "x" }),
                },
            },
            label: Some("Then"),
        },
        Instruction {
            operation: Load {
                reg: Register { name: "tmp" },
                addr: Binary {
                    op: Add,
                    lhs: RegisterRef(Register { name: "array2" }),
                    rhs: Binary {
                        op: Shl,
                        lhs: RegisterRef(Register { name: "v" }),
                        rhs: NumberLiteral(8),
                    },
                },
            },
            label: None,
        },
        Instruction {
            operation: Skip,
            label: Some("EndIf"),
        },
    ],
}
```
