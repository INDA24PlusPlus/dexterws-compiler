# Compiler
This compiler will compile a single input file into an executable

## Usage
```bash
./dexterws-compiler <input_file>
```

## BNF
```
<ws> ::= (" " | "\t")*
<number> ::= <digit> | <number> <digit>
<digit> ::= [0-9]
<letter> ::= "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z" | "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
<binop> ::= "+" | "-" | "*" | "/" | "%" | "==" | "!=" | "<" | "<=" | ">" | ">=" | "&&" | "||"
<unaryop> ::= "-"
<binary> ::= <expr> <ws> <binop> <ws> <expr>
<unary> ::= <unaryop> <ws> <expr>
<ident> ::= <letter> | <ident> <letter> | <ident> "_"
<bool> ::= "true" | "false"
<literal> ::= <number> | <bool>
<decl> ::= <ident> <ws> ":=" <ws> <expr>
<assign> ::= <ident> <ws> "=" <ws> <expr>
<expr> ::= "(" <ws> <expr> <ws> ")" | <ident> | <binary> | <unary> | <literal> | <assign> | <if> | <break> | <loop> | <call>
<args> ::= <expr> | <args> "," <expr>
<stmt> ::= <expr> ";" | <decl> ";"
<block> ::= "{" <ws> <stmt>* <ws> "}"
<sig_args> ::= <ident> | <sig_args> <ws> "," <ws> <ident>
<sig> ::= <ident> <ws> "::" <ws> "(" <sig_args> ")"
<function> ::= <sig> <ws> <block>
<break> ::= "break"
<if> ::= "if" <ws> <block> | "if" <ws> <block> <ws> "else" <ws> <block>
<loop> ::= "loop" <ws> <block>
<call> ::= <ident> <ws> "(" <args> ")"
<program> ::= <function>*
```

## Example syntax
```
main :: () {
    x := 5;
    y := 10;
    z := x + y;
    if z == 15 {
        x = 0;
    } else {
        x = 1;
    }
    loop {
        break;
    }
}
```
