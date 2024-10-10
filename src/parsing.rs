// BNF:
// <ws> ::= (" " | "\t")*
// <number> ::= <digit> | <number> <digit>
// <digit> ::= [0-9]
// <letter> ::= "A" | "B" | "C" | "D" | "E" | "F" | "G" | "H" | "I" | "J" | "K" | "L" | "M" | "N" | "O" | "P" | "Q" | "R" | "S" | "T" | "U" | "V" | "W" | "X" | "Y" | "Z" | "a" | "b" | "c" | "d" | "e" | "f" | "g" | "h" | "i" | "j" | "k" | "l" | "m" | "n" | "o" | "p" | "q" | "r" | "s" | "t" | "u" | "v" | "w" | "x" | "y" | "z"
// <binop> ::= "+" | "-" | "*" | "/" | "%"
// <unaryop> ::= "-"
// <binary> ::= <expr> <ws> <binop> <ws> <expr>
// <unary> ::= <unaryop> <ws> <expr>
// <ident> ::= <letter> | <ident> <letter> | <ident> "_"
// <bool> ::= "true" | "false"
// <literal> ::= <number> | <bool>
// <decl> ::= <ident> <ws> ":=" <ws> <expr>
// <assign> ::= <ident> <ws> "=" <ws> <expr>
// <expr> ::= "(" <ws> <expr> <ws> ")" | <ident> | <binary> | <unary> | <literal> | <assign> | <if> | <break> | <loop> | <call>
// <args> ::= <expr> | <args> "," <expr>
// <stmt> ::= <expr> ";" | <decl> ";"
// <block> ::= "{" <ws> <stmt>* <ws> "}"
// <sig_args> ::= <ident> | <sig_args> <ws> "," <ws> <ident>
// <sig> ::= <ident> <ws> "::" <ws> "(" <sig_args> ")"
// <function> ::= <sig> <ws> <block>
// <break> ::= "break"
// <if> ::= "if" <ws> <block> | "if" <ws> <block> <ws> "else" <ws> <block>
// <loop> ::= "loop" <ws> <block>
// <call> ::= <ident> <ws> "(" <args> ")"

use std::iter::Peekable;

use itertools::{peek_nth, PeekNth};

use crate::tokenizing::{self, SymbolKind, Token};

#[derive(Debug)]
pub struct Module {
    name: String,
    body: Block,
}

#[derive(Debug)]
pub enum ItemKind {
    Function(Function),
}

#[derive(Debug)]
pub struct Item {
    kind: ItemKind,
}

#[derive(Debug)]
pub struct FunctionSignature {
    name: String,
    args: Vec<String>,
}

#[derive(Debug)]
pub struct Function {
    sig: FunctionSignature,
    body: Block,
}

#[derive(Debug)]
pub struct Block {
    stmts: Vec<Statement>,
}

#[derive(Debug)]
pub enum StatementKind {
    Expr(Expr),
    Decl(Declare),
    Item(Item),
}

#[derive(Debug)]
pub struct Statement {
    kind: StatementKind,
}

#[derive(Debug)]
pub struct Declare {
    name: String,
    value: Option<Expr>,
}

#[derive(Debug)]
pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,

    And,
    Or,
    Eq,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,

    Assign,
}

impl TryFrom<SymbolKind> for BinOpKind {
    type Error = ();

    fn try_from(kind: SymbolKind) -> Result<Self, Self::Error> {
        match kind {
            SymbolKind::Plus => Ok(Self::Add),
            SymbolKind::Minus => Ok(Self::Sub),
            SymbolKind::Star => Ok(Self::Mul),
            SymbolKind::Slash => Ok(Self::Div),
            SymbolKind::Percent => Ok(Self::Mod),
            SymbolKind::And => Ok(Self::And),
            SymbolKind::Or => Ok(Self::Or),
            SymbolKind::Equals => Ok(Self::Eq),
            SymbolKind::Neq => Ok(Self::Neq),
            SymbolKind::Lt => Ok(Self::Lt),
            SymbolKind::Gt => Ok(Self::Gt),
            SymbolKind::Leq => Ok(Self::Leq),
            SymbolKind::Geq => Ok(Self::Geq),
            SymbolKind::AssignEq => Ok(Self::Assign),
            _ => Err(()),
        }
    }
}

impl BinOpKind {
    pub fn precedence(&self) -> u8 {
        match self {
            Self::Or => 1,
            Self::And => 2,
            Self::Eq | Self::Neq | Self::Lt | Self::Gt | Self::Leq | Self::Geq => 3,
            Self::Add | Self::Sub => 4,
            Self::Mul | Self::Div | Self::Mod => 5,
            _ => 0,
        }
    }
}

#[derive(Debug)]
pub enum UnaryOpKind {
    Neg,
    Not,
}

#[derive(Debug)]
pub enum LiteralKind {
    Number(f64),
    Bool(bool),
}

#[derive(Debug)]
pub enum ExprKind {
    Literal(LiteralKind),
    Var(String),
    BinOp {
        kind: BinOpKind,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    UnaryOp {
        kind: UnaryOpKind,
        rhs: Box<Expr>,
    },
    Call {
        name: String,
        args: Vec<Expr>,
    },
    If {
        cond: Box<Expr>,
        then: Block,
        or: Option<Block>,
    },
    Loop {
        body: Block,
    },
    Break,
    Return(Option<Box<Expr>>),
}

#[derive(Debug)]
pub struct Expr {
    kind: ExprKind,
}

pub struct Parser<I: Iterator<Item = Token>> {
    tokens: PeekNth<I>,
}

#[derive(Debug)]
pub enum ParsingError {
    UnexpectedToken(Token),
    UnexpectedEndOfInput,
}

pub type ParsingResult<T> = Result<T, ParsingError>;

impl<I: Iterator<Item = tokenizing::Token>> Parser<I> {
    pub fn new(tokens: I) -> Self {
        Self {
            tokens: peek_nth(tokens),
        }
    }

    pub fn peek(&mut self) -> ParsingResult<&Token> {
        self.tokens.peek().ok_or(ParsingError::UnexpectedEndOfInput)
    }

    pub fn peek_nth(&mut self, n: usize) -> ParsingResult<&Token> {
        self.tokens
            .peek_nth(n)
            .ok_or(ParsingError::UnexpectedEndOfInput)
    }

    pub fn eat(&mut self) -> ParsingResult<Token> {
        self.tokens.next().ok_or(ParsingError::UnexpectedEndOfInput)
    }

    pub fn parse_expr(&mut self, precedence: u8) -> ParsingResult<Expr> {
        let lhs_token = self.eat()?;

        let mut lhs = match lhs_token.kind {
            tokenizing::TokenKind::Number(n) => Expr {
                kind: ExprKind::Literal(LiteralKind::Number(n)),
            },
            tokenizing::TokenKind::Ident(ident) => {
                if ident == "true" || ident == "false" {
                    Expr {
                        kind: ExprKind::Literal(LiteralKind::Bool(ident == "true")),
                    }
                } else {
                    Expr {
                        kind: ExprKind::Var(ident),
                    }
                }
            }
            tokenizing::TokenKind::Symbol(ref symbol) => match symbol {
                SymbolKind::LParen => {
                    let expr = self.parse_expr(0)?;
                    let _ = self.eat()?;
                    expr
                }
                SymbolKind::Minus => {
                    let rhs = self.parse_expr(6)?;
                    Expr {
                        kind: ExprKind::UnaryOp {
                            kind: UnaryOpKind::Neg,
                            rhs: Box::new(rhs),
                        },
                    }
                }
                SymbolKind::Not => {
                    let rhs = self.parse_expr(6)?;
                    Expr {
                        kind: ExprKind::UnaryOp {
                            kind: UnaryOpKind::Not,
                            rhs: Box::new(rhs),
                        },
                    }
                }
                _ => return Err(ParsingError::UnexpectedToken(lhs_token)),
            },
        };

        // Left hand recursion

        loop {
            let rhs_token = self.peek()?;
            if rhs_token.kind == tokenizing::TokenKind::Symbol(tokenizing::SymbolKind::Semicolon) {
                break;
            }
            let op = match rhs_token.kind {
                tokenizing::TokenKind::Symbol(ref symbol) => {
                    match BinOpKind::try_from(symbol.clone()) {
                        Ok(op) => op,
                        Err(_) => break,
                    }
                }
                _ => break,
            };

            let rhs_precedence = op.precedence();
            if rhs_precedence < precedence {
                break;
            }

            let _ = self.eat()?;
            let rhs = self.parse_expr(rhs_precedence)?;
            lhs = Expr {
                kind: ExprKind::BinOp {
                    kind: op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                },
            };
        }

        Ok(lhs)
    }

    pub fn parse_ident(&mut self) -> ParsingResult<Statement> {
        let value = {
            let token = self.peek()?;
            match token.kind.clone() {
                tokenizing::TokenKind::Ident(s) => s,
                _ => return Err(ParsingError::UnexpectedToken(token.clone())),
            }
        };
        match value.as_str() {
            "true" | "false" => {
                let _ = self.eat()?;
                let kind = ExprKind::Literal(LiteralKind::Bool(value == "true"));
                Ok(Statement {
                    kind: StatementKind::Expr(Expr { kind }),
                })
            }
            "if" => {
                let _ = self.eat()?;
                let cond = self.parse_expr(0)?;
                let then = self.parse_block()?;
                let or = if let Ok(token) = self.peek() {
                    if token.kind == tokenizing::TokenKind::Ident("else".to_string()) {
                        self.eat()?;
                        Some(self.parse_block()?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let kind = ExprKind::If {
                    cond: Box::new(cond),
                    then,
                    or,
                };
                Ok(Statement {
                    kind: StatementKind::Expr(Expr { kind }),
                })
            }
            "loop" => {
                let _ = self.eat()?;
                let body = self.parse_block()?;
                let kind = ExprKind::Loop { body };
                Ok(Statement {
                    kind: StatementKind::Expr(Expr { kind }),
                })
            }
            "break" => {
                let _ = self.eat()?;
                let kind = ExprKind::Break;
                Ok(Statement {
                    kind: StatementKind::Expr(Expr { kind }),
                })
            }
            "return" => {
                let _ = self.eat()?;
                let value = if let Ok(token) = self.peek() {
                    if token.kind
                        == tokenizing::TokenKind::Symbol(tokenizing::SymbolKind::Semicolon)
                    {
                        None
                    } else {
                        Some(self.parse_expr(0)?)
                    }
                } else {
                    None
                };

                // Thank you clippy lol               vvvvvvvv
                let kind = ExprKind::Return(value.map(Box::new));
                Ok(Statement {
                    kind: StatementKind::Expr(Expr { kind }),
                })
            }
            _ => {
                let next = self.peek_nth(1)?;
                match next.kind {
                    tokenizing::TokenKind::Symbol(tokenizing::SymbolKind::Walrus) => {
                        self.parse_local()
                    }
                    tokenizing::TokenKind::Symbol(tokenizing::SymbolKind::ColonColon) => {
                        self.parse_function()
                    }
                    tokenizing::TokenKind::Symbol(tokenizing::SymbolKind::LParen) => {
                        todo!("Parse function call")
                    }
                    _ => {
                        let expr = self.parse_expr(0)?;
                        Ok(Statement {
                            kind: StatementKind::Expr(expr),
                        })
                    }
                }
            }
        }
    }

    pub fn parse_local(&mut self) -> ParsingResult<Statement> {
        let name = self.eat()?;
        let name = match name.kind {
            tokenizing::TokenKind::Ident(s) => s,
            _ => return Err(ParsingError::UnexpectedToken(name)),
        };
        let _walrus = self.eat()?;
        let value = self.parse_expr(0)?;
        Ok(Statement {
            kind: StatementKind::Decl(Declare {
                name,
                value: Some(value),
            }),
        })
    }

    pub fn parse_function(&mut self) -> ParsingResult<Statement> {
        let name = self.eat()?;
        let name = match name.kind {
            tokenizing::TokenKind::Ident(s) => s,
            _ => return Err(ParsingError::UnexpectedToken(name)),
        };

        let _coloncolon = self.eat()?;
        let _lparen = self.eat()?;
        let mut args = Vec::new();
        loop {
            let token = self.eat()?;
            match token.kind {
                tokenizing::TokenKind::Ident(s) => args.push(s),
                tokenizing::TokenKind::Symbol(tokenizing::SymbolKind::Comma) => (),
                tokenizing::TokenKind::Symbol(tokenizing::SymbolKind::RParen) => break,
                _ => return Err(ParsingError::UnexpectedToken(token)),
            }
        }

        let sig = FunctionSignature { name, args };
        let body = self.parse_block()?;

        let kind = ItemKind::Function(Function { sig, body });
        Ok(Statement {
            kind: StatementKind::Item(Item { kind }),
        })
    }

    pub fn parse_block(&mut self) -> ParsingResult<Block> {
        // eat '{'
        self.eat()?;
        let mut stmts = Vec::new();
        while let Ok(t) = self.peek() {
            match t.kind {
                tokenizing::TokenKind::Symbol(tokenizing::SymbolKind::RBrace) => break,
                tokenizing::TokenKind::Ident(_) => stmts.push(self.parse_ident()?),
                tokenizing::TokenKind::Symbol(tokenizing::SymbolKind::Semicolon) => {
                    let _ = self.eat()?;
                }
                _ => return Err(ParsingError::UnexpectedToken(t.clone())),
            }
        }

        self.eat()?;
        Ok(Block { stmts })
    }
}

impl<I: Iterator<Item = Token>> Iterator for Parser<I> {
    type Item = ParsingResult<Statement>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.peek().map(|tok| tok.kind.clone()) {
            Err(_) => None,
            Ok(_) => Some(self.parse_ident()),
        }
    }
}
