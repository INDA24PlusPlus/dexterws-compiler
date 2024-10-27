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

use crate::tokenizing::{NumberKind, SymbolKind, Token, TokenKind, TokenLocation};

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct NodeSpan {
    pub start: TokenLocation,
    pub end: TokenLocation,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Type {
    Int,
    Float,
    Bool,
    Void,
}

impl std::fmt::Display for Type {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        let s = match self {
            Type::Int => "int",
            Type::Float => "float",
            Type::Bool => "bool",
            Type::Void => "void",
        };
        write!(f, "{}", s)
    }
}

#[derive(Debug)]
pub struct Ast {
    pub nodes: Vec<Item>,
}

#[derive(Debug)]
pub enum ItemKind {
    Function(Function),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Identifier {
    pub value: String,
    pub span: NodeSpan,
}

#[derive(Debug)]
pub struct Item {
    pub kind: ItemKind,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FunctionSignature {
    pub name: Identifier,
    pub args: Vec<(String, Type)>,
    pub ret_ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Function {
    pub sig: FunctionSignature,
    pub body: Block,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Block {
    pub stmts: Vec<Statement>,
}

#[derive(Debug, Clone, PartialEq)]
pub enum StatementKind {
    Expr(Expr),
    Decl(Assignment),
    Assign(Assignment),
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

#[derive(Debug, Clone, PartialEq)]
pub struct Statement {
    pub kind: StatementKind,
    pub span: NodeSpan,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Assignment {
    pub name: Identifier,
    pub value: Expr,
}

#[derive(Debug, Clone, PartialEq, Eq)]
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
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct BinOp {
    pub kind: BinOpKind,
    pub lhs: Box<Expr>,
    pub rhs: Box<Expr>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum UnaryOpKind {
    Neg,
    Not,
    Cast(Type),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum LiteralKind {
    Int(i64),
    Float(f64),
    Bool(bool),
}

#[derive(Debug, Clone, PartialEq)]
pub enum ExprKind {
    Literal(LiteralKind),
    Var(Identifier),
    BinOp(BinOp),
    UnaryOp { kind: UnaryOpKind, rhs: Box<Expr> },
    Call { name: String, args: Vec<Expr> },
}

#[derive(Debug, Clone, PartialEq)]
pub struct Expr {
    pub kind: ExprKind,
    pub span: NodeSpan,
}

pub struct Parser<I: Iterator<Item = Token>> {
    tokens: PeekNth<I>,
    last_token: Option<Token>,
}

#[derive(Debug)]
pub enum ParsingError {
    UnexpectedToken(Token),
    UnexpectedEndOfInput,
}

pub type ParsingResult<T> = Result<T, ParsingError>;

impl<I: Iterator<Item = Token>> Parser<I> {
    pub fn new(tokens: I) -> Self {
        Self {
            tokens: peek_nth(tokens),
            last_token: None,
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
        let token = self.tokens.next().ok_or(ParsingError::UnexpectedEndOfInput)?;
        self.last_token = Some(token.clone());
        Ok(token)
    }

    pub fn eat_and_assert(&mut self, kind: TokenKind) -> ParsingResult<Token> {
        let token = self.eat()?;
        if token.kind == kind {
            Ok(token)
        } else {
            Err(ParsingError::UnexpectedToken(token))
        }
    }

    pub fn get_next_location(&mut self) -> ParsingResult<TokenLocation> {
        let token = self.peek()?;
        Ok(token.location.clone())
    }

    pub fn get_location(&mut self) -> ParsingResult<TokenLocation> {
        let token = self.last_token.clone();
        match token {
            Some(token) => Ok(token.location.clone()),
            None => self.get_next_location(),
        }
    }

    pub fn parse_expr(&mut self, precedence: u8) -> ParsingResult<Expr> {
        let lhs_token = self.eat()?;
        let start = lhs_token.location.clone();

        let lhs_kind = match lhs_token.kind {
            TokenKind::Number(n) => {
                let kind = match n {
                    NumberKind::Int(i) => LiteralKind::Int(i),
                    NumberKind::Float(f) => LiteralKind::Float(f),
                };
                ExprKind::Literal(kind)
            }
            TokenKind::Ident(ident) => {
                if ident == "true" || ident == "false" {
                    ExprKind::Literal(LiteralKind::Bool(ident == "true"))
                } else if let TokenKind::Symbol(SymbolKind::LParen) = self.peek()?.kind {
                    let mut args = Vec::new();
                    loop {
                        let token = self.eat()?;
                        match token.kind {
                            TokenKind::Symbol(SymbolKind::RParen) => break,
                            _ => {
                                let expr = self.parse_expr(0)?;
                                args.push(expr);
                            }
                        }
                    }
                    ExprKind::Call {
                        name: ident,
                        args,
                    }
                } else {
                    ExprKind::Var(Identifier {
                        value: ident,
                        span: NodeSpan {
                            start: lhs_token.location,
                            end: lhs_token.location,
                        },
                    })
                }
            }
            TokenKind::Symbol(ref symbol) => match symbol {
                SymbolKind::LParen => {
                    let expr = self.parse_expr(0)?;
                    let _ = self.eat()?;
                    expr.kind
                }
                SymbolKind::Minus => {
                    let rhs = self.parse_expr(6)?;
                    ExprKind::UnaryOp {
                        kind: UnaryOpKind::Neg,
                        rhs: Box::new(rhs),
                    }
                }
                SymbolKind::Not => {
                    let rhs = self.parse_expr(6)?;
                    ExprKind::UnaryOp {
                        kind: UnaryOpKind::Not,
                        rhs: Box::new(rhs),
                    }
                }
                SymbolKind::At => {
                    let ty = self.parse_type()?;
                    let rhs = self.parse_expr(6)?;
                    ExprKind::UnaryOp {
                        kind: UnaryOpKind::Cast(ty),
                        rhs: Box::new(rhs),
                    }
                }
                _ => return Err(ParsingError::UnexpectedToken(lhs_token)),
            },
        };

        let mut lhs = Expr {
            kind: lhs_kind,
            span: NodeSpan {
                start,
                end: self.get_location()?,
            },
        };

        // Left hand recursion

        loop {
            let rhs_token = self.peek()?;
            if rhs_token.kind == TokenKind::Symbol(SymbolKind::Semicolon) {
                break;
            }
            let op = match rhs_token.kind {
                TokenKind::Symbol(ref symbol) => {
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
                kind: ExprKind::BinOp(BinOp {
                    kind: op,
                    lhs: Box::new(lhs),
                    rhs: Box::new(rhs),
                }),
                span: NodeSpan {
                    start,
                    end: self.get_location()?,
                },
            };
        }

        Ok(lhs)
    }

    pub fn parse_ident(&mut self) -> ParsingResult<Statement> {
        let start = self.get_next_location()?;
        let value = {
            let token = self.peek()?;
            match token.kind.clone() {
                TokenKind::Ident(s) => s,
                _ => return Err(ParsingError::UnexpectedToken(token.clone())),
            }
        };
        match value.as_str() {
            "if" => {
                let _ = self.eat()?;
                let cond = self.parse_expr(0)?;
                let then = self.parse_block()?;
                let or = if let Ok(token) = self.peek() {
                    if token.kind == TokenKind::Ident("else".to_string()) {
                        self.eat()?;
                        Some(self.parse_block()?)
                    } else {
                        None
                    }
                } else {
                    None
                };
                let kind = StatementKind::If {
                    cond: Box::new(cond),
                    then,
                    or,
                };
                Ok(Statement { kind, span: NodeSpan { start, end: self.get_location()? } })
            }
            "loop" => {
                let _ = self.eat()?;
                let body = self.parse_block()?;
                let kind = StatementKind::Loop { body };
                Ok(Statement { kind, span: NodeSpan { start, end: self.get_location()? } })
            }
            "break" => {
                let _ = self.eat()?;
                Ok(Statement {
                    kind: StatementKind::Break,
                    span: NodeSpan { start, end: self.get_location()? },
                })
            }
            "return" => {
                let _ = self.eat()?;
                let value = if let Ok(token) = self.peek() {
                    if token.kind
                        == TokenKind::Symbol(SymbolKind::Semicolon)
                    {
                        None
                    } else {
                        Some(self.parse_expr(0)?)
                    }
                } else {
                    None
                };

                // Thank you clippy lol           vvvvvvvv
                let kind = StatementKind::Return(value.map(Box::new));
                Ok(Statement { kind, span: NodeSpan { start, end: self.get_location()? } })
            }
            _ => {
                let next = self.peek_nth(1)?;
                match next.kind {
                    TokenKind::Symbol(SymbolKind::Walrus) => {
                        Ok(Statement {
                            kind: StatementKind::Decl(self.parse_local()?),
                            span: NodeSpan { start, end: self.get_location()? },
                        })
                    }
                    TokenKind::Symbol(SymbolKind::AssignEq) => {
                        Ok(Statement {
                            kind: StatementKind::Assign(self.parse_local()?),
                            span: NodeSpan { start, end: self.get_location()? },
                        })
                    }
                    TokenKind::Symbol(SymbolKind::LParen) => {
                        let call = self.parse_expr(0)?;
                        Ok(Statement {
                            kind: StatementKind::Expr(call),
                            span: NodeSpan { start, end: self.get_location()? },
                        })
                    }
                    _ => {
                        let expr = self.parse_expr(0)?;
                        Ok(Statement {
                            kind: StatementKind::Expr(expr),
                            span: NodeSpan { start, end: self.get_location()? },
                        })
                    }
                }
            }
        }
    }

    pub fn parse_type(&mut self) -> ParsingResult<Type> {
        let token = self.eat()?;
        match &token.kind {
            TokenKind::Ident(s) => match s.as_str() {
                "int" => Ok(Type::Int),
                "float" => Ok(Type::Float),
                "bool" => Ok(Type::Bool),
                _ => Err(ParsingError::UnexpectedToken(token)),
            },
            _ => Err(ParsingError::UnexpectedToken(token)),
        }
    }

    pub fn parse_item_kind(&mut self) -> ParsingResult<ItemKind> {
        let token_kind = self.peek()?.kind.clone();
        match token_kind {
            TokenKind::Ident(_) => {
                if let TokenKind::Symbol(SymbolKind::ColonColon) =
                    self.peek_nth(1)?.kind
                {
                    let function = self.parse_function()?;
                    Ok(
                        ItemKind::Function(function),
                    )
                } else {
                    Err(ParsingError::UnexpectedToken(self.peek()?.clone()))
                }
            }
            _ => Err(ParsingError::UnexpectedToken(self.peek()?.clone())),
        }
    }

    pub fn parse_item(&mut self) -> ParsingResult<Item> {
        let kind = self.parse_item_kind()?;
        Ok(Item { kind })
    }

    pub fn parse_local_name(&mut self) -> ParsingResult<Identifier> {
        let token = self.eat()?;
        match token.kind {
            TokenKind::Ident(s) => Ok(Identifier {
                value: s,
                span: NodeSpan {
                    start: token.location,
                    end: token.location,
                },
            }),
            _ => Err(ParsingError::UnexpectedToken(token)),
        }
    }

    pub fn parse_local(&mut self) -> ParsingResult<Assignment> {
        let name = self.parse_local_name()?;
        let _walrus = self.eat()?;
        let value = self.parse_expr(0)?;
        Ok(Assignment { name, value })
    }

    pub fn parse_function_arg(&mut self) -> ParsingResult<(String, Type)> {
        let name = self.eat()?;
        let name = match name.kind {
            TokenKind::Ident(s) => s,
            _ => return Err(ParsingError::UnexpectedToken(name)),
        };
        let colon = self.eat()?;
        if let TokenKind::Symbol(SymbolKind::Colon) = colon.kind {
        } else {
            return Err(ParsingError::UnexpectedToken(colon));
        }
        let ty = self.parse_type()?;
        Ok((name, ty))
    }

    pub fn parse_function_args(&mut self) -> ParsingResult<Vec<(String, Type)>> {
        let lparen = self.eat()?;
        if let TokenKind::Symbol(SymbolKind::LParen) = lparen.kind {
        } else {
            return Err(ParsingError::UnexpectedToken(lparen));
        }
        let mut args = Vec::new();
        loop {
            let token = self.peek()?;
            match token.kind {
                TokenKind::Symbol(SymbolKind::RParen) => {
                    let _ = self.eat()?;
                    break;
                }
                TokenKind::Symbol(SymbolKind::Comma) => {
                    let _ = self.eat()?;
                },
                _ => {
                    let arg = self.parse_function_arg()?;
                    args.push(arg);
                }
            }
        }
        Ok(args)
    }

    pub fn parse_fn_name(&mut self) -> ParsingResult<Identifier> {
        let token = self.eat()?;
        match token.kind {
            TokenKind::Ident(s) => Ok(Identifier {
                value: s,
                span: NodeSpan {
                    start: token.location,
                    end: token.location,
                },
            }),
            _ => Err(ParsingError::UnexpectedToken(token)),
        }
    }

    pub fn parse_fn_ret_ty(&mut self) -> ParsingResult<Type> {
        let arrow = self.peek()?;
        if arrow.kind == TokenKind::Symbol(SymbolKind::LBrace) {
            return Ok(Type::Void);
        }
        let _ = self.eat()?;
        let token = self.eat()?;
        match &token.kind {
            TokenKind::Ident(s) => match s.as_str() {
                "int" => Ok(Type::Int),
                "float" => Ok(Type::Float),
                "bool" => Ok(Type::Bool),
                "void" => Ok(Type::Void),
                _ => Err(ParsingError::UnexpectedToken(token)),
            },
            _ => Err(ParsingError::UnexpectedToken(token)),
        }
    }

    pub fn parse_fn_sig(&mut self) -> ParsingResult<FunctionSignature> {
        let name = self.parse_fn_name()?;

        let _coloncolon = self.eat()?;
        let args = self.parse_function_args()?;
        let ret_ty = self.parse_fn_ret_ty()?;

        Ok(FunctionSignature { name, args, ret_ty })
    }

    pub fn parse_function(&mut self) -> ParsingResult<Function> {
        let sig = self.parse_fn_sig()?;
        let body = self.parse_block()?;

        Ok(Function { sig, body })
    }

    pub fn parse_block(&mut self) -> ParsingResult<Block> {
        // eat '{'
        self.eat()?;
        let mut stmts = Vec::new();
        while let Ok(t) = self.peek() {
            match t.kind {
                TokenKind::Symbol(SymbolKind::RBrace) => break,
                TokenKind::Ident(_) => stmts.push(self.parse_ident()?),
                TokenKind::Symbol(SymbolKind::Semicolon) => {
                    let _ = self.eat()?;
                }
                _ => return Err(ParsingError::UnexpectedToken(t.clone())),
            }
        }

        self.eat()?;
        Ok(Block { stmts })
    }

    pub fn parse(mut self) -> ParsingResult<Ast> {
        let mut nodes = Vec::new();
        while let Some(item) = self.next() {
            nodes.push(item?);
        }
        Ok(Ast { nodes })
    }
}

impl<I: Iterator<Item = Token>> Iterator for Parser<I> {
    type Item = ParsingResult<Item>;

    fn next(&mut self) -> Option<Self::Item> {
        match self.peek().map(|tok| tok.kind.clone()) {
            Err(_) => None,
            Ok(_) => Some(self.parse_item()),
        }
    }
}
