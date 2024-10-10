use std::iter::Peekable;

use crate::tokenizing;

pub enum BinOpKind {
    Add,
    Sub,
    Mul,
    Div,
    Mod,
    Assign,
    Declare,
}

pub type NodeLocation = usize;

pub enum ExprKind {
    Literal(f64),
    Var(String),
    BinOp {
        kind: BinOpKind,
        lhs: NodeLocation,
        rhs: NodeLocation,
    },
}

pub struct Expr {
    kind: ExprKind,
    pos: usize,
}



pub struct Parser<I: Iterator<Item = tokenizing::Token>> {
    tokens: Peekable<I>,
    pos: usize,
}

impl<I: Iterator<Item = tokenizing::Token>> Parser<I> {
    pub fn new(tokens: I) -> Self {
        Self {
            tokens: tokens.peekable(),
            pos: 0,
        }
    }

    pub fn parse_expr

    pub fn parse(&mut self) {
        while let Some(token) = self.tokens.next() {
            println!("{:?}", token);
        }
    }
}
