use std::iter::Peekable;

use crate::tokenizing;

pub enum AstNodeKind {
    Expr(Expr),
}

pub struct AstNode {
    kind: AstNodeKind,
    location: NodeLocation,
}

pub type Ast = Vec<AstNode>;

pub trait IntoNode {
    fn into_node(self, loc: NodeLocation) -> AstNode;
}

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
}

impl IntoNode for Expr {
    fn into_node(self, loc: NodeLocation) -> AstNode {
        AstNode {
            kind: AstNodeKind::Expr(self),
            location: loc,
        }
    }
}



pub struct Parser<I: Iterator<Item = tokenizing::Token>> {
    tokens: Peekable<I>,
    current_node_idx: NodeLocation,
}

impl<I: Iterator<Item = tokenizing::Token>> Parser<I> {
    pub fn new(tokens: I) -> Self {
        Self {
            tokens: tokens.peekable(),
            current_node_idx: 0,
        }
    }

    pub fn new_node(&mut self, node_kind: impl IntoNode) -> AstNode {
        let node = node_kind.into_node(self.current_node_idx);
        self.current_node_idx += 1;
        node
    }

    pub fn parse_expr(&mut self, precedence: u8) -> Expr

    pub fn parse(&mut self) {
        while let Some(token) = self.tokens.next() {
            println!("{:?}", token);
        }
    }
}
