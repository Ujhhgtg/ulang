use crate::token::Span;

#[derive(Debug)]
pub struct Program {
    pub funcs: Vec<Function>,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Function {
    pub name: String,
    pub body: Block,
}

#[allow(dead_code)]
#[derive(Debug)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug)]
pub enum Stmt {
    Let {
        name: String,
        init: Expr,
        #[allow(dead_code)]
        span: Span,
    },
    Expr(Expr),
}

#[derive(Debug)]
pub enum Expr {
    IntLit(i64),
    Ident(String),
    Call {
        callee: String,
        arg: Box<Expr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
}

#[derive(Debug, Clone, Copy)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
}
