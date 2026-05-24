use crate::token::Span;

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Attribute {
    pub name: String,
    pub args: Vec<String>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Program {
    pub uses: Vec<Use>,
    pub funcs: Vec<Function>,
    pub structs: Vec<StructDecl>,
    pub traits: Vec<TraitDecl>,
    pub impls: Vec<ImplDecl>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Use {
    pub path: Vec<String>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub body: Block,
    pub is_extern: bool,
    pub is_method: bool,
    pub attribs: Vec<Attribute>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Param {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Block {
    pub stmts: Vec<Stmt>,
    pub tail_expr: Option<Box<Expr>>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Stmt {
    Let {
        name: String,
        is_mut: bool,
        type_ann: Option<Type>,
        init: Expr,
        #[allow(dead_code)]
        span: Span,
    },
    Const {
        name: String,
        type_ann: Option<Type>,
        init: Expr,
        #[allow(dead_code)]
        span: Span,
    },
    Expr(Expr),
    Return {
        value: Option<Box<Expr>>,
        #[allow(dead_code)]
        span: Span,
    },
}

#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    I8,
    I16,
    I32,
    I64,
    U8,
    U16,
    U32,
    U64,
    Usize,
    Isize,
    F32,
    F64,
    Bool,
    Never, // ! (never type, valid in return position)
    Tuple(Vec<Type>),
    Unit,
    Str,
    Ptr { inner: Box<Type>, is_mut: bool },
    Ref { inner: Box<Type>, is_mut: bool },
    Struct(String),
    SelfType,
}

#[derive(Debug, Clone)]
pub enum Expr {
    BoolLit(bool),
    IntLit(i64),
    FloatLit(f64),
    StrLit(String),
    Ident(String),
    Call {
        callee: String,
        args: Vec<Expr>,
    },
    QualifiedCall {
        module: String,
        callee: String,
        args: Vec<Expr>,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
    },
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
    },
    Ref {
        expr: Box<Expr>,
        is_mut: bool,
    },
    Deref(Box<Expr>),
    Cast {
        expr: Box<Expr>,
        to_type: Type,
    },
    Tuple(Vec<Expr>),
    Unit,
    Member {
        expr: Box<Expr>,
        index: usize,
        field: Option<String>,
    },
    MethodCall {
        expr: Box<Expr>,
        method: String,
        args: Vec<Expr>,
    },
    StructLit {
        struct_name: String,
        fields: Vec<(String, Expr)>,
    },
    If {
        cond: Box<Expr>,
        then_block: Block,
        else_ifs: Vec<(Expr, Block)>,
        else_block: Option<Block>,
    },
    Loop {
        body: Block,
    },
    While {
        cond: Box<Expr>,
        body: Block,
    },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
    Div,
    Eq,
    Neq,
    Lt,
    Gt,
    Le,
    Ge,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<StructField>,
    pub attribs: Vec<Attribute>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TraitDecl {
    pub name: String,
    pub methods: Vec<TraitMethodDef>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TraitMethodDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ImplDecl {
    pub impl_type: Type,
    pub trait_name: Option<String>,
    pub methods: Vec<Function>,
}
