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
    pub enums: Vec<EnumDecl>,
    pub traits: Vec<TraitDecl>,
    pub impls: Vec<ImplDecl>,
    pub type_aliases: Vec<TypeAliasDecl>,
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
    Array { inner: Box<Type>, len: usize },
    Struct(String),
    GenericInstance(String, Vec<Type>),
    Alias(String, Vec<Type>),
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
    UnaryNot(Box<Expr>),
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
    EnumLit {
        enum_name: String,
        variant: String,
        payload: Option<Box<Expr>>,
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
    Array(Vec<Expr>),
    Repeat(Box<Expr>, usize),
    Index {
        array: Box<Expr>,
        index: Box<Expr>,
    },
    IfLet {
        pattern: Pattern,
        scrutinee: Box<Expr>,
        then_block: Block,
        else_block: Option<Block>,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
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
    And,
    Or,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StructDecl {
    pub name: String,
    pub fields: Vec<StructField>,
    pub type_params: Vec<String>,
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
    pub type_params: Vec<String>,
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
pub struct TypeAliasDecl {
    pub name: String,
    pub type_params: Vec<String>,
    pub aliased_type: Type,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<EnumVariant>,
    pub type_params: Vec<String>,
    pub attribs: Vec<Attribute>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EnumVariant {
    pub name: String,
    pub ty: Option<Type>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ImplDecl {
    pub impl_type: Type,
    pub trait_name: Option<String>,
    pub type_params: Vec<String>,
    pub methods: Vec<Function>,
}

/// A pattern for destructuring in `if let` and `match`.
#[derive(Debug, Clone)]
pub enum Pattern {
    /// `_` — matches anything, binds nothing
    Wildcard,
    /// `x` — matches anything, binds value to `x`
    Binding(String),
    /// `VariantName` or `VariantName(inner)` — matches an enum variant
    EnumVariant {
        /// Optional enum name qualifier: `Option` in `Option::Some(x)`, `None` when inferred
        enum_name: Option<String>,
        variant: String,
        /// Payload pattern; `None` for unit variants
        payload: Option<Box<Pattern>>,
    },
    /// Integer literal pattern: `42`
    IntLit(i64),
    /// Boolean literal pattern: `true`, `false`
    BoolLit(bool),
}

/// An arm of a `match` expression.
#[derive(Debug, Clone)]
pub struct MatchArm {
    pub pattern: Pattern,
    /// Optional `if` guard condition
    pub guard: Option<Box<Expr>>,
    pub body: Block,
}
