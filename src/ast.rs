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
pub struct ModuleDecl {
    pub name: String,
    pub body: Option<Program>,
    pub is_pub: bool,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Program {
    pub uses: Vec<Use>,
    pub modules: Vec<ModuleDecl>,
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
    pub is_pub: bool,
    pub module_path: Vec<String>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Function {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub type_params: Vec<GenericParam>,
    pub body: Block,
    pub is_extern: bool,
    pub is_intrinsic: bool,
    pub is_method: bool,
    pub is_pub: bool,
    pub attribs: Vec<Attribute>,
    pub span: Span,
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
        pattern: Pattern,
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
    Continue {
        #[allow(dead_code)]
        span: Span,
    },
    Break {
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
    GenericArray { inner: Box<Type>, len_var: String },
    Slice { inner: Box<Type> },
    Struct(String),
    GenericInstance(String, Vec<Type>),
    Alias(String, Vec<Type>),
    SelfType,
    ImplTrait(Vec<TraitBound>),
}

#[derive(Debug, Clone, PartialEq)]
pub struct TraitBound {
    pub trait_name: String,
    pub generic_args: Vec<Type>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    pub name: String,
    pub bounds: Vec<TraitBound>,
}

#[derive(Debug, Clone)]
pub enum Expr {
    BoolLit(bool, Span),
    IntLit(i64, Span),
    FloatLit(f64, Span),
    StrLit(String, Span),
    Ident(String, Span),
    Call {
        callee: String,
        args: Vec<Expr>,
        span: Span,
    },
    QualifiedCall {
        module: String,
        callee: String,
        args: Vec<Expr>,
        span: Span,
    },
    Binary {
        op: BinOp,
        lhs: Box<Expr>,
        rhs: Box<Expr>,
        span: Span,
    },
    Assign {
        target: Box<Expr>,
        value: Box<Expr>,
        span: Span,
    },
    Ref {
        expr: Box<Expr>,
        is_mut: bool,
        span: Span,
    },
    UnaryNot(Box<Expr>, Span),
    UnaryMinus(Box<Expr>, Span),
    Deref(Box<Expr>, Span),
    Cast {
        expr: Box<Expr>,
        to_type: Type,
        span: Span,
    },
    Tuple(Vec<Expr>, Span),
    Unit(Span),
    Member {
        expr: Box<Expr>,
        index: usize,
        field: Option<String>,
        span: Span,
    },
    MethodCall {
        expr: Box<Expr>,
        method: String,
        args: Vec<Expr>,
        span: Span,
    },
    StructLit {
        struct_name: String,
        fields: Vec<(String, Expr)>,
        span: Span,
    },
    EnumLit {
        enum_name: String,
        variant: String,
        payload: Option<Box<Expr>>,
        span: Span,
    },
    If {
        cond: Box<Expr>,
        then_block: Block,
        else_ifs: Vec<(Expr, Block)>,
        else_block: Option<Block>,
        span: Span,
    },
    Loop {
        body: Block,
        span: Span,
    },
    While {
        cond: Box<Expr>,
        body: Block,
        span: Span,
    },
    For {
        pattern: Pattern,
        container: Box<Expr>,
        body: Block,
        span: Span,
    },
    Array(Vec<Expr>, Span),
    Repeat(Box<Expr>, usize, Span),
    Index {
        array: Box<Expr>,
        index: Box<Expr>,
        span: Span,
    },
    IfLet {
        pattern: Pattern,
        scrutinee: Box<Expr>,
        then_block: Block,
        else_block: Option<Block>,
        span: Span,
    },
    Match {
        scrutinee: Box<Expr>,
        arms: Vec<MatchArm>,
        span: Span,
    },
    Block(Block, Span),
}

impl Expr {
    /// Returns the span of this expression.
    pub fn span(&self) -> Span {
        match self {
            Expr::BoolLit(_, s)
            | Expr::IntLit(_, s)
            | Expr::FloatLit(_, s)
            | Expr::StrLit(_, s)
            | Expr::Ident(_, s)
            | Expr::UnaryNot(_, s)
            | Expr::UnaryMinus(_, s)
            | Expr::Deref(_, s)
            | Expr::Tuple(_, s)
            | Expr::Unit(s)
            | Expr::Array(_, s)
            | Expr::Repeat(_, _, s)
            | Expr::Block(_, s) => *s,
            Expr::Call { span: s, .. }
            | Expr::QualifiedCall { span: s, .. }
            | Expr::Binary { span: s, .. }
            | Expr::Assign { span: s, .. }
            | Expr::Ref { span: s, .. }
            | Expr::Cast { span: s, .. }
            | Expr::Member { span: s, .. }
            | Expr::MethodCall { span: s, .. }
            | Expr::StructLit { span: s, .. }
            | Expr::EnumLit { span: s, .. }
            | Expr::If { span: s, .. }
            | Expr::Loop { span: s, .. }
            | Expr::While { span: s, .. }
            | Expr::For { span: s, .. }
            | Expr::Index { span: s, .. }
            | Expr::IfLet { span: s, .. }
            | Expr::Match { span: s, .. } => *s,
        }
    }
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
    pub type_params: Vec<GenericParam>,
    pub is_pub: bool,
    pub attribs: Vec<Attribute>,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StructField {
    pub name: String,
    pub ty: Type,
    pub is_pub: bool,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TraitDecl {
    pub name: String,
    pub type_params: Vec<GenericParam>,
    pub methods: Vec<TraitMethodDef>,
    pub is_pub: bool,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TraitMethodDef {
    pub name: String,
    pub params: Vec<Param>,
    pub return_type: Option<Type>,
    pub body: Option<Block>,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TypeAliasDecl {
    pub name: String,
    pub type_params: Vec<GenericParam>,
    pub aliased_type: Type,
    pub is_pub: bool,
    pub span: Span,
}

#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EnumDecl {
    pub name: String,
    pub variants: Vec<EnumVariant>,
    pub type_params: Vec<GenericParam>,
    pub is_pub: bool,
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
    pub trait_args: Vec<Type>,
    pub type_params: Vec<GenericParam>,
    pub const_params: Vec<String>,
    pub methods: Vec<Function>,
}

/// A pattern for destructuring in `if let` and `match`.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
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
