//! Abstract Syntax Tree (AST) types for the ulang compiler.
//!
//! These types represent parsed ulang programs after lexical analysis and parsing.
//! Every AST node carries a [`Span`] that records its byte-offset range in the
//! original source text, enabling source-anchored error reporting. The top-level
//! entry point is [`Program`], which holds all items declared at the crate root
//! or within a module.

use crate::token::Span;

/// An attribute applied to a declaration (e.g. `#[inline]`).
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Attribute {
    /// The attribute name (e.g. `"inline"`, `"repr"`).
    pub name: String,
    /// Positional arguments to the attribute.
    pub args: Vec<String>,
    /// Source span of the attribute.
    pub span: Span,
}

/// A module declaration: `mod name { ... }` or `mod name;`.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ModuleDecl {
    /// The module name.
    pub name: String,
    /// The module body, or `None` for file-based modules.
    pub body: Option<Program>,
    /// Whether the module is declared as `pub`.
    pub is_pub: bool,
    /// Source span of the entire module declaration.
    pub span: Span,
}

/// The top-level representation of a parsed ulang program or module body.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Program {
    /// Use/import declarations.
    pub uses: Vec<Use>,
    /// Nested module declarations.
    pub modules: Vec<ModuleDecl>,
    /// Function declarations.
    pub funcs: Vec<Function>,
    /// Struct type declarations.
    pub structs: Vec<StructDecl>,
    /// Enum type declarations.
    pub enums: Vec<EnumDecl>,
    /// Trait declarations.
    pub traits: Vec<TraitDecl>,
    /// Impl blocks.
    pub impls: Vec<ImplDecl>,
    /// Type alias declarations.
    pub type_aliases: Vec<TypeAliasDecl>,
}

/// A `use` import declaration.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Use {
    /// The imported path segments (e.g. `["std", "fmt"]` for `use std::fmt`).
    pub path: Vec<String>,
    /// Whether the import is re-exported as `pub use`.
    pub is_pub: bool,
    /// The module path in which this import resides.
    pub module_path: Vec<String>,
    /// Source span of the use declaration.
    pub span: Span,
}

/// A function declaration.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Function {
    /// The function name.
    pub name: String,
    /// The function parameters.
    pub params: Vec<Param>,
    /// The return type, or `None` if the unit type `()` is implied.
    pub return_type: Option<Type>,
    /// Generic type parameters.
    pub type_params: Vec<GenericParam>,
    /// The function body block.
    pub body: Block,
    /// Whether this is an `extern` function (no body in Rust source).
    pub is_extern: bool,
    /// Whether this is an intrinsic compiler built-in.
    pub is_intrinsic: bool,
    /// Whether this is a method (has a `self`-like parameter).
    pub is_method: bool,
    /// Whether the function is declared as `pub`.
    pub is_pub: bool,
    /// Attributes applied to this function.
    pub attribs: Vec<Attribute>,
    /// Source span of the entire function declaration.
    pub span: Span,
}

/// A single function parameter.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Param {
    /// The parameter name.
    pub name: String,
    /// The parameter type annotation.
    pub ty: Type,
    /// Source span of the parameter.
    pub span: Span,
}

/// A block of statements with an optional tail expression.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct Block {
    /// The statements in the block.
    pub stmts: Vec<Stmt>,
    /// An optional final expression whose value is yielded by the block (no trailing `;`).
    pub tail_expr: Option<Box<Expr>>,
    /// Source span of the block, including the braces.
    pub span: Span,
}

/// A statement appearing inside a block.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub enum Stmt {
    /// A `let` binding: `let [mut] pattern [: type] = expr;`.
    Let {
        /// The destructuring pattern for the binding.
        pattern: Pattern,
        /// Whether the binding is declared `mut`.
        is_mut: bool,
        /// Optional type annotation.
        type_ann: Option<Type>,
        /// The initializer expression.
        init: Expr,
        /// The optional else block for let-else statement.
        else_block: Option<Block>,
        #[allow(dead_code)]
        /// Source span of the let statement.
        span: Span,
    },
    /// A `const` declaration: `const name [: type] = expr;`.
    Const {
        /// The constant name.
        name: String,
        /// Optional type annotation.
        type_ann: Option<Type>,
        /// The constant value expression.
        init: Expr,
        #[allow(dead_code)]
        /// Source span of the const declaration.
        span: Span,
    },
    /// An expression statement: `expr;` — the value is discarded.
    Expr(Expr),
    /// A `return` statement: `return [expr];`.
    Return {
        /// The optional return value expression.
        value: Option<Box<Expr>>,
        #[allow(dead_code)]
        /// Source span of the return statement.
        span: Span,
    },
    /// A `continue` statement: `continue;` — skip to next loop iteration.
    Continue {
        #[allow(dead_code)]
        /// Source span of the continue statement.
        span: Span,
    },
    /// A `break` statement: `break [expr];` — exit a loop, optionally yielding a value.
    Break {
        /// The optional value yielded by the break.
        value: Option<Box<Expr>>,
        #[allow(dead_code)]
        /// Source span of the break statement.
        span: Span,
    },
}

/// A type annotation or type expression.
#[allow(clippy::enum_variant_names)]
#[derive(Debug, Clone, PartialEq)]
pub enum Type {
    /// The `i8` type — 8-bit signed integer.
    I8,
    /// The `i16` type — 16-bit signed integer.
    I16,
    /// The `i32` type — 32-bit signed integer.
    I32,
    /// The `i64` type — 64-bit signed integer.
    I64,
    /// The `u8` type — 8-bit unsigned integer.
    U8,
    /// The `u16` type — 16-bit unsigned integer.
    U16,
    /// The `u32` type — 32-bit unsigned integer.
    U32,
    /// The `u64` type — 64-bit unsigned integer.
    U64,
    /// The `usize` type — pointer-width unsigned integer.
    Usize,
    /// The `isize` type — pointer-width signed integer.
    Isize,
    /// The `f32` type — 32-bit floating point.
    F32,
    /// The `f64` type — 64-bit floating point.
    F64,
    /// The `bool` type — boolean.
    Bool,
    /// The `!` (never) type, valid in return position.
    Never,
    /// A tuple type: `(T1, T2, ...)`.
    Tuple(Vec<Type>),
    /// The unit type `()`.
    Unit,
    /// The `str` type — string slice.
    Str,
    /// A raw pointer type: `*const T` or `*mut T`.
    Ptr {
        /// The pointed-to type.
        inner: Box<Type>,
        /// Whether the pointer is mutable (`*mut T` vs `*const T`).
        is_mut: bool,
    },
    /// A reference type: `&T` or `&mut T`.
    Ref {
        /// The referenced type.
        inner: Box<Type>,
        /// Whether the reference is mutable.
        is_mut: bool,
    },
    /// A fixed-size array type: `[T; N]`.
    Array {
        /// The element type.
        inner: Box<Type>,
        /// The array length.
        len: usize,
    },
    /// A generic-parameter-sized array type: `[T; N]` where `N` is a const generic.
    GenericArray {
        /// The element type.
        inner: Box<Type>,
        /// The const generic name for the length.
        len_var: String,
    },
    /// A slice type: `[T]`.
    Slice {
        /// The element type.
        inner: Box<Type>,
    },
    /// A named struct type: `StructName`.
    Struct(String),
    /// A generic type instantiation: `TypeName<Args>`.
    GenericInstance(String, Vec<Type>),
    /// A type alias reference with generic args: `AliasName<Args>`.
    Alias(String, Vec<Type>),
    /// The `Self` type used inside an impl block.
    SelfType,
    /// An inferred type (written as `_`).
    Infer,
    /// An `impl Trait` opaque type.
    ImplTrait(Vec<TraitBound>),
}

/// A trait bound in a generic parameter or impl trait position.
#[derive(Debug, Clone, PartialEq)]
pub struct TraitBound {
    /// The trait name.
    pub trait_name: String,
    /// Generic arguments for the trait bound.
    pub generic_args: Vec<Type>,
}

/// A generic type parameter with optional trait bounds.
#[derive(Debug, Clone, PartialEq)]
pub struct GenericParam {
    /// The parameter name.
    pub name: String,
    /// Trait bounds constraining this parameter.
    pub bounds: Vec<TraitBound>,
}

/// An expression — the core evaluable unit of ulang programs.
#[derive(Debug, Clone)]
pub enum Expr {
    /// A boolean literal: `true` or `false`.
    BoolLit(bool, Span),
    /// An integer literal: `42`, `0xFF`, etc.
    IntLit(i64, Span),
    /// A floating-point literal: `3.14`, etc.
    FloatLit(f64, Span),
    /// A string literal: `"hello"`.
    StrLit(String, Span),
    /// An identifier reference: variable name, function name, etc.
    Ident(String, Span),
    /// A direct function call: `callee(args)`.
    Call {
        /// The callee name.
        callee: String,
        /// Arguments to the call.
        args: Vec<Expr>,
        /// Explicit type arguments (turbofish).
        type_args: Vec<Type>,
        /// Source span.
        span: Span,
    },
    /// A qualified function call: `module::callee(args)`.
    QualifiedCall {
        /// The module path segment.
        module: String,
        /// The callee name.
        callee: String,
        /// Arguments to the call.
        args: Vec<Expr>,
        /// Explicit type arguments (turbofish).
        type_args: Vec<Type>,
        /// Source span.
        span: Span,
    },
    /// A binary operation: `lhs op rhs`.
    Binary {
        /// The binary operator.
        op: BinOp,
        /// Left-hand side expression.
        lhs: Box<Expr>,
        /// Right-hand side expression.
        rhs: Box<Expr>,
        /// Source span.
        span: Span,
    },
    /// An assignment expression: `target = value`.
    Assign {
        /// The assignee (lvalue).
        target: Box<Expr>,
        /// The assigned value.
        value: Box<Expr>,
        /// Source span.
        span: Span,
    },
    /// A reference expression: `&expr` or `&mut expr`.
    Ref {
        /// The referenced expression.
        expr: Box<Expr>,
        /// Whether the reference is mutable.
        is_mut: bool,
        /// Source span.
        span: Span,
    },
    /// A logical not expression: `!expr`.
    UnaryNot(Box<Expr>, Span),
    /// A unary minus expression: `-expr`.
    UnaryMinus(Box<Expr>, Span),
    /// A dereference expression: `*expr`.
    Deref(Box<Expr>, Span),
    /// A type cast expression: `expr as Type`.
    Cast {
        /// The expression to cast.
        expr: Box<Expr>,
        /// The target type.
        to_type: Type,
        /// Source span.
        span: Span,
    },
    /// A tuple expression: `(a, b, ...)`.
    Tuple(Vec<Expr>, Span),
    /// The unit literal: `()`.
    Unit(Span),
    /// A member access expression: `expr.field` or `expr.0` (tuple indexing).
    Member {
        /// The container expression.
        expr: Box<Expr>,
        /// The field index (for tuple structs).
        index: usize,
        /// The field name (for named structs).
        field: Option<String>,
        /// Source span.
        span: Span,
    },
    /// A method call expression: `expr.method(args)`.
    MethodCall {
        /// The receiver expression.
        expr: Box<Expr>,
        /// The method name.
        method: String,
        /// Arguments to the method (excluding the receiver).
        args: Vec<Expr>,
        /// Explicit type arguments (turbofish).
        type_args: Vec<Type>,
        /// Source span.
        span: Span,
    },
    /// A struct literal expression: `StructName { field: value, ... }`.
    StructLit {
        /// The struct type name.
        struct_name: String,
        /// The field initializers.
        fields: Vec<(String, Expr)>,
        /// Source span.
        span: Span,
    },
    /// An enum literal expression: `EnumName::Variant` or `EnumName::Variant(payload)`.
    EnumLit {
        /// The enum type name.
        enum_name: String,
        /// The variant name.
        variant: String,
        /// Optional payload expression for variants with data.
        payload: Option<Box<Expr>>,
        /// Source span.
        span: Span,
    },
    /// An `if` expression: `if cond { ... } else if cond { ... } else { ... }`.
    If {
        /// The condition expression.
        cond: Box<Expr>,
        /// The then-block.
        then_block: Block,
        /// Chained `else if` branches.
        else_ifs: Vec<(Expr, Block)>,
        /// The optional final else block.
        else_block: Option<Block>,
        /// Source span.
        span: Span,
    },
    /// A `loop` expression: `loop { ... }`.
    Loop {
        /// The loop body.
        body: Block,
        /// Source span.
        span: Span,
    },
    /// A `while` expression: `while cond { ... }`.
    While {
        /// The loop condition.
        cond: Box<Expr>,
        /// The loop body.
        body: Block,
        /// Source span.
        span: Span,
    },
    /// A `for` expression: `for pattern in container { ... }`.
    For {
        /// The iteration pattern (destructuring).
        pattern: Pattern,
        /// The container expression being iterated.
        container: Box<Expr>,
        /// The loop body.
        body: Block,
        /// Source span.
        span: Span,
    },
    /// An array literal: `[a, b, c, ...]`.
    Array(Vec<Expr>, Span),
    /// A repeated array literal: `[expr; count]`.
    Repeat(Box<Expr>, usize, Span),
    /// An index expression: `array[index]`.
    Index {
        /// The array or slice expression.
        array: Box<Expr>,
        /// The index expression.
        index: Box<Expr>,
        /// Source span.
        span: Span,
    },
    /// An `if let` expression: `if let pattern = scrutinee { ... } else { ... }`.
    IfLet {
        /// The destructuring pattern.
        pattern: Pattern,
        /// The scrutinee expression.
        scrutinee: Box<Expr>,
        /// The then-block.
        then_block: Block,
        /// The optional else block.
        else_block: Option<Block>,
        /// Source span.
        span: Span,
    },
    /// A `match` expression: `match scrutinee { arms }`.
    Match {
        /// The scrutinee expression.
        scrutinee: Box<Expr>,
        /// The match arms.
        arms: Vec<MatchArm>,
        /// Source span.
        span: Span,
    },
    /// A block expression: `{ stmts; tail_expr }`.
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

/// A binary operator between two expressions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BinOp {
    /// The `+` operator — addition.
    Add,
    /// The `-` operator — subtraction.
    Sub,
    /// The `*` operator — multiplication.
    Mul,
    /// The `/` operator — division.
    Div,
    /// The `==` operator — equality comparison.
    Eq,
    /// The `!=` operator — inequality comparison.
    Neq,
    /// The `<` operator — less-than comparison.
    Lt,
    /// The `>` operator — greater-than comparison.
    Gt,
    /// The `<=` operator — less-than-or-equal comparison.
    Le,
    /// The `>=` operator — greater-than-or-equal comparison.
    Ge,
    /// The `&&` operator — logical AND.
    And,
    /// The `||` operator — logical OR.
    Or,
}

/// A struct type declaration.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StructDecl {
    /// The struct name.
    pub name: String,
    /// The struct's fields.
    pub fields: Vec<StructField>,
    /// Generic type parameters.
    pub type_params: Vec<GenericParam>,
    /// Whether the struct is declared as `pub`.
    pub is_pub: bool,
    /// Attributes applied to this struct.
    pub attribs: Vec<Attribute>,
    /// Source span of the struct declaration.
    pub span: Span,
}

/// A single field in a struct declaration.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct StructField {
    /// The field name.
    pub name: String,
    /// The field type.
    pub ty: Type,
    /// Whether the field is declared as `pub`.
    pub is_pub: bool,
    /// Source span of the field.
    pub span: Span,
}

/// A constant declaration inside a trait definition.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TraitConst {
    /// The constant name.
    pub name: String,
    /// The constant type.
    pub ty: Type,
    /// An optional default value for the constant.
    pub default_value: Option<Expr>,
    /// Source span of the trait constant declaration.
    pub span: Span,
}

/// A trait declaration.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TraitDecl {
    /// The trait name.
    pub name: String,
    /// Generic type parameters.
    pub type_params: Vec<GenericParam>,
    /// Methods defined in the trait.
    pub methods: Vec<TraitMethodDef>,
    /// Constants defined in the trait.
    pub consts: Vec<TraitConst>,
    /// Whether the trait is declared as `pub`.
    pub is_pub: bool,
    /// Source span of the trait declaration.
    pub span: Span,
}

/// A method signature or definition inside a trait.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TraitMethodDef {
    /// The method name.
    pub name: String,
    /// The method parameters.
    pub params: Vec<Param>,
    /// The return type, or `None` if unit is implied.
    pub return_type: Option<Type>,
    /// The optional default implementation body.
    pub body: Option<Block>,
}

/// A type alias declaration: `type Name<Params> = AliasedType;`.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct TypeAliasDecl {
    /// The alias name.
    pub name: String,
    /// Generic type parameters.
    pub type_params: Vec<GenericParam>,
    /// The type being aliased.
    pub aliased_type: Type,
    /// Whether the alias is declared as `pub`.
    pub is_pub: bool,
    /// Source span of the type alias declaration.
    pub span: Span,
}

/// An enum type declaration.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EnumDecl {
    /// The enum name.
    pub name: String,
    /// The enum variants.
    pub variants: Vec<EnumVariant>,
    /// Generic type parameters.
    pub type_params: Vec<GenericParam>,
    /// Whether the enum is declared as `pub`.
    pub is_pub: bool,
    /// Attributes applied to this enum.
    pub attribs: Vec<Attribute>,
    /// Source span of the enum declaration.
    pub span: Span,
}

/// A single variant in an enum declaration.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct EnumVariant {
    /// The variant name.
    pub name: String,
    /// The optional payload type (for variants with data).
    pub ty: Option<Type>,
    /// Source span of the variant.
    pub span: Span,
}

/// A constant value declared inside an impl block as an associated item.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct AssociatedConst {
    /// The constant name.
    pub name: String,
    /// The constant type.
    pub ty: Type,
    /// The constant value expression.
    pub value: Expr,
    /// Whether the constant is declared as `pub`.
    pub is_pub: bool,
    /// Source span of the associated constant.
    pub span: Span,
}

/// An `impl` block: `impl [Trait for] Type { ... }`.
#[allow(dead_code)]
#[derive(Debug, Clone)]
pub struct ImplDecl {
    /// The type being implemented.
    pub impl_type: Type,
    /// Optional trait name for trait impl blocks.
    pub trait_name: Option<String>,
    /// Generic arguments for the trait being implemented.
    pub trait_args: Vec<Type>,
    /// Generic type parameters for the impl block.
    pub type_params: Vec<GenericParam>,
    /// Const generic parameter names for the impl block.
    pub const_params: Vec<String>,
    /// Methods defined in the impl block.
    pub methods: Vec<Function>,
    /// Associated constants defined in the impl block.
    pub consts: Vec<AssociatedConst>,
    /// Source span of the impl block.
    pub span: Span,
}

/// A pattern for destructuring in `if let` and `match`.
#[allow(dead_code)]
#[derive(Debug, Clone, PartialEq)]
pub enum Pattern {
    /// `_` — matches anything, binds nothing.
    Wildcard,
    /// `x` — matches anything, binds value to `x`.
    Binding(String),
    /// `VariantName` or `VariantName(inner)` — matches an enum variant.
    EnumVariant {
        /// Optional enum name qualifier: `Option` in `Option::Some(x)`, `None` when inferred.
        enum_name: Option<String>,
        /// The variant name.
        variant: String,
        /// Payload pattern; `None` for unit variants.
        payload: Option<Box<Pattern>>,
    },
    /// Integer literal pattern: `42`.
    IntLit(i64),
    /// Boolean literal pattern: `true`, `false`.
    BoolLit(bool),
}

/// An arm of a `match` expression.
#[derive(Debug, Clone)]
pub struct MatchArm {
    /// The pattern to match against.
    pub pattern: Pattern,
    /// Optional `if` guard condition.
    pub guard: Option<Box<Expr>>,
    /// The arm body block.
    pub body: Block,
}
