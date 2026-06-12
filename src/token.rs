//! Token types and span tracking for the ulang compiler.
//!
//! This module defines [`Span`], a byte-offset range used throughout the compiler
//! for error reporting, and [`Token`], the complete set of lexical tokens produced
//! by the lexer and consumed by the parser. Every variant of [`Token`] corresponds
//! to a single syntactic element in a ulang source program.

/// A byte-offset range in source code, used for error reporting.
///
/// `Span` records the start (`lo`) and end (`hi`) byte positions of a syntactic element
/// within the source text. It is a zero-copy, `Copy`-able type that all AST nodes carry
/// so that diagnostics can point to the relevant source locations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    /// Inclusive starting byte offset of the span.
    pub lo: usize,
    /// Exclusive ending byte offset of the span (one past the last byte).
    pub hi: usize,
}

impl Span {
    /// Creates a new span from the given byte-offset bounds.
    ///
    /// `lo` is the inclusive starting byte offset and `hi` is the exclusive ending
    /// byte offset. Both are expected to be valid byte positions within the source text.
    pub fn new(lo: usize, hi: usize) -> Self {
        Self { lo, hi }
    }

    /// Creates a zero-width span at the given byte position.
    ///
    /// Useful for synthetic elements or error recovery where no real source range exists.
    pub fn empty(pos: usize) -> Self {
        Self { lo: pos, hi: pos }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_new() {
        let span = Span::new(3, 7);
        assert_eq!(span.lo, 3);
        assert_eq!(span.hi, 7);
    }

    #[test]
    fn test_span_empty() {
        let span = Span::empty(5);
        assert_eq!(span.lo, 5);
        assert_eq!(span.hi, 5);
    }

    #[test]
    fn test_token_equality() {
        assert_eq!(Token::Fn, Token::Fn);
        assert_eq!(Token::Let, Token::Let);
        assert_eq!(Token::Continue, Token::Continue);
        assert_eq!(Token::Break, Token::Break);
        assert_eq!(Token::Mod, Token::Mod);
        assert_eq!(Token::Mut, Token::Mut);
        assert_eq!(Token::Const, Token::Const);
        assert_eq!(Token::As, Token::As);
        assert_eq!(Token::Use, Token::Use);
        assert_eq!(Token::Extern, Token::Extern);
        assert_eq!(Token::StrLit("hello".into()), Token::StrLit("hello".into()));
        assert_ne!(Token::StrLit("hello".into()), Token::StrLit("world".into()));
        assert_eq!(Token::Str, Token::Str);
        assert_eq!(Token::Bool, Token::Bool);
        assert_eq!(Token::True, Token::True);
        assert_eq!(Token::False, Token::False);
        assert_eq!(Token::I32, Token::I32);
        assert_eq!(Token::F64, Token::F64);
        assert_eq!(Token::IntLit(42), Token::IntLit(42));
        assert_eq!(
            Token::Ident("foo".to_string()),
            Token::Ident("foo".to_string())
        );
        assert_ne!(
            Token::Ident("foo".to_string()),
            Token::Ident("bar".to_string())
        );
        assert_ne!(Token::IntLit(1), Token::IntLit(2));
        assert_eq!(Token::FloatLit(3.14), Token::FloatLit(3.14));
        assert_ne!(Token::FloatLit(1.0), Token::FloatLit(2.0));
        assert_eq!(
            Token::IntSuffixLit(42, Box::new(Token::I32)),
            Token::IntSuffixLit(42, Box::new(Token::I32))
        );
        assert_eq!(
            Token::FloatSuffixLit(3.14, Box::new(Token::F64)),
            Token::FloatSuffixLit(3.14, Box::new(Token::F64))
        );
        assert_ne!(
            Token::IntSuffixLit(42, Box::new(Token::I32)),
            Token::IntSuffixLit(42, Box::new(Token::I64))
        );
        assert_eq!(Token::Ampersand, Token::Ampersand);
        assert_eq!(Token::Bang, Token::Bang);
        assert_eq!(Token::DoubleColon, Token::DoubleColon);
        assert_eq!(Token::RArrow, Token::RArrow);
        assert_eq!(Token::Dot, Token::Dot);
        assert_eq!(Token::Ellipsis, Token::Ellipsis);
        assert_eq!(Token::Enum, Token::Enum);
        assert_eq!(Token::Pub, Token::Pub);
        assert_eq!(Token::Struct, Token::Struct);
        assert_eq!(Token::Impl, Token::Impl);
        assert_eq!(Token::Type, Token::Type);
        assert_eq!(Token::Trait, Token::Trait);
        assert_eq!(Token::For, Token::For);
        assert_eq!(Token::In, Token::In);
        assert_eq!(Token::Self_, Token::Self_);
        assert_eq!(Token::SelfType, Token::SelfType);
        assert_eq!(Token::Underscore, Token::Underscore);
        assert_eq!(Token::Pound, Token::Pound);
        assert_eq!(Token::LBracket, Token::LBracket);
        assert_eq!(Token::RBracket, Token::RBracket);
        assert_eq!(Token::If, Token::If);
        assert_eq!(Token::Else, Token::Else);
        assert_eq!(Token::Loop, Token::Loop);
        assert_eq!(Token::While, Token::While);
        assert_eq!(Token::Return, Token::Return);
        assert_eq!(Token::EqEq, Token::EqEq);
        assert_eq!(Token::BangEq, Token::BangEq);
        assert_eq!(Token::Lt, Token::Lt);
        assert_eq!(Token::Gt, Token::Gt);
        assert_eq!(Token::Le, Token::Le);
        assert_eq!(Token::Ge, Token::Ge);
        assert_eq!(Token::AndAnd, Token::AndAnd);
        assert_eq!(Token::OrOr, Token::OrOr);
        assert_eq!(Token::Pipe, Token::Pipe);
        assert_eq!(Token::Percent, Token::Percent);
        assert_eq!(Token::Caret, Token::Caret);
        assert_eq!(Token::Eof, Token::Eof);
    }
}

/// A lexical token produced by the lexer and consumed by the parser.
///
/// Each variant corresponds to a keyword, literal, identifier, operator, or piece of
/// punctuation in a ulang source program. Tokens carrying payload data (e.g. [`IntLit`],
/// [`StrLit`], [`Ident`]) store the parsed value directly.
///
/// [`IntLit`]: Token::IntLit
/// [`StrLit`]: Token::StrLit
/// [`Ident`]: Token::Ident
#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    /// The `fn` keyword — function declaration.
    Fn,
    /// The `let` keyword — variable binding.
    Let,
    /// The `mut` keyword — mutable binding or mutable reference.
    Mut,
    /// The `const` keyword — constant declaration.
    Const,
    /// The `if` keyword — conditional expression.
    If,
    /// The `else` keyword — alternative branch.
    Else,
    /// The `loop` keyword — infinite loop.
    Loop,
    /// The `while` keyword — condition-loop.
    While,
    /// The `return` keyword — return from function.
    Return,
    /// The `continue` keyword — skip to next loop iteration.
    Continue,
    /// The `break` keyword — exit a loop.
    Break,
    /// The `mod` keyword — module declaration.
    Mod,
    // Literals
    /// The `bool` type keyword.
    Bool,
    /// The `true` literal keyword.
    True,
    /// The `false` literal keyword.
    False,
    /// An integer literal (e.g. `42`, `0xFF`).
    IntLit(i64),
    /// A floating-point literal (e.g. `3.14`).
    FloatLit(f64),
    /// An integer literal with a type suffix (e.g. `42u32`).
    IntSuffixLit(i64, Box<Token>),
    /// A floating-point literal with a type suffix (e.g. `3.14f64`).
    FloatSuffixLit(f64, Box<Token>),
    // Identifiers
    /// A user-defined identifier (e.g. variable name, function name).
    Ident(String),
    // Operators & Punctuation
    /// The `+` operator — addition.
    Plus,
    /// The `:` punctuation — type annotation separator.
    Colon,
    /// The `-` operator — subtraction.
    Minus,
    /// The `*` operator — multiplication.
    Star,
    /// The `/` operator — division.
    Slash,
    /// The `,` punctuation — item separator.
    Comma,
    /// The `=` operator — assignment.
    Eq,
    /// The `;` punctuation — statement terminator.
    Semicolon,
    /// The `(` punctuation — opening parenthesis.
    LParen,
    /// The `)` punctuation — closing parenthesis.
    RParen,
    /// The `{` punctuation — opening brace (block start).
    LBrace,
    /// The `}` punctuation — closing brace (block end).
    RBrace,
    // Keywords (cont.)
    /// The `as` keyword — type cast.
    As,
    /// The `use` keyword — import declaration.
    Use,
    /// The `extern` keyword — external function declaration.
    Extern,
    // Literals (cont.)
    /// A string literal (e.g. `"hello"`).
    StrLit(String),
    // Operators & Punctuation (cont.)
    /// The `::` punctuation — path separator for qualified names.
    DoubleColon, // ::
    /// The `->` punctuation — return type arrow.
    RArrow, // ->
    /// The `...` punctuation — spread/rest pattern.
    Ellipsis, // ...
    // Type name tokens
    /// The `i8` type keyword — 8-bit signed integer.
    I8,
    /// The `str` type keyword — string slice.
    Str,
    /// The `i16` type keyword — 16-bit signed integer.
    I16,
    /// The `i32` type keyword — 32-bit signed integer.
    I32,
    /// The `i64` type keyword — 64-bit signed integer.
    I64,
    /// The `u8` type keyword — 8-bit unsigned integer.
    U8,
    /// The `u16` type keyword — 16-bit unsigned integer.
    U16,
    /// The `u32` type keyword — 32-bit unsigned integer.
    U32,
    /// The `u64` type keyword — 64-bit unsigned integer.
    U64,
    /// The `usize` type keyword — pointer-width unsigned integer.
    Usize,
    /// The `isize` type keyword — pointer-width signed integer.
    Isize,
    /// The `f32` type keyword — 32-bit floating point.
    F32,
    /// The `f64` type keyword — 64-bit floating point.
    F64,
    /// The `!` operator — logical not or never type annotation.
    Bang, // !
    /// The `!=` operator — inequality comparison.
    BangEq, // !=
    /// The `&` operator — reference creation.
    Ampersand, // &
    /// The `.` punctuation — member access.
    Dot, // .
    // Struct / Impl / Trait / Type / Enum keywords
    /// The `pub` keyword — public visibility.
    Pub,
    /// The `struct` keyword — struct declaration.
    Struct,
    /// The `enum` keyword — enum declaration.
    Enum,
    /// The `impl` keyword — implementation block.
    Impl,
    /// The `trait` keyword — trait declaration.
    Trait,
    /// The `type` keyword — type alias declaration.
    Type,
    /// The `for` keyword — for-loop and trait impl binding.
    For,
    /// The `in` keyword — for-loop iteration source.
    In,
    /// The `Self` keyword — the implementing type in an impl block.
    Self_,
    /// The `Self` type keyword — the implementing type used as a type.
    SelfType,
    /// The `_` pattern — wildcard (matches anything, binds nothing).
    Underscore,
    // New punctuation
    /// The `#` punctuation — attribute marker.
    Pound, // #
    /// The `[` punctuation — opening bracket (array, generic).
    LBracket, // [
    /// The `]` punctuation — closing bracket.
    RBracket, // ]
    // Operators (cont.)
    /// The `==` operator — equality comparison.
    EqEq, // ==
    /// The `<` operator — less-than comparison.
    Lt, // <
    /// The `>` operator — greater-than comparison.
    Gt, // >
    /// The `<=` operator — less-than-or-equal comparison.
    Le, // <=
    /// The `>=` operator — greater-than-or-equal comparison.
    Ge, // >=
    /// The `&&` operator — logical AND.
    AndAnd, // &&
    /// The `||` operator — logical OR.
    OrOr, // ||
    /// The `|` operator — bitwise OR or pattern alternatives.
    Pipe, // |
    // Pattern matching
    /// The `match` keyword — pattern matching expression.
    Match,
    /// The `=>` punctuation — match arm arrow.
    FatArrow, // =>
    /// The `%` operator — remainder/modulo.
    Percent,
    /// The `^` operator — bitwise XOR.
    Caret,
    // Special
    /// End-of-file sentinel — emitted when the lexer reaches the end of input.
    Eof,
}
