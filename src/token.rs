#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Span {
    pub lo: usize,
    pub hi: usize,
}

impl Span {
    pub fn new(lo: usize, hi: usize) -> Self {
        Self { lo, hi }
    }

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
        assert_eq!(Token::Struct, Token::Struct);
        assert_eq!(Token::Impl, Token::Impl);
        assert_eq!(Token::Type, Token::Type);
        assert_eq!(Token::Trait, Token::Trait);
        assert_eq!(Token::For, Token::For);
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
        assert_eq!(Token::Eof, Token::Eof);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Fn,
    Let,
    Mut,
    Const,
    If,
    Else,
    Loop,
    While,
    Return,
    // Literals
    Bool,
    True,
    False,
    IntLit(i64),
    FloatLit(f64),
    IntSuffixLit(i64, Box<Token>),
    FloatSuffixLit(f64, Box<Token>),
    // Identifiers
    Ident(String),
    // Operators & Punctuation
    Plus,
    Colon,
    Minus,
    Star,
    Slash,
    Comma,
    Eq,
    Semicolon,
    LParen,
    RParen,
    LBrace,
    RBrace,
    // Keywords (cont.)
    As,
    Use,
    Extern,
    // Literals (cont.)
    StrLit(String),
    // Operators & Punctuation (cont.)
    DoubleColon, // ::
    RArrow,      // ->
    Ellipsis,    // ...
    // Type name tokens
    I8,
    Str,
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
    Bang,      // !
    BangEq,    // !=
    Ampersand, // &
    Dot,       // .
    // Struct / Impl / Trait / Type / Enum keywords
    Struct,
    Enum,
    Impl,
    Trait,
    Type,
    For,
    Self_,
    SelfType,
    Underscore,
    // New punctuation
    Pound,    // #
    LBracket, // [
    RBracket, // ]
    // Operators (cont.)
    EqEq, // ==
    Lt,   // <
    Gt,   // >
    Le,   // <=
    Ge,   // >=
    // Pattern matching
    Match,
    FatArrow, // =>
    // Special
    Eof,
}
