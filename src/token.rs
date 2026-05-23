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
        assert_eq!(Token::Eof, Token::Eof);
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    // Keywords
    Fn,
    Let,
    // Literals
    IntLit(i64),
    // Identifiers
    Ident(String),
    // Operators & Punctuation
    Plus,
    Minus,
    Star,
    Slash,
    Eq,
    Semicolon,
    LParen,
    RParen,
    LBrace,
    RBrace,
    // Special
    Eof,
}
