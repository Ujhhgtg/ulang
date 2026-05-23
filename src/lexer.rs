use crate::token::{Span, Token};

pub struct Lexer<'a> {
    source: &'a str,
    pos: usize,
}

impl<'a> Lexer<'a> {
    pub fn new(source: &'a str) -> Self {
        Self { source, pos: 0 }
    }
    /// Return the current byte position in source (for error reporting)
    pub fn pos(&self) -> usize {
        self.pos
    }

    pub fn next_token(&mut self) -> Result<(Token, Span), String> {
        self.skip_whitespace();

        if self.pos >= self.source.len() {
            let pos = self.pos;
            return Ok((Token::Eof, Span::empty(pos)));
        }

        let lo = self.pos;
        let c = self.current_char().unwrap();

        // Single-line comments
        if c == '/' && self.peek_next() == Some('/') {
            self.advance(); // skip first /
            self.advance(); // skip second /
            while self.pos < self.source.len() && self.current_char().unwrap() != '\n' {
                self.advance();
            }
            return self.next_token();
        }

        let token = match c {
            '+' => {
                self.advance();
                Token::Plus
            }
            '-' => {
                self.advance();
                Token::Minus
            }
            '*' => {
                self.advance();
                Token::Star
            }
            '/' => {
                self.advance();
                Token::Slash
            }
            '=' => {
                self.advance();
                Token::Eq
            }
            ';' => {
                self.advance();
                Token::Semicolon
            }
            '(' => {
                self.advance();
                Token::LParen
            }
            ')' => {
                self.advance();
                Token::RParen
            }
            '{' => {
                self.advance();
                Token::LBrace
            }
            '}' => {
                self.advance();
                Token::RBrace
            }
            c if c.is_ascii_digit() => {
                let num = self.read_number();
                Token::IntLit(num)
            }
            c if c.is_ascii_alphabetic() || c == '_' => {
                let ident = self.read_identifier();
                match ident.as_str() {
                    "fn" => Token::Fn,
                    "let" => Token::Let,
                    _ => Token::Ident(ident),
                }
            }
            _ => {
                return Err(format!("unexpected character '{}'", c));
            }
        };

        let hi = self.pos;
        Ok((token, Span::new(lo, hi)))
    }

    fn skip_whitespace(&mut self) {
        while self.pos < self.source.len() {
            let c = self.current_char().unwrap();
            if c == ' ' || c == '\t' || c == '\n' || c == '\r' {
                self.advance();
            } else {
                break;
            }
        }
    }

    fn read_number(&mut self) -> i64 {
        let lo = self.pos;
        while self.pos < self.source.len() && self.current_char().unwrap().is_ascii_digit() {
            self.advance();
        }
        self.source[lo..self.pos].parse().unwrap()
    }

    fn read_identifier(&mut self) -> String {
        let lo = self.pos;
        while self.pos < self.source.len() {
            let c = self.current_char().unwrap();
            if c.is_ascii_alphanumeric() || c == '_' {
                self.advance();
            } else {
                break;
            }
        }
        self.source[lo..self.pos].to_string()
    }

    fn current_char(&self) -> Option<char> {
        self.source[self.pos..].chars().next()
    }

    fn peek_next(&self) -> Option<char> {
        let mut chars = self.source[self.pos..].chars();
        chars.next()?;
        chars.next()
    }

    fn advance(&mut self) {
        if self.pos < self.source.len() {
            self.pos += self.current_char().unwrap().len_utf8();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn lex_all(src: &str) -> Vec<(Token, Span)> {
        let mut lexer = Lexer::new(src);
        let mut tokens = Vec::new();
        loop {
            let (token, span) = lexer.next_token().unwrap();
            let is_eof = matches!(token, Token::Eof);
            tokens.push((token, span));
            if is_eof {
                break;
            }
        }
        tokens
    }

    #[test]
    fn test_empty_source() {
        let tokens = lex_all("");
        assert_eq!(tokens.len(), 1);
        assert_eq!(tokens[0].0, Token::Eof);
    }

    #[test]
    fn test_integer() {
        let tokens = lex_all("42");
        assert_eq!(tokens[0].0, Token::IntLit(42));
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_large_integer() {
        let tokens = lex_all("123456789");
        assert_eq!(tokens[0].0, Token::IntLit(123456789));
    }

    #[test]
    fn test_identifiers() {
        let tokens = lex_all("foo _bar x42");
        assert_eq!(tokens[0].0, Token::Ident("foo".into()));
        assert_eq!(tokens[1].0, Token::Ident("_bar".into()));
        assert_eq!(tokens[2].0, Token::Ident("x42".into()));
        assert!(matches!(tokens[3].0, Token::Eof));
    }

    #[test]
    fn test_keywords() {
        let tokens = lex_all("fn let");
        assert_eq!(tokens[0].0, Token::Fn);
        assert_eq!(tokens[1].0, Token::Let);
        assert!(matches!(tokens[2].0, Token::Eof));
    }

    #[test]
    fn test_operators_and_punctuation() {
        let tokens = lex_all("+ - * / = ; ( ) { }");
        assert_eq!(tokens[0].0, Token::Plus);
        assert_eq!(tokens[1].0, Token::Minus);
        assert_eq!(tokens[2].0, Token::Star);
        assert_eq!(tokens[3].0, Token::Slash);
        assert_eq!(tokens[4].0, Token::Eq);
        assert_eq!(tokens[5].0, Token::Semicolon);
        assert_eq!(tokens[6].0, Token::LParen);
        assert_eq!(tokens[7].0, Token::RParen);
        assert_eq!(tokens[8].0, Token::LBrace);
        assert_eq!(tokens[9].0, Token::RBrace);
        assert!(matches!(tokens[10].0, Token::Eof));
    }

    #[test]
    fn test_whitespace() {
        let tokens = lex_all("  let  x\t=\n42");
        assert_eq!(tokens[0].0, Token::Let);
        assert_eq!(tokens[1].0, Token::Ident("x".into()));
        assert_eq!(tokens[2].0, Token::Eq);
        assert_eq!(tokens[3].0, Token::IntLit(42));
        assert!(matches!(tokens[4].0, Token::Eof));
    }

    #[test]
    fn test_comment() {
        let tokens = lex_all("// hello\nlet");
        assert_eq!(tokens[0].0, Token::Let);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_comment_eof() {
        let tokens = lex_all("// hello");
        assert!(matches!(tokens[0].0, Token::Eof));
    }

    #[test]
    fn test_mixed() {
        let tokens = lex_all("fn main() { print(2+3); }");
        assert_eq!(tokens[0].0, Token::Fn);
        assert_eq!(tokens[1].0, Token::Ident("main".into()));
        assert_eq!(tokens[2].0, Token::LParen);
        assert_eq!(tokens[3].0, Token::RParen);
        assert_eq!(tokens[4].0, Token::LBrace);
        assert_eq!(tokens[5].0, Token::Ident("print".into()));
        assert_eq!(tokens[6].0, Token::LParen);
        assert_eq!(tokens[7].0, Token::IntLit(2));
        assert_eq!(tokens[8].0, Token::Plus);
        assert_eq!(tokens[9].0, Token::IntLit(3));
        assert_eq!(tokens[10].0, Token::RParen);
        assert_eq!(tokens[11].0, Token::Semicolon);
        assert_eq!(tokens[12].0, Token::RBrace);
        assert!(matches!(tokens[13].0, Token::Eof));
    }

    #[test]
    fn test_invalid_char() {
        let mut lexer = Lexer::new("@");
        let result = lexer.next_token();
        assert!(result.is_err());
        assert_eq!(result.unwrap_err(), "unexpected character '@'");
    }

    #[test]
    fn test_spans_are_contiguous() {
        let src = "fn foo() {}";
        let mut lexer = Lexer::new(src);
        let mut prev_hi = 0;
        loop {
            let (token, span) = lexer.next_token().unwrap();
            if matches!(token, Token::Eof) {
                break;
            }
            // Each token's span should start at or after the previous end
            assert!(
                span.lo >= prev_hi,
                "span {:?} starts before previous end {}",
                span,
                prev_hi
            );
            prev_hi = span.hi;
        }
    }
}
