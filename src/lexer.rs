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
                if self.current_char() == Some('>') {
                    self.advance();
                    Token::RArrow
                } else {
                    Token::Minus
                }
            }
            '*' => {
                self.advance();
                Token::Star
            }
            '/' => {
                self.advance();
                Token::Slash
            }
            ':' => {
                self.advance();
                if self.current_char() == Some(':') {
                    self.advance();
                    Token::DoubleColon
                } else {
                    Token::Colon
                }
            }
            '=' => {
                self.advance();
                if self.current_char() == Some('=') {
                    self.advance();
                    Token::EqEq
                } else if self.current_char() == Some('>') {
                    self.advance();
                    Token::FatArrow
                } else {
                    Token::Eq
                }
            }
            ';' => {
                self.advance();
                Token::Semicolon
            }
            '.' => {
                self.advance();
                if self.current_char() == Some('.') {
                    self.advance();
                    if self.current_char() == Some('.') {
                        self.advance();
                        Token::Ellipsis
                    } else {
                        return Err("unexpected token '..'".to_string());
                    }
                } else {
                    Token::Dot
                }
            }
            '(' => {
                self.advance();
                Token::LParen
            }
            ',' => {
                self.advance();
                Token::Comma
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
            '&' => {
                self.advance();
                if self.current_char() == Some('&') {
                    self.advance();
                    Token::AndAnd
                } else {
                    Token::Ampersand
                }
            }
            '|' => {
                self.advance();
                if self.current_char() == Some('|') {
                    self.advance();
                    Token::OrOr
                } else {
                    Token::Pipe
                }
            }
            '!' => {
                self.advance();
                if self.current_char() == Some('=') {
                    self.advance();
                    Token::BangEq
                } else {
                    Token::Bang
                }
            }
            '<' => {
                self.advance();
                if self.current_char() == Some('=') {
                    self.advance();
                    Token::Le
                } else {
                    Token::Lt
                }
            }
            '>' => {
                self.advance();
                if self.current_char() == Some('=') {
                    self.advance();
                    Token::Ge
                } else {
                    Token::Gt
                }
            }
            '#' => {
                self.advance();
                Token::Pound
            }
            '[' => {
                self.advance();
                Token::LBracket
            }
            ']' => {
                self.advance();
                Token::RBracket
            }
            '"' => {
                self.advance(); // consume opening "
                let s = self.read_string()?;
                Token::StrLit(s)
            }
            c if c.is_ascii_digit() => self.read_number_or_float()?,
            c if c.is_ascii_alphabetic() || c == '_' => {
                let ident = self.read_identifier();
                match ident.as_str() {
                    "fn" => Token::Fn,
                    "let" => Token::Let,
                    "mod" => Token::Mod,
                    "mut" => Token::Mut,
                    "const" => Token::Const,
                    "use" => Token::Use,
                    "extern" => Token::Extern,
                    "as" => Token::As,
                    // Type names
                    "i8" => Token::I8,
                    "i16" => Token::I16,
                    "i32" => Token::I32,
                    "i64" => Token::I64,
                    "u8" => Token::U8,
                    "u16" => Token::U16,
                    "u32" => Token::U32,
                    "u64" => Token::U64,
                    "usize" => Token::Usize,
                    "isize" => Token::Isize,
                    "f32" => Token::F32,
                    "f64" => Token::F64,
                    "str" => Token::Str,
                    "bool" => Token::Bool,
                    "true" => Token::True,
                    "false" => Token::False,
                    "if" => Token::If,
                    "else" => Token::Else,
                    "loop" => Token::Loop,
                    "while" => Token::While,
                    "match" => Token::Match,
                    "return" => Token::Return,
                    "enum" => Token::Enum,
                    "pub" => Token::Pub,
                    "struct" => Token::Struct,
                    "type" => Token::Type,
                    "impl" => Token::Impl,
                    "trait" => Token::Trait,
                    "for" => Token::For,
                    "in" => Token::In,
                    "self" => Token::Self_,
                    "Self" => Token::SelfType,
                    "_" => Token::Underscore,
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

    fn read_number_or_float(&mut self) -> Result<Token, String> {
        let lo = self.pos;
        while self.pos < self.source.len() && self.current_char().unwrap().is_ascii_digit() {
            self.advance();
        }
        // Check for float literal (digit-sequence . digit-sequence)
        if self.pos < self.source.len() && self.current_char() == Some('.') {
            // Check if the next char after '.' is a digit (to avoid parsing e.g. "foo.bar")
            let saved = self.pos;
            self.advance(); // consume '.'
            if self.pos < self.source.len() && self.current_char().unwrap().is_ascii_digit() {
                while self.pos < self.source.len() && self.current_char().unwrap().is_ascii_digit()
                {
                    self.advance();
                }
                let val: f64 = self.source[lo..self.pos].parse().map_err(|_| {
                    format!("invalid float literal '{}'", &self.source[lo..self.pos])
                })?;
                if let Some(suffix_token) = self.try_read_type_suffix() {
                    return Ok(Token::FloatSuffixLit(val, Box::new(suffix_token)));
                }
                return Ok(Token::FloatLit(val));
            } else {
                // Not a float, restore position and treat as integer
                self.pos = saved;
            }
        }
        let val: i64 = self.source[lo..self.pos]
            .parse()
            .map_err(|_| format!("invalid integer literal '{}'", &self.source[lo..self.pos]))?;
        // Check for type suffix like 42i32, 255u8
        if let Some(suffix_token) = self.try_read_type_suffix() {
            Ok(Token::IntSuffixLit(val, Box::new(suffix_token)))
        } else {
            Ok(Token::IntLit(val))
        }
    }

    /// After reading the numeric part, check if the upcoming chars form a type name suffix.
    /// If yes, consume them and return the corresponding type name token.
    /// If no, return None without advancing the position.
    fn try_read_type_suffix(&mut self) -> Option<Token> {
        let saved = self.pos;
        let ident = self.read_identifier();
        match ident.as_str() {
            "i8" => Some(Token::I8),
            "i16" => Some(Token::I16),
            "i32" => Some(Token::I32),
            "i64" => Some(Token::I64),
            "u8" => Some(Token::U8),
            "u16" => Some(Token::U16),
            "u32" => Some(Token::U32),
            "u64" => Some(Token::U64),
            "usize" => Some(Token::Usize),
            "isize" => Some(Token::Isize),
            "f32" => Some(Token::F32),
            "f64" => Some(Token::F64),
            _ => {
                // Not a type suffix — restore position
                self.pos = saved;
                None
            }
        }
    }

    fn read_string(&mut self) -> Result<String, String> {
        let mut s = String::new();
        loop {
            match self.current_char() {
                Some('"') => {
                    self.advance(); // consume closing "
                    return Ok(s);
                }
                Some('\\') => {
                    self.advance();
                    match self.current_char() {
                        Some('n') => {
                            s.push('\n');
                            self.advance();
                        }
                        Some('\\') => {
                            s.push('\\');
                            self.advance();
                        }
                        Some('"') => {
                            s.push('"');
                            self.advance();
                        }
                        Some(c) => {
                            return Err(format!("invalid escape sequence '\\{}'", c));
                        }
                        None => {
                            return Err("unterminated string literal after backslash".to_string());
                        }
                    }
                }
                Some(c) => {
                    s.push(c);
                    self.advance();
                }
                None => {
                    return Err("unterminated string literal".to_string());
                }
            }
        }
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
        let tokens = lex_all("fn let mut const mod pub in");
        assert_eq!(tokens[0].0, Token::Fn);
        assert_eq!(tokens[1].0, Token::Let);
        assert_eq!(tokens[2].0, Token::Mut);
        assert_eq!(tokens[3].0, Token::Const);
        assert_eq!(tokens[4].0, Token::Mod);
        assert_eq!(tokens[5].0, Token::Pub);
        assert_eq!(tokens[6].0, Token::In);
        assert!(matches!(tokens[7].0, Token::Eof));
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
    fn test_float_literal() {
        let tokens = lex_all("3.14");
        assert_eq!(tokens[0].0, Token::FloatLit(3.14));
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_float_leading_zero() {
        let tokens = lex_all("0.5");
        assert_eq!(tokens[0].0, Token::FloatLit(0.5));
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_colon_token() {
        let tokens = lex_all(":");
        assert_eq!(tokens[0].0, Token::Colon);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_ampersand_token() {
        let tokens = lex_all("&");
        assert_eq!(tokens[0].0, Token::Ampersand);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_as_keyword() {
        let tokens = lex_all("as");
        assert_eq!(tokens[0].0, Token::As);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_bool_keyword() {
        let tokens = lex_all("bool");
        assert_eq!(tokens[0].0, Token::Bool);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_true_false_literals() {
        let tokens = lex_all("true false");
        assert_eq!(tokens[0].0, Token::True);
        assert_eq!(tokens[1].0, Token::False);
        assert!(matches!(tokens[2].0, Token::Eof));
    }

    #[test]
    fn test_type_name_tokens() {
        let types = [
            ("i8", Token::I8),
            ("i16", Token::I16),
            ("i32", Token::I32),
            ("i64", Token::I64),
            ("u8", Token::U8),
            ("u16", Token::U16),
            ("u32", Token::U32),
            ("u64", Token::U64),
            ("usize", Token::Usize),
            ("isize", Token::Isize),
            ("f32", Token::F32),
            ("f64", Token::F64),
        ];
        for (s, expected) in &types {
            let tokens = lex_all(s);
            assert_eq!(tokens[0].0, *expected, "type token '{}' mismatch", s);
            assert!(matches!(tokens[1].0, Token::Eof));
        }
    }

    #[test]
    fn test_typed_let() {
        let tokens = lex_all("let x: i32 = 42");
        assert_eq!(tokens[0].0, Token::Let);
        assert_eq!(tokens[1].0, Token::Ident("x".into()));
        assert_eq!(tokens[2].0, Token::Colon);
        assert_eq!(tokens[3].0, Token::I32);
        assert_eq!(tokens[4].0, Token::Eq);
        assert_eq!(tokens[5].0, Token::IntLit(42));
        assert!(matches!(tokens[6].0, Token::Eof));
    }

    #[test]
    fn test_as_cast_syntax() {
        let tokens = lex_all("x as f64");
        assert_eq!(tokens[0].0, Token::Ident("x".into()));
        assert_eq!(tokens[1].0, Token::As);
        assert_eq!(tokens[2].0, Token::F64);
        assert!(matches!(tokens[3].0, Token::Eof));
    }

    #[test]
    fn test_int_suffix_lit() {
        let tokens = lex_all("42i32");
        assert_eq!(tokens[0].0, Token::IntSuffixLit(42, Box::new(Token::I32)));
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_u8_suffix_lit() {
        let tokens = lex_all("255u8");
        assert_eq!(tokens[0].0, Token::IntSuffixLit(255, Box::new(Token::U8)));
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_i64_suffix_lit() {
        let tokens = lex_all("1000i64");
        assert_eq!(tokens[0].0, Token::IntSuffixLit(1000, Box::new(Token::I64)));
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_float_suffix_lit() {
        let tokens = lex_all("3.14f64");
        assert_eq!(
            tokens[0].0,
            Token::FloatSuffixLit(3.14, Box::new(Token::F64))
        );
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_f32_suffix_lit() {
        let tokens = lex_all("1.5f32");
        assert_eq!(
            tokens[0].0,
            Token::FloatSuffixLit(1.5, Box::new(Token::F32))
        );
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_if_keyword() {
        let tokens = lex_all("if");
        assert_eq!(tokens[0].0, Token::If);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_else_keyword() {
        let tokens = lex_all("else");
        assert_eq!(tokens[0].0, Token::Else);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_loop_keyword() {
        let tokens = lex_all("loop");
        assert_eq!(tokens[0].0, Token::Loop);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_while_keyword() {
        let tokens = lex_all("while");
        assert_eq!(tokens[0].0, Token::While);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_return_keyword() {
        let tokens = lex_all("return");
        assert_eq!(tokens[0].0, Token::Return);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_eqeq_operator() {
        let tokens = lex_all("==");
        assert_eq!(tokens[0].0, Token::EqEq);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_bangeq_operator() {
        let tokens = lex_all("!=");
        assert_eq!(tokens[0].0, Token::BangEq);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_lt_operator() {
        let tokens = lex_all("<");
        assert_eq!(tokens[0].0, Token::Lt);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_gt_operator() {
        let tokens = lex_all(">");
        assert_eq!(tokens[0].0, Token::Gt);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_le_operator() {
        let tokens = lex_all("<=");
        assert_eq!(tokens[0].0, Token::Le);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_ge_operator() {
        let tokens = lex_all(">=");
        assert_eq!(tokens[0].0, Token::Ge);
        assert!(matches!(tokens[1].0, Token::Eof));
    }

    #[test]
    fn test_bang_equals_does_not_break_bang() {
        let tokens = lex_all("!x");
        assert_eq!(tokens[0].0, Token::Bang);
        assert_eq!(tokens[1].0, Token::Ident("x".into()));
        assert!(matches!(tokens[2].0, Token::Eof));
    }

    #[test]
    fn test_equals_does_not_break_eqeq() {
        let tokens = lex_all("=x");
        assert_eq!(tokens[0].0, Token::Eq);
        assert_eq!(tokens[1].0, Token::Ident("x".into()));
        assert!(matches!(tokens[2].0, Token::Eof));
    }

    #[test]
    fn test_control_flow_lexing() {
        let tokens = lex_all("if cond { 1 } else { 2 }");
        assert_eq!(tokens[0].0, Token::If);
        assert_eq!(tokens[1].0, Token::Ident("cond".into()));
        assert_eq!(tokens[2].0, Token::LBrace);
        assert_eq!(tokens[3].0, Token::IntLit(1));
        assert_eq!(tokens[4].0, Token::RBrace);
        assert_eq!(tokens[5].0, Token::Else);
        assert_eq!(tokens[6].0, Token::LBrace);
        assert_eq!(tokens[7].0, Token::IntLit(2));
        assert_eq!(tokens[8].0, Token::RBrace);
        assert!(matches!(tokens[9].0, Token::Eof));
    }

    #[test]
    fn test_comparison_in_expression() {
        let tokens = lex_all("x == y");
        assert_eq!(tokens[0].0, Token::Ident("x".into()));
        assert_eq!(tokens[1].0, Token::EqEq);
        assert_eq!(tokens[2].0, Token::Ident("y".into()));
        assert!(matches!(tokens[3].0, Token::Eof));
    }

    #[test]
    fn test_suffix_in_expression() {
        let tokens = lex_all("let x = 42i32;");
        assert_eq!(tokens[0].0, Token::Let);
        assert_eq!(tokens[1].0, Token::Ident("x".into()));
        assert_eq!(tokens[2].0, Token::Eq);
        assert_eq!(tokens[3].0, Token::IntSuffixLit(42, Box::new(Token::I32)));
        assert_eq!(tokens[4].0, Token::Semicolon);
        assert!(matches!(tokens[5].0, Token::Eof));
    }

    #[test]
    fn test_non_type_suffix_not_consumed() {
        // 42foo should lex as IntLit(42) then Ident("foo")
        let tokens = lex_all("42foo");
        assert_eq!(tokens[0].0, Token::IntLit(42));
        assert_eq!(tokens[1].0, Token::Ident("foo".into()));
        assert!(matches!(tokens[2].0, Token::Eof));
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

    #[test]
    fn test_logical_operators() {
        let tokens = lex_all("&& || & |");
        assert_eq!(tokens[0].0, Token::AndAnd);
        assert_eq!(tokens[1].0, Token::OrOr);
        assert_eq!(tokens[2].0, Token::Ampersand);
        assert_eq!(tokens[3].0, Token::Pipe);
        assert!(matches!(tokens[4].0, Token::Eof));
    }
}
