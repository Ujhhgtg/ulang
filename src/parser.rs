use crate::ast::{BinOp, Block, Expr, Function, Program, Stmt};
use crate::token::{Span, Token};

#[derive(Debug)]
pub struct ParseError {
    pub span: Span,
    pub msg: String,
}

pub struct Parser<'a> {
    tokens: &'a [(Token, Span)],
    pos: usize,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [(Token, Span)]) -> Self {
        Self { tokens, pos: 0 }
    }

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut funcs = Vec::new();
        while *self.peek_token() != Token::Eof {
            let func = self.parse_function()?;
            funcs.push(func);
        }
        Ok(Program { funcs })
    }

    fn parse_function(&mut self) -> Result<Function, ParseError> {
        self.expect(&Token::Fn)?;
        let name = match self.peek_token() {
            Token::Ident(s) => {
                let s = s.clone();
                self.advance();
                s
            }
            _ => {
                let (_, span) = self.current().unwrap();
                return Err(ParseError {
                    span: *span,
                    msg: "expected function name".to_string(),
                });
            }
        };
        self.expect(&Token::LParen)?;
        self.expect(&Token::RParen)?;
        let body = self.parse_block()?;
        Ok(Function { name, body })
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let lo = match self.current() {
            Some((_, span)) => span.lo,
            None => self.last_span_end(),
        };
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        while *self.peek_token() != Token::RBrace && *self.peek_token() != Token::Eof {
            let stmt = self.parse_stmt()?;
            stmts.push(stmt);
        }
        let hi = match self.current() {
            Some((_, span)) => span.hi,
            None => self.last_span_end(),
        };
        self.expect(&Token::RBrace)?;
        Ok(Block {
            stmts,
            span: Span::new(lo, hi),
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek_token() {
            Token::Let => {
                let lo = match self.current() {
                    Some((_, span)) => span.lo,
                    None => self.last_span_end(),
                };
                self.advance();
                let name = match self.peek_token() {
                    Token::Ident(s) => {
                        let s = s.clone();
                        self.advance();
                        s
                    }
                    _ => {
                        let (_, span) = self.current().unwrap();
                        return Err(ParseError {
                            span: *span,
                            msg: "expected variable name after 'let'".to_string(),
                        });
                    }
                };
                self.expect(&Token::Eq)?;
                let init = self.parse_expr()?;
                self.expect(&Token::Semicolon)?;
                let hi = self.last_span_end();
                Ok(Stmt::Let {
                    name,
                    init,
                    span: Span::new(lo, hi),
                })
            }
            _ => {
                let expr = self.parse_expr()?;
                self.expect(&Token::Semicolon)?;
                Ok(Stmt::Expr(expr))
            }
        }
    }

    fn parse_expr(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_term()?;
        loop {
            match self.peek_token() {
                Token::Plus => {
                    self.advance();
                    let rhs = self.parse_term()?;
                    lhs = Expr::Binary {
                        op: BinOp::Add,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                Token::Minus => {
                    self.advance();
                    let rhs = self.parse_term()?;
                    lhs = Expr::Binary {
                        op: BinOp::Sub,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_term(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_primary()?;
        loop {
            match self.peek_token() {
                Token::Star => {
                    self.advance();
                    let rhs = self.parse_primary()?;
                    lhs = Expr::Binary {
                        op: BinOp::Mul,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                Token::Slash => {
                    self.advance();
                    let rhs = self.parse_primary()?;
                    lhs = Expr::Binary {
                        op: BinOp::Div,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                _ => break,
            }
        }
        Ok(lhs)
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        match self.peek_token() {
            Token::IntLit(val) => {
                let val = *val;
                self.advance();
                Ok(Expr::IntLit(val))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance();
                if *self.peek_token() == Token::LParen {
                    self.advance(); // '('
                    let arg = self.parse_expr()?;
                    self.expect(&Token::RParen)?;
                    return Ok(Expr::Call {
                        callee: name,
                        arg: Box::new(arg),
                    });
                }
                Ok(Expr::Ident(name))
            }
            Token::LParen => {
                self.advance();
                let expr = self.parse_expr()?;
                self.expect(&Token::RParen)?;
                Ok(expr)
            }
            _ => {
                let default = (Token::Eof, Span::empty(0));
                let (_, span) = self.current().unwrap_or(&default);
                Err(ParseError {
                    span: *span,
                    msg: format!("unexpected token {:?}", self.peek_token()),
                })
            }
        }
    }

    fn peek_token(&self) -> &Token {
        let idx = self.pos.min(self.tokens.len().saturating_sub(1));
        &self.tokens[idx].0
    }

    fn current(&self) -> Option<&(Token, Span)> {
        self.tokens.get(self.pos)
    }

    fn advance(&mut self) {
        if self.pos < self.tokens.len() {
            self.pos += 1;
        }
    }

    fn expect(&mut self, expected: &Token) -> Result<(), ParseError> {
        if self.peek_token() == expected {
            self.advance();
            Ok(())
        } else {
            let default = (Token::Eof, Span::empty(0));
            let (_, span) = self.current().unwrap_or(&default);
            Err(ParseError {
                span: *span,
                msg: format!("expected {:?}, found {:?}", expected, self.peek_token()),
            })
        }
    }

    fn last_span_end(&self) -> usize {
        if self.pos == 0 {
            return 0;
        }
        self.tokens[self.pos - 1].1.hi
    }
}

impl std::fmt::Display for ParseError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.msg)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::token::Token;

    fn parse(src: &str) -> Result<Program, ParseError> {
        let mut lexer = Lexer::new(src);
        let mut tokens = Vec::new();
        loop {
            let (token, span) = lexer.next_token().expect("lexer error");
            tokens.push((token, span));
            if matches!(tokens.last().unwrap().0, Token::Eof) {
                break;
            }
        }
        let mut parser = Parser::new(&tokens);
        parser.parse_program()
    }

    #[test]
    fn test_empty_program() {
        let prog = parse("").unwrap();
        assert!(prog.funcs.is_empty());
    }

    #[test]
    fn test_single_function_no_body() {
        let prog = parse("fn foo() {}").unwrap();
        assert_eq!(prog.funcs.len(), 1);
        assert_eq!(prog.funcs[0].name, "foo");
        assert!(prog.funcs[0].body.stmts.is_empty());
    }

    #[test]
    fn test_function_with_let() {
        let prog = parse("fn main() { let x = 42; }").unwrap();
        assert_eq!(prog.funcs.len(), 1);
        assert_eq!(prog.funcs[0].name, "main");
        assert_eq!(prog.funcs[0].body.stmts.len(), 1);
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { name, init, .. } => {
                assert_eq!(name, "x");
                assert!(matches!(init, Expr::IntLit(42)));
            }
            _ => panic!("expected Let stmt"),
        }
    }

    #[test]
    fn test_function_with_expr() {
        let prog = parse("fn main() { 42; }").unwrap();
        assert_eq!(prog.funcs[0].body.stmts.len(), 1);
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::IntLit(42)) => {}
            _ => panic!("expected Expr(IntLit(42))"),
        }
    }

    #[test]
    fn test_binary_add() {
        let prog = parse("fn f() { 1+2; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Binary {
                op: BinOp::Add,
                lhs,
                rhs,
            }) => {
                assert!(matches!(lhs.as_ref(), Expr::IntLit(1)));
                assert!(matches!(rhs.as_ref(), Expr::IntLit(2)));
            }
            _ => panic!("expected Binary(Add, 1, 2)"),
        }
    }

    #[test]
    fn test_binary_sub() {
        let prog = parse("fn f() { 1-2; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Binary { op: BinOp::Sub, .. }) => {}
            _ => panic!("expected Binary(Sub, ..)"),
        }
    }

    #[test]
    fn test_binary_mul() {
        let prog = parse("fn f() { 1*2; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Binary { op: BinOp::Mul, .. }) => {}
            _ => panic!("expected Binary(Mul, ..)"),
        }
    }

    #[test]
    fn test_binary_div() {
        let prog = parse("fn f() { 1/2; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Binary { op: BinOp::Div, .. }) => {}
            _ => panic!("expected Binary(Div, ..)"),
        }
    }

    #[test]
    fn test_precedence() {
        // 1+2*3 should parse as +(1, *(2, 3))
        let prog = parse("fn f() { 1+2*3; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Binary {
                op: BinOp::Add,
                lhs,
                rhs,
            }) => {
                assert!(matches!(lhs.as_ref(), Expr::IntLit(1)));
                match rhs.as_ref() {
                    Expr::Binary {
                        op: BinOp::Mul,
                        lhs: inner_lhs,
                        rhs: inner_rhs,
                    } => {
                        assert!(matches!(inner_lhs.as_ref(), Expr::IntLit(2)));
                        assert!(matches!(inner_rhs.as_ref(), Expr::IntLit(3)));
                    }
                    _ => panic!("expected Mul as rhs of Add"),
                }
            }
            _ => panic!("expected Binary(Add, ..)"),
        }
    }

    #[test]
    fn test_parentheses() {
        // (1+2)*3 should parse as *(+(1, 2), 3)
        let prog = parse("fn f() { (1+2)*3; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Binary {
                op: BinOp::Mul,
                lhs,
                rhs,
            }) => {
                match lhs.as_ref() {
                    Expr::Binary {
                        op: BinOp::Add,
                        lhs: inner_lhs,
                        rhs: inner_rhs,
                    } => {
                        assert!(matches!(inner_lhs.as_ref(), Expr::IntLit(1)));
                        assert!(matches!(inner_rhs.as_ref(), Expr::IntLit(2)));
                    }
                    _ => panic!("expected Add as lhs of Mul"),
                }
                assert!(matches!(rhs.as_ref(), Expr::IntLit(3)));
            }
            _ => panic!("expected Binary(Mul, ..)"),
        }
    }

    #[test]
    fn test_call() {
        let prog = parse("fn f() { print(42); }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Call { callee, arg }) => {
                assert_eq!(callee, "print");
                assert!(matches!(arg.as_ref(), Expr::IntLit(42)));
            }
            _ => panic!("expected Call(print, 42)"),
        }
    }

    #[test]
    fn test_multiple_functions() {
        let prog = parse("fn a(){} fn b(){}").unwrap();
        assert_eq!(prog.funcs.len(), 2);
        assert_eq!(prog.funcs[0].name, "a");
        assert_eq!(prog.funcs[1].name, "b");
    }

    #[test]
    fn test_invalid_missing_semicolon() {
        let result = parse("fn f() { 42 }");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().msg.contains("expected"),
            "error should mention expected token"
        );
    }

    #[test]
    fn test_invalid_missing_fn_name() {
        let result = parse("fn ()");
        assert!(result.is_err());
        assert!(result.unwrap_err().msg.contains("function name"));
    }

    #[test]
    fn test_invalid_trailing_garbage() {
        // The '!' character causes a lexer error before the parser runs.
        // Verify that the combined pipeline rejects it.
        let mut lexer = Lexer::new("fn f() {} !");
        let mut lex_err = None;
        let mut tokens = Vec::new();
        loop {
            match lexer.next_token() {
                Ok((token, span)) => {
                    tokens.push((token, span));
                    if matches!(tokens.last().unwrap().0, Token::Eof) {
                        break;
                    }
                }
                Err(e) => {
                    lex_err = Some(e);
                    break;
                }
            }
        }
        if let Some(err) = lex_err {
            assert!(!err.is_empty(), "lexer should produce a non-empty error");
        } else {
            // If lexing succeeded, the parser should still reject it
            let mut parser = Parser::new(&tokens);
            assert!(parser.parse_program().is_err());
        }
    }
}
