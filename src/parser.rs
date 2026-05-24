use std::collections::{HashMap, HashSet};

use crate::ast::{
    Attribute, BinOp, Block, EnumDecl, EnumVariant, Expr, Function, ImplDecl, Param, Program,
    Stmt, StructDecl, StructField, TraitDecl, TraitMethodDef, Type, TypeAliasDecl, Use,
};
use crate::token::{Span, Token};

#[derive(Debug)]
pub struct ParseError {
    pub span: Span,
    pub msg: String,
}

pub struct Parser<'a> {
    tokens: &'a [(Token, Span)],
    pos: usize,
    struct_defs: HashMap<String, Vec<StructField>>,
    type_aliases: HashMap<String, (Vec<String>, Type)>,
    struct_names: HashSet<String>,
    enum_names: HashSet<String>,
}

impl<'a> Parser<'a> {
    pub fn new(tokens: &'a [(Token, Span)]) -> Self {
        Self {
            tokens,
            pos: 0,
            struct_defs: HashMap::new(),
            type_aliases: HashMap::new(),
            struct_names: HashSet::new(),
            enum_names: HashSet::new(),
        }
    }

    pub fn parse_program(&mut self) -> Result<Program, ParseError> {
        let mut uses = Vec::new();
        let mut funcs = Vec::new();
        let mut structs = Vec::new();
        let mut enums = Vec::new();
        let mut traits = Vec::new();
        let mut impls = Vec::new();
        let mut type_aliases = Vec::new();

        loop {
            let attribs = self.parse_attrs()?;

            match self.peek_token() {
                Token::Eof => break,
                Token::Use => {
                    uses.push(self.parse_use_decl()?);
                }
                Token::Extern => {
                    self.parse_extern_block(&mut funcs)?;
                }
                Token::Enum => {
                    let decl = self.parse_enum_decl(attribs)?;
                    // Check for duplicate name
                    if self.struct_names.contains(&decl.name)
                        || self.type_aliases.contains_key(&decl.name)
                    {
                        return Err(ParseError {
                            span: decl.span,
                            msg: format!(
                                "enum `{}` conflicts with existing declaration",
                                decl.name
                            ),
                        });
                    }
                    self.enum_names.insert(decl.name.clone());
                    enums.push(decl);
                }
                Token::Struct => {
                    let decl = self.parse_struct_decl(attribs)?;
                    // Check for duplicate name with existing type alias
                    if self.type_aliases.contains_key(&decl.name) {
                        return Err(ParseError {
                            span: decl.span,
                            msg: format!(
                                "struct `{}` conflicts with existing type alias",
                                decl.name
                            ),
                        });
                    }
                    self.struct_defs
                        .insert(decl.name.clone(), decl.fields.clone());
                    self.struct_names.insert(decl.name.clone());
                    structs.push(decl);
                }
                Token::Impl => {
                    impls.push(self.parse_impl_decl()?);
                }
                Token::Trait => {
                    traits.push(self.parse_trait_decl()?);
                }
                Token::Type => {
                    let decl = self.parse_type_alias()?;
                    // Check for duplicate name with existing struct
                    if self.struct_names.contains(&decl.name)
                        || funcs.iter().any(|f| f.name == decl.name)
                        || traits.iter().any(|t| t.name == decl.name)
                    {
                        return Err(ParseError {
                            span: decl.span,
                            msg: format!(
                                "type alias `{}` conflicts with existing declaration",
                                decl.name
                            ),
                        });
                    }
                    let params = decl.type_params.clone();
                    let aliased = decl.aliased_type.clone();
                    self.type_aliases
                        .insert(decl.name.clone(), (params, aliased));
                    type_aliases.push(decl);
                }
                _ => {
                    funcs.push(self.parse_function(attribs)?);
                }
            }
        }

        let mut program = Program {
            uses,
            funcs,
            structs,
            enums,
            traits,
            impls,
            type_aliases,
        };

        self.resolve_type_aliases(&mut program)?;

        Ok(program)
    }

    fn parse_attrs(&mut self) -> Result<Vec<Attribute>, ParseError> {
        let mut attrs = Vec::new();
        let default = (Token::Eof, Span::empty(0));
        loop {
            // Check for #[ ... ]
            if self.pos + 1 < self.tokens.len()
                && self.tokens[self.pos].0 == Token::Pound
                && self.tokens[self.pos + 1].0 == Token::LBracket
            {
                let lo = self.current_span_lo();
                self.advance(); // consume #
                self.advance(); // consume [

                // Parse attribute name (identifier)
                let name = match self.peek_token() {
                    Token::Ident(s) => {
                        let s = s.clone();
                        self.advance();
                        s
                    }
                    _ => {
                        let (_, span) = self.current().unwrap_or(&default);
                        return Err(ParseError {
                            span: *span,
                            msg: "expected attribute name after '#['".to_string(),
                        });
                    }
                };

                // Parse optional parenthesized args
                let args = if *self.peek_token() == Token::LParen {
                    self.advance(); // consume (
                    let mut args = Vec::new();
                    if *self.peek_token() != Token::RParen {
                        loop {
                            match self.peek_token() {
                                Token::Ident(s) => {
                                    args.push(s.clone());
                                    self.advance();
                                }
                                _ => {
                                    let (_, span) = self.current().unwrap_or(&default);
                                    return Err(ParseError {
                                        span: *span,
                                        msg: "expected identifier in attribute argument"
                                            .to_string(),
                                    });
                                }
                            }
                            if *self.peek_token() == Token::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                    }
                    self.expect(&Token::RParen)?;
                    args
                } else {
                    Vec::new()
                };

                self.expect(&Token::RBracket)?;
                let hi = self.last_span_end();
                attrs.push(Attribute {
                    name,
                    args,
                    span: Span::new(lo, hi),
                });
            } else {
                break;
            }
        }
        Ok(attrs)
    }

    fn parse_use_decl(&mut self) -> Result<Use, ParseError> {
        let lo = self.current_span_lo();
        self.expect(&Token::Use)?;
        let mut path = Vec::new();
        loop {
            match self.peek_token() {
                Token::Ident(s) => {
                    path.push(s.clone());
                    self.advance();
                }
                _ => {
                    let default = (Token::Eof, Span::empty(0));
                    let (_, span) = self.current().unwrap_or(&default);
                    return Err(ParseError {
                        span: *span,
                        msg: "expected identifier in use path".to_string(),
                    });
                }
            }
            if *self.peek_token() == Token::DoubleColon {
                self.advance();
            } else {
                break;
            }
        }
        if path.len() < 2 {
            let default = (Token::Eof, Span::empty(0));
            let (_, span) = self.current().unwrap_or(&default);
            return Err(ParseError {
                span: *span,
                msg: format!(
                    "use path must have at least 2 segments (e.g. 'std::io'), found {}",
                    path.len()
                ),
            });
        }
        self.expect(&Token::Semicolon)?;
        let hi = self.last_span_end();
        Ok(Use {
            path,
            span: Span::new(lo, hi),
        })
    }

    fn parse_extern_block(&mut self, funcs: &mut Vec<Function>) -> Result<(), ParseError> {
        self.expect(&Token::Extern)?;
        // Expect "C" string literal
        match self.peek_token() {
            Token::StrLit(s) if s == "C" => {
                self.advance();
            }
            _ => {
                let default = (Token::Eof, Span::empty(0));
                let (_, span) = self.current().unwrap_or(&default);
                return Err(ParseError {
                    span: *span,
                    msg: "expected \"C\" after extern".to_string(),
                });
            }
        }
        self.expect(&Token::LBrace)?;
        while *self.peek_token() != Token::RBrace && *self.peek_token() != Token::Eof {
            funcs.push(self.parse_extern_fn()?);
        }
        self.expect(&Token::RBrace)?;
        Ok(())
    }

    fn parse_extern_fn(&mut self) -> Result<Function, ParseError> {
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
                    msg: "expected function name in extern declaration".to_string(),
                });
            }
        };
        self.expect(&Token::LParen)?;
        let params = self.parse_fn_params()?;
        // Check for optional variadic `...`
        let _is_variadic = if *self.peek_token() == Token::Ellipsis {
            self.advance();
            true
        } else {
            false
        };
        self.expect(&Token::RParen)?;
        // Return type is required for extern, but we default to i32 if omitted
        let return_type = if *self.peek_token() == Token::RArrow {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        self.expect(&Token::Semicolon)?;

        // Extern function: no body, mark as extern
        // We store variadic info by using an empty body (special handling in codegen)
        let body = Block {
            stmts: Vec::new(),
            tail_expr: None,
            span: Span::empty(0),
        };
        Ok(Function {
            name,
            params,
            return_type,
            body,
            is_extern: true,
            is_method: false,
            attribs: Vec::new(),
        })
    }

    fn parse_struct_decl(&mut self, attribs: Vec<Attribute>) -> Result<StructDecl, ParseError> {
        let lo = self.current_span_lo();
        self.expect(&Token::Struct)?;
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
                    msg: "expected struct name".to_string(),
                });
            }
        };
        // Parse optional generic type parameters: <T, U, ...>
        let type_params = if *self.peek_token() == Token::Lt {
            self.advance(); // consume <
            let mut params = Vec::new();
            loop {
                match self.peek_token() {
                    Token::Ident(s) => {
                        params.push(s.clone());
                        self.advance();
                    }
                    _ => {
                        let (_, span) = self.current().unwrap();
                        return Err(ParseError {
                            span: *span,
                            msg: "expected type parameter name".to_string(),
                        });
                    }
                }
                if *self.peek_token() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&Token::Gt)?;
            params
        } else {
            Vec::new()
        };
        self.expect(&Token::LBrace)?;
        // ... rest of the fields parsing
        let mut fields = Vec::new();
        while *self.peek_token() != Token::RBrace && *self.peek_token() != Token::Eof {
            let field_lo = self.current_span_lo();
            let field_name = match self.peek_token() {
                Token::Ident(s) => {
                    let s = s.clone();
                    self.advance();
                    s
                }
                _ => {
                    let (_, span) = self.current().unwrap();
                    return Err(ParseError {
                        span: *span,
                        msg: "expected field name".to_string(),
                    });
                }
            };
            self.expect(&Token::Colon)?;
            let ty = self.parse_type()?;
            let field_span = Span::new(field_lo, self.last_span_end());
            fields.push(StructField {
                name: field_name,
                ty,
                span: field_span,
            });
            if *self.peek_token() == Token::Comma {
                self.advance();
            }
        }
        self.expect(&Token::RBrace)?;
        let hi = self.last_span_end();
        Ok(StructDecl {
            name,
            fields,
            type_params,
            attribs,
            span: Span::new(lo, hi),
        })
    }

    fn parse_enum_decl(&mut self, attribs: Vec<Attribute>) -> Result<EnumDecl, ParseError> {
        let lo = self.current_span_lo();
        self.expect(&Token::Enum)?;
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
                    msg: "expected enum name".to_string(),
                });
            }
        };
        // Parse optional generic type parameters: <T, U, ...>
        let type_params = if *self.peek_token() == Token::Lt {
            self.advance(); // consume <
            let mut params = Vec::new();
            loop {
                match self.peek_token() {
                    Token::Ident(s) => {
                        params.push(s.clone());
                        self.advance();
                    }
                    _ => {
                        let (_, span) = self.current().unwrap();
                        return Err(ParseError {
                            span: *span,
                            msg: "expected type parameter name".to_string(),
                        });
                    }
                }
                if *self.peek_token() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&Token::Gt)?;
            params
        } else {
            Vec::new()
        };
        self.expect(&Token::LBrace)?;
        let mut variants = Vec::new();
        while *self.peek_token() != Token::RBrace && *self.peek_token() != Token::Eof {
            let var_lo = self.current_span_lo();
            let var_name = match self.peek_token() {
                Token::Ident(s) => {
                    let s = s.clone();
                    self.advance();
                    s
                }
                _ => {
                    let (_, span) = self.current().unwrap();
                    return Err(ParseError {
                        span: *span,
                        msg: "expected variant name".to_string(),
                    });
                }
            };
            // Check for optional payload type: VarName(Type) or VarName
            let ty = if *self.peek_token() == Token::LParen {
                self.advance(); // consume (
                let t = self.parse_type()?;
                self.expect(&Token::RParen)?;
                Some(t)
            } else {
                None
            };
            let var_span = Span::new(var_lo, self.last_span_end());
            variants.push(EnumVariant {
                name: var_name,
                ty,
                span: var_span,
            });
            if *self.peek_token() == Token::Comma {
                self.advance();
            }
        }
        self.expect(&Token::RBrace)?;
        let hi = self.last_span_end();
        Ok(EnumDecl {
            name,
            variants,
            type_params,
            attribs,
            span: Span::new(lo, hi),
        })
    }

    fn parse_type_alias(&mut self) -> Result<TypeAliasDecl, ParseError> {
        let lo = self.current_span_lo();
        self.expect(&Token::Type)?;
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
                    msg: "expected type alias name after 'type'".to_string(),
                });
            }
        };
        // Parse optional generic type parameters: <T, U, ...>
        let type_params = if *self.peek_token() == Token::Lt {
            self.advance(); // consume <
            let mut params = Vec::new();
            loop {
                match self.peek_token() {
                    Token::Ident(s) => {
                        params.push(s.clone());
                        self.advance();
                    }
                    _ => {
                        let (_, span) = self.current().unwrap();
                        return Err(ParseError {
                            span: *span,
                            msg: "expected type parameter name".to_string(),
                        });
                    }
                }
                if *self.peek_token() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&Token::Gt)?;
            params
        } else {
            Vec::new()
        };
        self.expect(&Token::Eq)?;
        let aliased_type = self.parse_type()?;
        self.expect(&Token::Semicolon)?;
        let hi = self.last_span_end();
        Ok(TypeAliasDecl {
            name,
            type_params,
            aliased_type,
            span: Span::new(lo, hi),
        })
    }

    fn parse_impl_decl(&mut self) -> Result<ImplDecl, ParseError> {
        self.expect(&Token::Impl)?;
        // Parse optional generic type parameters: <T, U, ...>
        let type_params = if *self.peek_token() == Token::Lt {
            self.advance(); // consume <
            let mut params = Vec::new();
            loop {
                match self.peek_token() {
                    Token::Ident(s) => {
                        params.push(s.clone());
                        self.advance();
                    }
                    _ => {
                        let (_, span) = self.current().unwrap();
                        return Err(ParseError {
                            span: *span,
                            msg: "expected type parameter name".to_string(),
                        });
                    }
                }
                if *self.peek_token() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&Token::Gt)?;
            params
        } else {
            Vec::new()
        };
        // Check for `impl Trait for Type` or `impl Type`
        let trait_name = if self.pos + 2 < self.tokens.len() {
            let is_ident_for = matches!(&self.tokens[self.pos].0, Token::Ident(_))
                && self.tokens[self.pos + 1].0 == Token::For;
            if is_ident_for {
                let name = match self.peek_token() {
                    Token::Ident(s) => {
                        let s = s.clone();
                        self.advance();
                        s
                    }
                    _ => unreachable!(),
                };
                self.expect(&Token::For)?;
                Some(name)
            } else {
                None
            }
        } else {
            None
        };
        let impl_type = self.parse_type()?;
        self.expect(&Token::LBrace)?;
        let mut methods = Vec::new();
        while *self.peek_token() != Token::RBrace && *self.peek_token() != Token::Eof {
            let mut func = self.parse_function(Vec::new())?;
            func.is_method = true;
            methods.push(func);
        }
        self.expect(&Token::RBrace)?;
        Ok(ImplDecl {
            impl_type,
            trait_name,
            type_params,
            methods,
        })
    }

    fn parse_trait_decl(&mut self) -> Result<TraitDecl, ParseError> {
        let lo = self.current_span_lo();
        self.expect(&Token::Trait)?;
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
                    msg: "expected trait name".to_string(),
                });
            }
        };
        self.expect(&Token::LBrace)?;
        let mut methods = Vec::new();
        while *self.peek_token() != Token::RBrace && *self.peek_token() != Token::Eof {
            self.expect(&Token::Fn)?;
            let method_name = match self.peek_token() {
                Token::Ident(s) => {
                    let s = s.clone();
                    self.advance();
                    s
                }
                _ => {
                    let (_, span) = self.current().unwrap();
                    return Err(ParseError {
                        span: *span,
                        msg: "expected trait method name".to_string(),
                    });
                }
            };
            self.expect(&Token::LParen)?;
            let params = self.parse_fn_params()?;
            self.expect(&Token::RParen)?;
            let return_type = if *self.peek_token() == Token::RArrow {
                self.advance();
                Some(self.parse_type()?)
            } else {
                None
            };
            self.expect(&Token::Semicolon)?;
            methods.push(TraitMethodDef {
                name: method_name,
                params,
                return_type,
            });
        }
        self.expect(&Token::RBrace)?;
        let hi = self.last_span_end();
        Ok(TraitDecl {
            name,
            methods,
            span: Span::new(lo, hi),
        })
    }

    fn parse_function(&mut self, attribs: Vec<Attribute>) -> Result<Function, ParseError> {
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
        let params = self.parse_fn_params()?;
        self.expect(&Token::RParen)?;
        // Optional return type
        let return_type = if *self.peek_token() == Token::RArrow {
            self.advance();
            Some(self.parse_type()?)
        } else {
            None
        };
        let body = self.parse_block()?;
        Ok(Function {
            name,
            params,
            return_type,
            body,
            is_extern: false,
            is_method: false,
            attribs,
        })
    }

    /// Parse a comma-separated list of `name : type` parameters.
    /// Stops on `)` or `...`.
    fn parse_fn_params(&mut self) -> Result<Vec<Param>, ParseError> {
        let mut params = Vec::new();
        if *self.peek_token() == Token::RParen || *self.peek_token() == Token::Ellipsis {
            return Ok(params);
        }
        loop {
            if *self.peek_token() == Token::Ellipsis || *self.peek_token() == Token::RParen {
                break;
            }
            // Handle `&self` and `&mut self` shorthand
            if *self.peek_token() == Token::Ampersand && params.is_empty() {
                let lo = self.current_span_lo();
                self.advance(); // consume &
                let is_mut = if *self.peek_token() == Token::Mut {
                    self.advance();
                    true
                } else {
                    false
                };
                if *self.peek_token() == Token::Self_ {
                    self.advance();
                    let ty = Type::Ref {
                        inner: Box::new(Type::SelfType),
                        is_mut,
                    };
                    let param_span = Span::new(lo, self.last_span_end());
                    params.push(Param {
                        name: "self".to_string(),
                        ty,
                        span: param_span,
                    });
                    if *self.peek_token() == Token::Comma {
                        self.advance();
                    } else {
                        break;
                    }
                    continue;
                }
            }
            let lo = self.current_span_lo();
            let name = match self.peek_token() {
                Token::Ident(s) => {
                    let s = s.clone();
                    self.advance();
                    s
                }
                Token::Self_ => {
                    self.advance();
                    "self".to_string()
                }
                _ => {
                    let (_, span) = self.current().unwrap();
                    return Err(ParseError {
                        span: *span,
                        msg: "expected parameter name".to_string(),
                    });
                }
            };
            self.expect(&Token::Colon)?;
            let ty = self.parse_type()?;
            let param_span = Span::new(lo, self.last_span_end());
            params.push(Param {
                name,
                ty,
                span: param_span,
            });
            if *self.peek_token() == Token::Comma {
                self.advance();
            } else {
                break;
            }
        }
        Ok(params)
    }

    fn parse_block(&mut self) -> Result<Block, ParseError> {
        let lo = match self.current() {
            Some((_, span)) => span.lo,
            None => self.last_span_end(),
        };
        self.expect(&Token::LBrace)?;
        let mut stmts = Vec::new();
        let mut tail_expr = None;
        loop {
            if *self.peek_token() == Token::RBrace || *self.peek_token() == Token::Eof {
                break;
            }
            // let / const statements are always statements terminated by ;
            if matches!(self.peek_token(), Token::Let | Token::Const | Token::Return) {
                stmts.push(self.parse_stmt()?);
                continue;
            }
            // Parse an expression — could be a statement (; terminated) or tail expression
            let expr = self.parse_expr()?;
            if *self.peek_token() == Token::Semicolon {
                self.advance();
                stmts.push(Stmt::Expr(expr));
            } else {
                // No semicolon — tail expression
                tail_expr = Some(Box::new(expr));
                break;
            }
        }
        let hi = match self.current() {
            Some((_, span)) => span.hi,
            None => self.last_span_end(),
        };
        self.expect(&Token::RBrace)?;
        Ok(Block {
            stmts,
            tail_expr,
            span: Span::new(lo, hi),
        })
    }

    fn parse_stmt(&mut self) -> Result<Stmt, ParseError> {
        match self.peek_token() {
            Token::Let => {
                let lo = self.current_span_lo();
                self.advance();
                let is_mut = if *self.peek_token() == Token::Mut {
                    self.advance();
                    true
                } else {
                    false
                };
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
                // Optional type annotation
                let type_ann = if *self.peek_token() == Token::Colon {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };
                self.expect(&Token::Eq)?;
                let init = self.parse_expr()?;
                self.expect(&Token::Semicolon)?;
                let hi = self.last_span_end();
                Ok(Stmt::Let {
                    name,
                    is_mut,
                    type_ann,
                    init,
                    span: Span::new(lo, hi),
                })
            }
            Token::Return => {
                let lo = self.current_span_lo();
                self.advance();
                let value = if *self.peek_token() == Token::Semicolon {
                    None
                } else {
                    Some(Box::new(self.parse_expr()?))
                };
                self.expect(&Token::Semicolon)?;
                let hi = self.last_span_end();
                Ok(Stmt::Return {
                    value,
                    span: Span::new(lo, hi),
                })
            }
            Token::Const => {
                let lo = self.current_span_lo();
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
                            msg: "expected constant name after 'const'".to_string(),
                        });
                    }
                };
                // Optional type annotation
                let type_ann = if *self.peek_token() == Token::Colon {
                    self.advance();
                    Some(self.parse_type()?)
                } else {
                    None
                };
                self.expect(&Token::Eq)?;
                let init = self.parse_expr()?;
                self.expect(&Token::Semicolon)?;
                let hi = self.last_span_end();
                Ok(Stmt::Const {
                    name,
                    type_ann,
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
        self.parse_assign()
    }

    fn parse_assign(&mut self) -> Result<Expr, ParseError> {
        let lhs = self.parse_comparison()?;
        if *self.peek_token() == Token::Eq {
            match &lhs {
                Expr::Ident(_) | Expr::Deref(_) | Expr::Member { .. } => {
                    self.advance(); // consume '='
                    let value = self.parse_assign()?; // right-associative
                    Ok(Expr::Assign {
                        target: Box::new(lhs),
                        value: Box::new(value),
                    })
                }
                _ => {
                    let default = (Token::Eof, Span::empty(0));
                    let (_, span) = self.current().unwrap_or(&default);
                    Err(ParseError {
                        span: *span,
                        msg:
                            "left-hand side of assignment must be a variable, dereference, or field"
                                .to_string(),
                    })
                }
            }
        } else {
            Ok(lhs)
        }
    }

    fn parse_comparison(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_additive()?;
        loop {
            let op = match self.peek_token() {
                Token::EqEq => BinOp::Eq,
                Token::BangEq => BinOp::Neq,
                Token::Lt => BinOp::Lt,
                Token::Gt => BinOp::Gt,
                Token::Le => BinOp::Le,
                Token::Ge => BinOp::Ge,
                _ => break,
            };
            self.advance();
            let rhs = self.parse_additive()?;
            lhs = Expr::Binary {
                op,
                lhs: Box::new(lhs),
                rhs: Box::new(rhs),
            };
        }
        Ok(lhs)
    }

    fn parse_additive(&mut self) -> Result<Expr, ParseError> {
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

    /// Parse a primary expression followed by postfix operators:
    /// - `.N` tuple member access
    /// - `.ident(args)` method call
    ///
    /// This has the highest precedence.
    fn parse_postfix(&mut self) -> Result<Expr, ParseError> {
        let mut expr = self.parse_primary()?;
        while *self.peek_token() == Token::Dot {
            self.advance(); // consume '.'
            match self.peek_token() {
                Token::IntLit(idx) => {
                    let idx = *idx;
                    if idx < 0 {
                        let default = (Token::Eof, Span::empty(0));
                        let (_, span) = self.current().unwrap_or(&default);
                        return Err(ParseError {
                            span: *span,
                            msg: "tuple member index must be non-negative".to_string(),
                        });
                    }
                    self.advance();
                    expr = Expr::Member {
                        expr: Box::new(expr),
                        index: idx as usize,
                        field: None,
                    };
                }
                Token::Ident(name) => {
                    let field = name.clone();
                    self.advance(); // consume field/method name
                    if *self.peek_token() == Token::LParen {
                        let args = self.parse_call_args()?;
                        expr = Expr::MethodCall {
                            expr: Box::new(expr),
                            method: field,
                            args,
                        };
                    } else {
                        // Named field access: resolve field index here
                        // or defer to codegen with the field name
                        expr = Expr::Member {
                            expr: Box::new(expr),
                            index: 0,
                            field: Some(field),
                        };
                    }
                }
                _ => {
                    let default = (Token::Eof, Span::empty(0));
                    let (_, span) = self.current().unwrap_or(&default);
                    return Err(ParseError {
                        span: *span,
                        msg: "expected tuple member index or method name after '.'".to_string(),
                    });
                }
            }
        }
        Ok(expr)
    }

    /// Parse a primary expression followed by zero or more `as Type` casts.
    /// This binds tighter than binary operators: `a * b as i32` → `a * (b as i32)`
    fn parse_primary_as(&mut self) -> Result<Expr, ParseError> {
        // Handle prefix `!` operator (after postfix ops, so `!x.f` = `!(x.f)`)
        if *self.peek_token() == Token::Bang {
            self.advance(); // consume !
            let expr = self.parse_primary_as()?;
            return Ok(Expr::UnaryNot(Box::new(expr)));
        }
        let mut expr = self.parse_postfix()?;
        while *self.peek_token() == Token::As {
            self.advance();
            let to_type = self.parse_type()?;
            expr = Expr::Cast {
                expr: Box::new(expr),
                to_type,
            };
        }
        Ok(expr)
    }

    fn parse_term(&mut self) -> Result<Expr, ParseError> {
        let mut lhs = self.parse_primary_as()?;
        loop {
            match self.peek_token() {
                Token::Star => {
                    self.advance();
                    let rhs = self.parse_primary_as()?;
                    lhs = Expr::Binary {
                        op: BinOp::Mul,
                        lhs: Box::new(lhs),
                        rhs: Box::new(rhs),
                    };
                }
                Token::Slash => {
                    self.advance();
                    let rhs = self.parse_primary_as()?;
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

    fn parse_type(&mut self) -> Result<Type, ParseError> {
        // Handle reference types: &Type, &mut Type
        if *self.peek_token() == Token::Ampersand {
            self.advance();
            let is_mut = if *self.peek_token() == Token::Mut {
                self.advance();
                true
            } else {
                false
            };
            let inner = self.parse_type()?;
            return Ok(Type::Ref {
                inner: Box::new(inner),
                is_mut,
            });
        }
        // Handle pointer types: *const Type, *mut Type
        if *self.peek_token() == Token::Star {
            self.advance();
            let is_mut = if *self.peek_token() == Token::Mut {
                self.advance();
                true
            } else if *self.peek_token() == Token::Const {
                self.advance();
                false
            } else {
                let default = (Token::Eof, Span::empty(0));
                let (_, span) = self.current().unwrap_or(&default);
                return Err(ParseError {
                    span: *span,
                    msg: "expected 'const' or 'mut' after '*' in pointer type".to_string(),
                });
            };
            let inner = self.parse_type()?;
            return Ok(Type::Ptr {
                inner: Box::new(inner),
                is_mut,
            });
        }
        // Handle tuple / unit types: (Type1, Type2) or ()
        if *self.peek_token() == Token::LParen {
            return self.parse_tuple_type();
        }
        let token = self.peek_token().clone();
        self.advance();
        match token {
            Token::I8 => Ok(Type::I8),
            Token::I16 => Ok(Type::I16),
            Token::I32 => Ok(Type::I32),
            Token::I64 => Ok(Type::I64),
            Token::U8 => Ok(Type::U8),
            Token::U16 => Ok(Type::U16),
            Token::U32 => Ok(Type::U32),
            Token::U64 => Ok(Type::U64),
            Token::Usize => Ok(Type::Usize),
            Token::Isize => Ok(Type::Isize),
            Token::F32 => Ok(Type::F32),
            Token::F64 => Ok(Type::F64),
            Token::Bool => Ok(Type::Bool),
            Token::Str => Ok(Type::Str),
            Token::SelfType => Ok(Type::SelfType),
            Token::Ident(name) => {
                // Check for generic args: name < type, type, ... >
                if *self.peek_token() == Token::Lt {
                    if self.type_aliases.contains_key(&name) {
                        self.advance(); // consume <
                        let mut args = Vec::new();
                        loop {
                            args.push(self.parse_type()?);
                            if *self.peek_token() == Token::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        self.expect(&Token::Gt)?;
                        return Ok(Type::Alias(name, args));
                    } else {
                        // Assume generic struct (or will be resolved later via stdlib imports)
                        self.advance(); // consume <
                        let mut args = Vec::new();
                        loop {
                            args.push(self.parse_type()?);
                            if *self.peek_token() == Token::Comma {
                                self.advance();
                            } else {
                                break;
                            }
                        }
                        self.expect(&Token::Gt)?;
                        return Ok(Type::GenericInstance(name, args));
                    }
                }
                Ok(Type::Struct(name))
            }
            Token::Bang => Ok(Type::Never),
            _ => {
                let default = (Token::Eof, Span::empty(0));
                let (_, span) = self.current().unwrap_or(&default);
                Err(ParseError {
                    span: *span,
                    msg: format!("expected type, found {:?}", token),
                })
            }
        }
    }

    fn parse_tuple_type(&mut self) -> Result<Type, ParseError> {
        self.expect(&Token::LParen)?;
        // Empty parens → Unit
        if *self.peek_token() == Token::RParen {
            self.advance();
            return Ok(Type::Unit);
        }
        let first = self.parse_type()?;
        // If comma follows → it's a tuple
        if *self.peek_token() == Token::Comma {
            self.advance();
            let mut types = vec![first];
            while *self.peek_token() != Token::RParen {
                types.push(self.parse_type()?);
                if *self.peek_token() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
            self.expect(&Token::RParen)?;
            return Ok(Type::Tuple(types));
        }
        // (Type) — parenthesized type, not a tuple
        self.expect(&Token::RParen)?;
        Ok(first)
    }

    fn token_to_type(token: &Token) -> Option<Type> {
        match token {
            Token::Bool => Some(Type::Bool),
            Token::I8 => Some(Type::I8),
            Token::I16 => Some(Type::I16),
            Token::I32 => Some(Type::I32),
            Token::I64 => Some(Type::I64),
            Token::U8 => Some(Type::U8),
            Token::U16 => Some(Type::U16),
            Token::U32 => Some(Type::U32),
            Token::U64 => Some(Type::U64),
            Token::Usize => Some(Type::Usize),
            Token::Isize => Some(Type::Isize),
            Token::F32 => Some(Type::F32),
            Token::F64 => Some(Type::F64),
            Token::Str => Some(Type::Str),
            _ => None,
        }
    }

    fn parse_call_args(&mut self) -> Result<Vec<Expr>, ParseError> {
        self.expect(&Token::LParen)?;
        let mut args = Vec::new();
        if *self.peek_token() != Token::RParen {
            loop {
                args.push(self.parse_expr()?);
                if *self.peek_token() == Token::Comma {
                    self.advance();
                } else {
                    break;
                }
            }
        }
        self.expect(&Token::RParen)?;
        Ok(args)
    }

    fn parse_struct_lit(&mut self, struct_name: &str) -> Result<Expr, ParseError> {
        self.expect(&Token::LBrace)?;
        let mut fields = Vec::new();
        while *self.peek_token() != Token::RBrace && *self.peek_token() != Token::Eof {
            let field_name = match self.peek_token() {
                Token::Ident(s) => {
                    let s = s.clone();
                    self.advance();
                    s
                }
                _ => {
                    let (_, span) = self.current().unwrap();
                    return Err(ParseError {
                        span: *span,
                        msg: "expected field name in struct literal".to_string(),
                    });
                }
            };
            if *self.peek_token() == Token::Colon {
                self.advance();
                let value = self.parse_expr()?;
                fields.push((field_name, value));
            } else {
                // Shorthand: field_name expands to field_name: field_name
                fields.push((field_name.clone(), Expr::Ident(field_name)));
            }
            if *self.peek_token() == Token::Comma {
                self.advance();
            }
        }
        self.expect(&Token::RBrace)?;
        Ok(Expr::StructLit {
            struct_name: struct_name.to_string(),
            fields,
        })
    }

    fn parse_primary(&mut self) -> Result<Expr, ParseError> {
        // Handle prefix &, &mut, and * operators
        if *self.peek_token() == Token::Ampersand {
            self.advance(); // consume &
            let is_mut = if *self.peek_token() == Token::Mut {
                self.advance();
                true
            } else {
                false
            };
            let expr = self.parse_primary()?;
            return Ok(Expr::Ref {
                expr: Box::new(expr),
                is_mut,
            });
        }
        if *self.peek_token() == Token::Star {
            self.advance(); // consume *
            let expr = self.parse_primary()?;
            return Ok(Expr::Deref(Box::new(expr)));
        }
        match self.peek_token() {
            Token::IntLit(val) => {
                let val = *val;
                self.advance();
                Ok(Expr::IntLit(val))
            }
            Token::IntSuffixLit(val, suffix) => {
                let val = *val;
                let to_type = Self::token_to_type(suffix).ok_or_else(|| {
                    let default = (Token::Eof, Span::empty(0));
                    let (_, span) = self.current().unwrap_or(&default);
                    ParseError {
                        span: *span,
                        msg: format!("invalid type suffix {:?}", suffix),
                    }
                })?;
                self.advance();
                Ok(Expr::Cast {
                    expr: Box::new(Expr::IntLit(val)),
                    to_type,
                })
            }
            Token::FloatLit(val) => {
                let val = *val;
                self.advance();
                Ok(Expr::FloatLit(val))
            }
            Token::FloatSuffixLit(val, suffix) => {
                let val = *val;
                let to_type = Self::token_to_type(suffix).ok_or_else(|| {
                    let default = (Token::Eof, Span::empty(0));
                    let (_, span) = self.current().unwrap_or(&default);
                    ParseError {
                        span: *span,
                        msg: format!("invalid type suffix {:?}", suffix),
                    }
                })?;
                self.advance();
                Ok(Expr::Cast {
                    expr: Box::new(Expr::FloatLit(val)),
                    to_type,
                })
            }
            Token::StrLit(s) => {
                let s = s.clone();
                self.advance();
                Ok(Expr::StrLit(s))
            }
            Token::True => {
                self.advance();
                Ok(Expr::BoolLit(true))
            }
            Token::False => {
                self.advance();
                Ok(Expr::BoolLit(false))
            }
            Token::Self_ => {
                self.advance();
                Ok(Expr::Ident("self".to_string()))
            }
            Token::Ident(name) => {
                let name = name.clone();
                self.advance();
                // Check for struct literal: Name { field: expr, ... }
                if *self.peek_token() == Token::LBrace {
                    return self.parse_struct_lit(&name);
                }
                // Check for qualified path: ident :: ident
                if *self.peek_token() == Token::DoubleColon {
                    self.advance(); // consume ::
                    let variant_or_fn = match self.peek_token() {
                        Token::Ident(s) => {
                            let s = s.clone();
                            self.advance();
                            s
                        }
                        _ => {
                            let (_, span) = self.current().unwrap();
                            return Err(ParseError {
                                span: *span,
                                msg: "expected identifier after '::'".to_string(),
                            });
                        }
                    };
                    // If followed by `(`, check if it's a known enum (enum literal) or not (qualified call)
                    if *self.peek_token() == Token::LParen {
                        if self.enum_names.contains(&name) {
                            self.advance(); // consume '('
                            if *self.peek_token() == Token::RParen {
                                // Empty parens → unit variant with no payload
                                self.advance(); // consume ')'
                                return Ok(Expr::EnumLit {
                                    enum_name: name,
                                    variant: variant_or_fn,
                                    payload: None,
                                });
                            }
                            let inner = self.parse_expr()?;
                            self.expect(&Token::RParen)?;
                            return Ok(Expr::EnumLit {
                                enum_name: name,
                                variant: variant_or_fn,
                                payload: Some(Box::new(inner)),
                            });
                        } else {
                            let args = self.parse_call_args()?;
                            return Ok(Expr::QualifiedCall {
                                module: name,
                                callee: variant_or_fn,
                                args,
                            });
                        }
                    } else {
                        // No parens → unit enum variant
                        return Ok(Expr::EnumLit {
                            enum_name: name,
                            variant: variant_or_fn,
                            payload: None,
                        });
                    }
                }
                // Check for function call: ident ( ... )
                if *self.peek_token() == Token::LParen {
                    let args = self.parse_call_args()?;
                    return Ok(Expr::Call { callee: name, args });
                }
                Ok(Expr::Ident(name))
            }
            Token::LParen => {
                self.advance();
                // Check for empty parens → unit
                if *self.peek_token() == Token::RParen {
                    self.advance();
                    return Ok(Expr::Unit);
                }
                let first = self.parse_expr()?;
                // Comma → tuple
                if *self.peek_token() == Token::Comma {
                    self.advance();
                    let mut exprs = vec![first];
                    while *self.peek_token() != Token::RParen {
                        exprs.push(self.parse_expr()?);
                        if *self.peek_token() == Token::Comma {
                            self.advance();
                        } else {
                            break;
                        }
                    }
                    self.expect(&Token::RParen)?;
                    return Ok(Expr::Tuple(exprs));
                }
                // (expr) — parenthesized expression
                self.expect(&Token::RParen)?;
                Ok(first)
            }
            Token::If => self.parse_if_expr(),
            Token::Loop => self.parse_loop_expr(),
            Token::While => self.parse_while_expr(),
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

    fn parse_if_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect(&Token::If)?;
        let cond = self.parse_expr()?;
        let then_block = self.parse_block()?;
        let mut else_ifs = Vec::new();
        let mut else_block = None;

        // Parse optional else clause
        if *self.peek_token() == Token::Else {
            self.advance();
            if *self.peek_token() == Token::If {
                // Recursively parse else if as a nested if expression
                let elif = self.parse_if_expr()?;
                // Extract the components from the nested if
                if let Expr::If {
                    cond: elif_cond,
                    then_block: elif_then,
                    else_ifs: elif_else_ifs,
                    else_block: elif_else,
                } = elif
                {
                    else_ifs.push((*elif_cond, elif_then));
                    else_ifs.extend(elif_else_ifs);
                    else_block = elif_else;
                }
            } else {
                else_block = Some(self.parse_block()?);
            }
        }
        Ok(Expr::If {
            cond: Box::new(cond),
            then_block,
            else_ifs,
            else_block,
        })
    }

    fn parse_loop_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect(&Token::Loop)?;
        let body = self.parse_block()?;
        Ok(Expr::Loop { body })
    }

    fn parse_while_expr(&mut self) -> Result<Expr, ParseError> {
        self.expect(&Token::While)?;
        let cond = self.parse_expr()?;
        let body = self.parse_block()?;
        Ok(Expr::While {
            cond: Box::new(cond),
            body,
        })
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

    /// Walk the entire program AST and resolve all type alias references.
    /// Replaces Type::Struct(name) where `name` is a simple alias, and
    /// Type::Alias(name, args) where `name` is a generic alias.
    fn resolve_type_aliases(&self, program: &mut Program) -> Result<(), ParseError> {
        // Clone alias data to avoid borrow conflicts
        let alias_defs: HashMap<String, TypeAliasDecl> = program
            .type_aliases
            .iter()
            .map(|d| (d.name.clone(), d.clone()))
            .collect();

        for func in &mut program.funcs {
            for param in &mut func.params {
                self.resolve_type_in_place_owned(&mut param.ty, &alias_defs, &mut HashSet::new())?;
            }
            if let Some(ref mut ret) = func.return_type {
                self.resolve_type_in_place_owned(ret, &alias_defs, &mut HashSet::new())?;
            }
            // Resolve types in the function body (let type annotations, casts, etc.)
            self.resolve_types_in_block(&mut func.body, &alias_defs)?;
        }
        for s in &mut program.structs {
            for field in &mut s.fields {
                self.resolve_type_in_place_owned(&mut field.ty, &alias_defs, &mut HashSet::new())?;
            }
        }
        for t in &mut program.traits {
            for method in &mut t.methods {
                for param in &mut method.params {
                    self.resolve_type_in_place_owned(
                        &mut param.ty,
                        &alias_defs,
                        &mut HashSet::new(),
                    )?;
                }
                if let Some(ref mut ret) = method.return_type {
                    self.resolve_type_in_place_owned(ret, &alias_defs, &mut HashSet::new())?;
                }
            }
        }
        for i in &mut program.impls {
            self.resolve_type_in_place_owned(&mut i.impl_type, &alias_defs, &mut HashSet::new())?;
            for method in &mut i.methods {
                for param in &mut method.params {
                    self.resolve_type_in_place_owned(
                        &mut param.ty,
                        &alias_defs,
                        &mut HashSet::new(),
                    )?;
                }
                if let Some(ref mut ret) = method.return_type {
                    self.resolve_type_in_place_owned(ret, &alias_defs, &mut HashSet::new())?;
                }
            }
        }
        // Also resolve within the alias definitions themselves (for chained aliases)
        for alias in &mut program.type_aliases {
            self.resolve_type_in_place_owned(
                &mut alias.aliased_type,
                &alias_defs,
                &mut HashSet::new(),
            )?;
        }

        Ok(())
    }

    /// Recursively resolve a type in place, substituting aliases.
    fn resolve_type_in_place_owned(
        &self,
        ty: &mut Type,
        aliases: &HashMap<String, TypeAliasDecl>,
        visiting: &mut HashSet<String>,
    ) -> Result<(), ParseError> {
        match ty {
            Type::Struct(name) => {
                // Check if this is a simple (non-generic) alias reference
                if let Some(alias) = aliases.get(name) {
                    if !alias.type_params.is_empty() {
                        return Ok(()); // generic alias requires Type::Alias form, not plain Struct
                    }
                    // Cycle detection
                    if !visiting.insert(name.clone()) {
                        return Err(ParseError {
                            span: alias.span,
                            msg: format!("cyclic type alias `{}`", name),
                        });
                    }
                    // Substitute the aliased type and continue resolving
                    let subst = alias.aliased_type.clone();
                    *ty = subst;
                    // Recurse into the substituted type
                    return self.resolve_type_in_place_owned(ty, aliases, visiting);
                }
                Ok(())
            }
            Type::Alias(name, args) => {
                // Look up the generic alias
                if let Some(alias) = aliases.get(name) {
                    if args.len() != alias.type_params.len() {
                        return Err(ParseError {
                            span: alias.span,
                            msg: format!(
                                "type alias `{}` expects {} type arguments, got {}",
                                name,
                                alias.type_params.len(),
                                args.len()
                            ),
                        });
                    }
                    // Build substitution map: param_name -> concrete type arg
                    // Substitute type params in the aliased type with concrete args
                    let mut substituted = alias.aliased_type.clone();
                    Self::substitute_type_params(&mut substituted, &alias.type_params, args);
                    // Check for cycles
                    if !visiting.insert(name.clone()) {
                        return Err(ParseError {
                            span: alias.span,
                            msg: format!("cyclic type alias `{}`", name),
                        });
                    }
                    *ty = substituted;
                    // Recurse into the substituted type
                    return self.resolve_type_in_place_owned(ty, aliases, visiting);
                }
                // Unknown alias — let codegen produce the error
                Ok(())
            }
            Type::GenericInstance(_name, args) => {
                // Resolve type args within the generic instance
                for arg in args.iter_mut() {
                    self.resolve_type_in_place_owned(arg, aliases, visiting)?;
                }
                Ok(())
            }
            Type::Ref { inner, .. } | Type::Ptr { inner, .. } => {
                self.resolve_type_in_place_owned(inner, aliases, visiting)
            }
            Type::Tuple(types) => {
                for t in types.iter_mut() {
                    self.resolve_type_in_place_owned(t, aliases, visiting)?;
                }
                Ok(())
            }
            _ => Ok(()),
        }
    }

    /// Substitute type parameter references in a type with concrete type arguments.
    pub fn substitute_type_params(ty: &mut Type, params: &[String], args: &[Type]) {
        match ty {
            Type::Struct(name) => {
                if let Some(pos) = params.iter().position(|p| p == name)
                    && let Some(arg) = args.get(pos)
                {
                    *ty = arg.clone();
                }
            }
            Type::GenericInstance(_name, type_args) => {
                for arg in type_args.iter_mut() {
                    Self::substitute_type_params(arg, params, args);
                }
                // If the base name itself is a type param, it would have been caught by Type::Struct above
            }
            Type::Ref { inner, .. } | Type::Ptr { inner, .. } => {
                Self::substitute_type_params(inner, params, args);
            }
            Type::Tuple(types) => {
                for t in types.iter_mut() {
                    Self::substitute_type_params(t, params, args);
                }
            }
            _ => {}
        }
    }

    /// Resolve types within a block body: type annotations, cast types,
    /// and recurse into nested blocks (if, loop, while).
    fn resolve_types_in_block(
        &self,
        body: &mut Block,
        aliases: &HashMap<String, TypeAliasDecl>,
    ) -> Result<(), ParseError> {
        for stmt in &mut body.stmts {
            match stmt {
                Stmt::Let {
                    type_ann: Some(ty), ..
                } => {
                    self.resolve_type_in_place_owned(ty, aliases, &mut HashSet::new())?;
                }
                Stmt::Const {
                    type_ann: Some(ty), ..
                } => {
                    self.resolve_type_in_place_owned(ty, aliases, &mut HashSet::new())?;
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn current_span_lo(&self) -> usize {
        match self.current() {
            Some((_, span)) => span.lo,
            None => self.last_span_end(),
        }
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
        assert!(prog.uses.is_empty());
    }

    #[test]
    fn test_single_function_no_body() {
        let prog = parse("fn foo() {}").unwrap();
        assert_eq!(prog.funcs.len(), 1);
        assert_eq!(prog.funcs[0].name, "foo");
        assert!(prog.funcs[0].body.stmts.is_empty());
        assert!(!prog.funcs[0].is_extern);
        assert!(prog.funcs[0].params.is_empty());
        assert!(prog.funcs[0].return_type.is_none());
    }

    #[test]
    fn test_let_mut() {
        let prog = parse("fn main() { let mut x = 42; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                name,
                is_mut: true,
                init,
                ..
            } => {
                assert_eq!(name, "x");
                assert!(matches!(init, Expr::IntLit(42)));
            }
            _ => panic!("expected Let with is_mut=true"),
        }
    }

    #[test]
    fn test_let_immutable_default() {
        let prog = parse("fn main() { let x = 99; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { is_mut: false, .. } => {}
            _ => panic!("expected Let with is_mut=false"),
        }
    }

    #[test]
    fn test_const_declaration() {
        let prog = parse("fn main() { const X = 100; }").unwrap();
        assert_eq!(prog.funcs[0].body.stmts.len(), 1);
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Const { name, init, .. } => {
                assert_eq!(name, "X");
                assert!(matches!(init, Expr::IntLit(100)));
            }
            _ => panic!("expected Const stmt"),
        }
    }

    #[test]
    fn test_assignment_expression() {
        let prog = parse("fn main() { let mut x = 1; x = 2; }").unwrap();
        let stmts = &prog.funcs[0].body.stmts;
        assert_eq!(stmts.len(), 2);
        match &stmts[1] {
            Stmt::Expr(Expr::Assign { target, value }) => {
                assert!(matches!(target.as_ref(), Expr::Ident(name) if name == "x"));
                assert!(matches!(value.as_ref(), Expr::IntLit(2)));
            }
            _ => panic!("expected Expr(Assign(x, 2))"),
        }
    }

    #[test]
    fn test_assignment_non_variable() {
        let result = parse("fn f() { 3 = 4; }");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.msg.contains("left-hand side"),
                "error should mention left-hand side: {}",
                e.msg
            );
        }
    }

    #[test]
    fn test_ref_expr() {
        let prog = parse("fn main() { let x = 42; let r = &x; }").unwrap();
        match &prog.funcs[0].body.stmts[1] {
            Stmt::Let { init, .. } => match init {
                Expr::Ref {
                    expr,
                    is_mut: false,
                } => {
                    assert!(matches!(expr.as_ref(), Expr::Ident(name) if name == "x"));
                }
                _ => panic!("expected Ref {{ expr: Ident, is_mut: false }}"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_mut_ref_expr() {
        let prog = parse("fn main() { let mut x = 42; let r = &mut x; }").unwrap();
        match &prog.funcs[0].body.stmts[1] {
            Stmt::Let { init, .. } => match init {
                Expr::Ref { expr, is_mut: true } => {
                    assert!(matches!(expr.as_ref(), Expr::Ident(name) if name == "x"));
                }
                _ => panic!("expected Ref {{ expr: Ident, is_mut: true }}"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_deref_expr() {
        let prog = parse("fn main() { let x = 42; let r = &x; let v = *r; }").unwrap();
        match &prog.funcs[0].body.stmts[2] {
            Stmt::Let { init, .. } => match init {
                Expr::Deref(expr) => {
                    assert!(matches!(expr.as_ref(), Expr::Ident(name) if name == "r"));
                }
                _ => panic!("expected Deref(Ident(r))"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_ref_type_annotation() {
        let prog = parse("fn main() { let r: &i32 = &42; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                type_ann: Some(ty), ..
            } => {
                assert_eq!(
                    *ty,
                    Type::Ref {
                        inner: Box::new(Type::I32),
                        is_mut: false
                    }
                );
            }
            _ => panic!("expected Let with type annotation"),
        }
    }

    #[test]
    fn test_mut_ref_type_annotation() {
        let prog = parse("fn main() { let r: &mut i32 = &mut 42; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                type_ann: Some(ty), ..
            } => {
                assert_eq!(
                    *ty,
                    Type::Ref {
                        inner: Box::new(Type::I32),
                        is_mut: true
                    }
                );
            }
            _ => panic!("expected Let with &mut i32 type"),
        }
    }

    #[test]
    fn test_deref_assign() {
        let prog = parse("fn main() { let mut x = 42; let r = &mut x; *r = 99; }").unwrap();
        let stmts = &prog.funcs[0].body.stmts;
        assert_eq!(stmts.len(), 3);
        match &stmts[2] {
            Stmt::Expr(Expr::Assign { target, value }) => {
                assert!(matches!(target.as_ref(), Expr::Deref(_)));
                assert!(matches!(value.as_ref(), Expr::IntLit(99)));
            }
            _ => panic!("expected Expr(Assign(*r, 99))"),
        }
    }

    #[test]
    fn test_str_literal_type_annotation() {
        // String literals should accept &str type annotation
        // Note: parse will succeed because &str is a valid type annotation
        let prog = parse("fn main() { let s: &str = \"hello\"; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                type_ann: Some(ty), ..
            } => {
                assert_eq!(
                    *ty,
                    Type::Ref {
                        inner: Box::new(Type::Str),
                        is_mut: false
                    }
                );
            }
            _ => panic!("expected Let with &str type"),
        }
    }

    #[test]
    fn test_const_with_arithmetic() {
        let prog = parse("fn main() { const X = 10 + 20; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Const { init, .. } => match init {
                Expr::Binary {
                    op: BinOp::Add,
                    lhs,
                    rhs,
                } => {
                    assert!(matches!(lhs.as_ref(), Expr::IntLit(10)));
                    assert!(matches!(rhs.as_ref(), Expr::IntLit(20)));
                }
                _ => panic!("expected Binary(Add, 10, 20)"),
            },
            _ => panic!("expected Const stmt"),
        }
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
    fn test_call_print() {
        let prog = parse("fn f() { print(42); }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Call { callee, args }) => {
                assert_eq!(callee, "print");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::IntLit(42)));
            }
            _ => panic!("expected Call(print, [42])"),
        }
    }

    #[test]
    fn test_call_println() {
        let prog = parse("fn f() { println(42); }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Call { callee, args }) => {
                assert_eq!(callee, "println");
                assert_eq!(args.len(), 1);
                assert!(matches!(&args[0], Expr::IntLit(42)));
            }
            _ => panic!("expected Call(println, [42])"),
        }
    }

    #[test]
    fn test_multi_arg_call() {
        let prog = parse("fn f() { foo(1, 2, 3); }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Call { callee, args }) => {
                assert_eq!(callee, "foo");
                assert_eq!(args.len(), 3);
                assert!(matches!(&args[0], Expr::IntLit(1)));
                assert!(matches!(&args[1], Expr::IntLit(2)));
                assert!(matches!(&args[2], Expr::IntLit(3)));
            }
            _ => panic!("expected multi-arg Call"),
        }
    }

    #[test]
    fn test_empty_call_args() {
        let prog = parse("fn f() { foo(); }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Call { callee, args }) => {
                assert_eq!(callee, "foo");
                assert!(args.is_empty());
            }
            _ => panic!("expected empty-arg Call"),
        }
    }

    #[test]
    fn test_use_decl_namespace() {
        let prog = parse("use std::io;\nfn main() {}").unwrap();
        assert_eq!(prog.uses.len(), 1);
        assert_eq!(prog.uses[0].path, vec!["std", "io"]);
    }

    #[test]
    fn test_use_decl_direct() {
        let prog = parse("use std::io::println;\nfn main() {}").unwrap();
        assert_eq!(prog.uses.len(), 1);
        assert_eq!(prog.uses[0].path, vec!["std", "io", "println"]);
    }

    #[test]
    fn test_use_decl_error_short_path() {
        let result = parse("use std;\nfn main() {}");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.msg.contains("at least 2 segments"),
                "error should mention path length: {}",
                e.msg
            );
        }
    }

    #[test]
    fn test_extern_c_block() {
        let prog =
            parse(r#"extern "C" { fn printf(fmt: *const i8, ...) -> i32; } fn main() {}"#).unwrap();
        let printf_fn = prog.funcs.iter().find(|f| f.name == "printf").unwrap();
        assert!(printf_fn.is_extern);
        assert_eq!(printf_fn.params.len(), 1);
        assert_eq!(printf_fn.params[0].name, "fmt");
        assert_eq!(
            printf_fn.params[0].ty,
            Type::Ptr {
                inner: Box::new(Type::I8),
                is_mut: false
            }
        );
        assert_eq!(printf_fn.return_type, Some(Type::I32));
    }

    #[test]
    fn test_fn_with_params() {
        let prog = parse("fn add(x: i32, y: i32) -> i32 { x; }").unwrap();
        let f = &prog.funcs[0];
        assert_eq!(f.params.len(), 2);
        assert_eq!(f.params[0].name, "x");
        assert_eq!(f.params[0].ty, Type::I32);
        assert_eq!(f.params[1].name, "y");
        assert_eq!(f.params[1].ty, Type::I32);
        assert_eq!(f.return_type, Some(Type::I32));
    }

    #[test]
    fn test_fn_no_return_type() {
        let prog = parse("fn foo(x: i32) {}").unwrap();
        let f = &prog.funcs[0];
        assert_eq!(f.params.len(), 1);
        assert!(f.return_type.is_none());
    }

    #[test]
    fn test_string_literal() {
        let prog = parse(r#"fn f() { "hello"; }"#).unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::StrLit(s)) => {
                assert_eq!(s, "hello");
            }
            _ => panic!("expected StrLit"),
        }
    }

    #[test]
    fn test_qualified_call() {
        // Multiple args → qualified call (single arg → enum literal)
        let prog = parse("fn f() { io::println(42, 43); }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::QualifiedCall {
                module,
                callee,
                args,
            }) => {
                assert_eq!(module, "io");
                assert_eq!(callee, "println");
                assert_eq!(args.len(), 2);
                assert!(matches!(&args[0], Expr::IntLit(42)));
                assert!(matches!(&args[1], Expr::IntLit(43)));
            }
            _ => panic!("expected QualifiedCall"),
        }
    }

    #[test]
    fn test_enum_lit_with_payload() {
        let prog = parse("enum Option<T> { Some(T), None } fn f() { let x = Option::Some(42); }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { init, .. } => match init {
                Expr::EnumLit {
                    enum_name,
                    variant,
                    payload,
                } => {
                    assert_eq!(enum_name, "Option");
                    assert_eq!(variant, "Some");
                    assert!(matches!(payload, Some(p) if matches!(**p, Expr::IntLit(42))));
                }
                _ => panic!("expected EnumLit"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_enum_lit_unit_variant() {
        let prog = parse("enum Option<T> { Some(T), None } fn f() { let x = Option::None; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { init, .. } => match init {
                Expr::EnumLit {
                    enum_name,
                    variant,
                    payload,
                } => {
                    assert_eq!(enum_name, "Option");
                    assert_eq!(variant, "None");
                    assert!(payload.is_none());
                }
                _ => panic!("expected EnumLit"),
            },
            _ => panic!("expected Let"),
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
    fn test_tail_expr_as_implicit_return() {
        let prog = parse("fn f() { 42 }").unwrap();
        assert!(prog.funcs[0].body.tail_expr.is_some());
        assert!(matches!(
            prog.funcs[0].body.tail_expr.as_ref().unwrap().as_ref(),
            Expr::IntLit(42)
        ));
    }

    #[test]
    fn test_invalid_missing_fn_name() {
        let result = parse("fn ()");
        assert!(result.is_err());
        assert!(result.unwrap_err().msg.contains("function name"));
    }

    #[test]
    fn test_invalid_trailing_garbage() {
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
            let mut parser = Parser::new(&tokens);
            assert!(parser.parse_program().is_err());
        }
    }

    #[test]
    fn test_typed_let() {
        let prog = parse("fn main() { let x: i32 = 42; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                name,
                type_ann: Some(ty),
                init,
                ..
            } => {
                assert_eq!(name, "x");
                assert_eq!(*ty, Type::I32);
                assert!(matches!(init, Expr::IntLit(42)));
            }
            _ => panic!("expected Let with type annotation"),
        }
    }

    #[test]
    fn test_typed_let_f64() {
        let prog = parse("fn main() { let x: f64 = 3.14; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                name,
                type_ann: Some(ty),
                init,
                ..
            } => {
                assert_eq!(name, "x");
                assert_eq!(*ty, Type::F64);
                assert!(matches!(init, Expr::FloatLit(v) if (*v - 3.14).abs() < 1e-10));
            }
            _ => panic!("expected Let with f64 annotation"),
        }
    }

    #[test]
    fn test_let_no_type_ann() {
        let prog = parse("fn main() { let x = 42; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { type_ann: None, .. } => {}
            _ => panic!("expected Let without type annotation"),
        }
    }

    #[test]
    fn test_typed_const() {
        let prog = parse("fn main() { const X: u64 = 100; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Const {
                name,
                type_ann: Some(ty),
                ..
            } => {
                assert_eq!(name, "X");
                assert_eq!(*ty, Type::U64);
            }
            _ => panic!("expected Const with type annotation"),
        }
    }

    #[test]
    fn test_float_literal_expr() {
        let prog = parse("fn main() { 3.14; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::FloatLit(v)) => {
                assert!((*v - 3.14).abs() < 1e-10);
            }
            _ => panic!("expected FloatLit"),
        }
    }

    #[test]
    fn test_bool_type() {
        let prog = parse("fn main() { let x: bool = true; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                name,
                type_ann: Some(ty),
                init,
                ..
            } => {
                assert_eq!(name, "x");
                assert_eq!(*ty, Type::Bool);
                assert!(matches!(init, Expr::BoolLit(true)));
            }
            _ => panic!("expected Let with bool type"),
        }
    }

    #[test]
    fn test_bool_literals() {
        let prog = parse("fn main() { let a = true; let b = false; }").unwrap();
        let stmts = &prog.funcs[0].body.stmts;
        match &stmts[0] {
            Stmt::Let { init, .. } => assert!(matches!(init, Expr::BoolLit(true))),
            _ => panic!("expected BoolLit(true)"),
        }
        match &stmts[1] {
            Stmt::Let { init, .. } => assert!(matches!(init, Expr::BoolLit(false))),
            _ => panic!("expected BoolLit(false)"),
        }
    }

    #[test]
    fn test_as_cast() {
        let prog = parse("fn main() { let x = 42 as i64; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { init, .. } => match init {
                Expr::Cast { expr, to_type } => {
                    assert!(matches!(expr.as_ref(), Expr::IntLit(42)));
                    assert_eq!(*to_type, Type::I64);
                }
                _ => panic!("expected Cast"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_chained_cast() {
        let prog = parse("fn main() { let x = 42 as i64 as u8; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { init, .. } => match init {
                Expr::Cast {
                    expr,
                    to_type: outer_ty,
                } => {
                    assert_eq!(*outer_ty, Type::U8);
                    match expr.as_ref() {
                        Expr::Cast {
                            expr: inner_expr,
                            to_type: inner_ty,
                        } => {
                            assert_eq!(*inner_ty, Type::I64);
                            assert!(matches!(inner_expr.as_ref(), Expr::IntLit(42)));
                        }
                        _ => panic!("expected inner Cast"),
                    }
                }
                _ => panic!("expected Cast"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_cast_precedence_add() {
        // a + b as i32 → a + (b as i32)
        let prog = parse("fn f() { 1 + 2 as i32; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Binary {
                op: BinOp::Add,
                lhs,
                rhs,
            }) => {
                assert!(matches!(lhs.as_ref(), Expr::IntLit(1)));
                match rhs.as_ref() {
                    Expr::Cast {
                        expr,
                        to_type: Type::I32,
                    } => {
                        assert!(matches!(expr.as_ref(), Expr::IntLit(2)));
                    }
                    _ => panic!("expected Cast(rhs, I32)"),
                }
            }
            _ => panic!("expected Binary(Add, ..)"),
        }
    }

    #[test]
    fn test_cast_precedence_mul() {
        // a * b as i32 → a * (b as i32)
        let prog = parse("fn f() { 3 * 4 as i64; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Binary {
                op: BinOp::Mul,
                lhs,
                rhs,
            }) => {
                assert!(matches!(lhs.as_ref(), Expr::IntLit(3)));
                match rhs.as_ref() {
                    Expr::Cast {
                        expr,
                        to_type: Type::I64,
                    } => {
                        assert!(matches!(expr.as_ref(), Expr::IntLit(4)));
                    }
                    _ => panic!("expected Cast(rhs, I64)"),
                }
            }
            _ => panic!("expected Binary(Mul, ..)"),
        }
    }

    #[test]
    fn test_cast_paren_grouping() {
        // (1 + 2) as f64 → Cast(+(1,2), f64)
        let prog = parse("fn f() { (1 + 2) as f64; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Cast {
                expr,
                to_type: Type::F64,
            }) => match expr.as_ref() {
                Expr::Binary { op: BinOp::Add, .. } => {}
                _ => panic!("expected Binary(Add, ..) inside Cast"),
            },
            _ => panic!("expected Cast"),
        }
    }

    #[test]
    fn test_ident_as_struct_type() {
        // Identifiers are now valid as struct type names
        let prog = parse("fn f() { let x: MyStruct = 1; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                type_ann: Some(ty), ..
            } => {
                assert_eq!(*ty, Type::Struct("MyStruct".to_string()));
            }
            _ => panic!("expected Let with type annotation"),
        }
    }

    #[test]
    fn test_int_suffix_literal() {
        let prog = parse("fn main() { let x = 42i32; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { init, .. } => match init {
                Expr::Cast { expr, to_type } => {
                    assert!(matches!(expr.as_ref(), Expr::IntLit(42)));
                    assert_eq!(*to_type, Type::I32);
                }
                _ => panic!("expected Cast"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_u8_suffix_literal() {
        let prog = parse("fn main() { let x = 255u8; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { init, .. } => match init {
                Expr::Cast { expr, to_type } => {
                    assert!(matches!(expr.as_ref(), Expr::IntLit(255)));
                    assert_eq!(*to_type, Type::U8);
                }
                _ => panic!("expected Cast"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_float_suffix_literal() {
        let prog = parse("fn main() { let x = 3.14f64; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { init, .. } => match init {
                Expr::Cast { expr, to_type } => {
                    assert!(
                        matches!(expr.as_ref(), Expr::FloatLit(v) if (*v - 3.14).abs() < 1e-10)
                    );
                    assert_eq!(*to_type, Type::F64);
                }
                _ => panic!("expected Cast"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_suffix_in_expression() {
        let prog = parse("fn main() { let x = 10i64 + 20i32; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { init, .. } => match init {
                Expr::Binary {
                    op: BinOp::Add,
                    lhs,
                    rhs,
                } => {
                    match lhs.as_ref() {
                        Expr::Cast {
                            expr,
                            to_type: Type::I64,
                        } => assert!(matches!(expr.as_ref(), Expr::IntLit(10))),
                        _ => panic!("expected Cast lhs"),
                    }
                    match rhs.as_ref() {
                        Expr::Cast {
                            expr,
                            to_type: Type::I32,
                        } => assert!(matches!(expr.as_ref(), Expr::IntLit(20))),
                        _ => panic!("expected Cast rhs"),
                    }
                }
                _ => panic!("expected Binary"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_method_call_len() {
        let prog = parse("fn f() { \"hello\".len(); }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::MethodCall { expr, method, args }) => {
                assert_eq!(method, "len");
                assert!(args.is_empty());
                assert!(matches!(expr.as_ref(), Expr::StrLit(s) if s == "hello"));
            }
            _ => panic!("expected MethodCall(StrLit(hello), len, [])"),
        }
    }

    #[test]
    fn test_method_call_on_ident() {
        let prog = parse("fn f() { let s = \"hello\"; s.len(); }").unwrap();
        match &prog.funcs[0].body.stmts[1] {
            Stmt::Expr(Expr::MethodCall { expr, method, args }) => {
                assert_eq!(method, "len");
                assert!(args.is_empty());
                assert!(matches!(expr.as_ref(), Expr::Ident(name) if name == "s"));
            }
            _ => panic!("expected MethodCall"),
        }
    }

    #[test]
    fn test_parse_pointer_type() {
        let prog = parse("fn f(x: *const i8) {}").unwrap();
        assert_eq!(
            prog.funcs[0].params[0].ty,
            Type::Ptr {
                inner: Box::new(Type::I8),
                is_mut: false
            }
        );
    }

    #[test]
    fn test_struct_decl_empty() {
        let prog = parse("struct Empty {} fn main() {}").unwrap();
        assert_eq!(prog.structs.len(), 1);
        assert_eq!(prog.structs[0].name, "Empty");
        assert!(prog.structs[0].fields.is_empty());
    }

    #[test]
    fn test_struct_decl_fields() {
        let prog = parse("struct Point { x: i32, y: i32, } fn main() {}").unwrap();
        assert_eq!(prog.structs.len(), 1);
        assert_eq!(prog.structs[0].name, "Point");
        assert_eq!(prog.structs[0].fields.len(), 2);
        assert_eq!(prog.structs[0].fields[0].name, "x");
        assert_eq!(prog.structs[0].fields[0].ty, Type::I32);
        assert_eq!(prog.structs[0].fields[1].name, "y");
        assert_eq!(prog.structs[0].fields[1].ty, Type::I32);
    }

    #[test]
    fn test_struct_literal() {
        let prog = parse(
            "struct Point { x: i32, y: i32, }\nfn main() { let p = Point { x: 10, y: 20 }; }",
        )
        .unwrap();
        assert_eq!(prog.structs.len(), 1);
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let { init, .. } => match init {
                Expr::StructLit {
                    struct_name,
                    fields,
                } => {
                    assert_eq!(struct_name, "Point");
                    assert_eq!(fields.len(), 2);
                    assert_eq!(fields[0].0, "x");
                    assert!(matches!(fields[0].1, Expr::IntLit(10)));
                    assert_eq!(fields[1].0, "y");
                    assert!(matches!(fields[1].1, Expr::IntLit(20)));
                }
                _ => panic!("expected StructLit"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_struct_literal_shorthand() {
        let prog = parse(
            "struct Point { x: i32, y: i32, }\nfn main() { let x = 10; let p = Point { x, y: 20 }; }",
        )
        .unwrap();
        match &prog.funcs[0].body.stmts[1] {
            Stmt::Let { init, .. } => match init {
                Expr::StructLit { fields, .. } => {
                    assert_eq!(fields.len(), 2);
                    assert_eq!(fields[0].0, "x");
                    assert!(matches!(&fields[0].1, Expr::Ident(name) if name == "x"));
                    assert_eq!(fields[1].0, "y");
                    assert!(matches!(fields[1].1, Expr::IntLit(20)));
                }
                _ => panic!("expected StructLit"),
            },
            _ => panic!("expected Let"),
        }
    }

    #[test]
    fn test_impl_block_inherent() {
        let prog = parse(
            "struct Point { x: i32, y: i32, }\nimpl Point { fn new(x: i32, y: i32) -> Point { Point { x: x, y: y }; } fn area(&self) -> i32 { self.x * self.y; } }\nfn main() {}",
        )
        .unwrap();
        assert_eq!(prog.impls.len(), 1);
        assert!(prog.impls[0].trait_name.is_none());
        assert_eq!(prog.impls[0].methods.len(), 2);
        assert_eq!(prog.impls[0].methods[0].name, "new");
        assert_eq!(prog.impls[0].methods[0].params.len(), 2);
        assert!(prog.impls[0].methods[0].is_method);
        assert_eq!(prog.impls[0].methods[1].name, "area");
        assert_eq!(prog.impls[0].methods[1].params.len(), 1);
        assert_eq!(prog.impls[0].methods[1].params[0].name, "self");
    }

    #[test]
    fn test_impl_for_trait() {
        let prog = parse(
            "struct Point { x: i32, y: i32, }\ntrait Drawable { fn draw(&self); }\nimpl Drawable for Point { fn draw(&self) { } }\nfn main() {}",
        )
        .unwrap();
        assert_eq!(prog.traits.len(), 1);
        assert_eq!(prog.traits[0].name, "Drawable");
        assert_eq!(prog.traits[0].methods.len(), 1);
        assert_eq!(prog.traits[0].methods[0].name, "draw");

        assert_eq!(prog.impls.len(), 1);
        assert_eq!(prog.impls[0].trait_name, Some("Drawable".to_string()));
    }

    #[test]
    fn test_trait_decl_syntax() {
        let prog = parse("trait Default { fn default() -> Self; } fn main() {}").unwrap();
        assert_eq!(prog.traits.len(), 1);
        assert_eq!(prog.traits[0].name, "Default");
        assert_eq!(prog.traits[0].methods.len(), 1);
        assert_eq!(prog.traits[0].methods[0].name, "default");
        assert!(prog.traits[0].methods[0].params.is_empty());
        assert_eq!(prog.traits[0].methods[0].return_type, Some(Type::SelfType));
    }

    #[test]
    fn test_struct_in_program() {
        let prog = parse("struct A {} struct B {} fn main() {}").unwrap();
        assert_eq!(prog.structs.len(), 2);
        assert_eq!(prog.structs[0].name, "A");
        assert_eq!(prog.structs[1].name, "B");
    }

    #[test]
    fn test_trait_in_program() {
        let prog = parse("trait A { fn foo(); } trait B { fn bar(); } fn main() {}").unwrap();
        assert_eq!(prog.traits.len(), 2);
        assert_eq!(prog.traits[0].name, "A");
        assert_eq!(prog.traits[1].name, "B");
    }

    #[test]
    fn test_derive_attr_on_struct() {
        let prog =
            parse("#[derive(Default)]\nstruct Point { x: i32, y: i32, }\nfn main() {}").unwrap();
        assert_eq!(prog.structs.len(), 1);
        assert_eq!(prog.structs[0].name, "Point");
        assert_eq!(prog.structs[0].attribs.len(), 1);
        assert_eq!(prog.structs[0].attribs[0].name, "derive");
        assert_eq!(prog.structs[0].attribs[0].args, vec!["Default"]);
    }

    #[test]
    fn test_derive_attr_multiple_traits() {
        let prog =
            parse("#[derive(Default, Clone)]\nstruct Point { x: i32, y: i32, }\nfn main() {}")
                .unwrap();
        assert_eq!(prog.structs[0].attribs.len(), 1);
        assert_eq!(prog.structs[0].attribs[0].name, "derive");
        assert_eq!(prog.structs[0].attribs[0].args, vec!["Default", "Clone"]);
    }

    #[test]
    fn test_derive_attr_empty_parens() {
        let prog = parse("#[derive()]\nstruct Point { x: i32, y: i32, }\nfn main() {}").unwrap();
        assert_eq!(prog.structs[0].attribs.len(), 1);
        assert_eq!(prog.structs[0].attribs[0].name, "derive");
        assert!(prog.structs[0].attribs[0].args.is_empty());
    }

    #[test]
    fn test_derive_attr_on_fn() {
        let prog = parse("#[inline]\nfn foo() {}\nfn main() {}").unwrap();
        assert_eq!(prog.funcs.len(), 2);
        assert_eq!(prog.funcs[0].name, "foo");
        assert_eq!(prog.funcs[0].attribs.len(), 1);
        assert_eq!(prog.funcs[0].attribs[0].name, "inline");
        assert!(prog.funcs[0].attribs[0].args.is_empty());
    }

    #[test]
    fn test_multiple_attrs_on_struct() {
        let prog = parse(
            "#[derive(Default)]\n#[derive(Clone)]\nstruct Point { x: i32, y: i32, }\nfn main() {}",
        )
        .unwrap();
        assert_eq!(prog.structs[0].attribs.len(), 2);
        assert_eq!(prog.structs[0].attribs[0].name, "derive");
        assert_eq!(prog.structs[0].attribs[0].args, vec!["Default"]);
        assert_eq!(prog.structs[0].attribs[1].name, "derive");
        assert_eq!(prog.structs[0].attribs[1].args, vec!["Clone"]);
    }

    #[test]
    fn test_derive_attr_malformed_no_name() {
        let result = parse("#[ ]\nstruct Point { x: i32, }\nfn main() {}");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().msg.contains("attribute name"),
            "error should mention attribute name"
        );
    }

    #[test]
    fn test_if_expr() {
        let prog = parse("fn main() { if 1 { 2 } }").unwrap();
        assert!(prog.funcs[0].body.tail_expr.is_some());
        match prog.funcs[0].body.tail_expr.as_ref().unwrap().as_ref() {
            Expr::If {
                cond, then_block, ..
            } => {
                assert!(matches!(cond.as_ref(), Expr::IntLit(1)));
                assert!(then_block.tail_expr.is_some());
                assert!(then_block.stmts.is_empty());
            }
            _ => panic!("expected If expr"),
        }
    }

    #[test]
    fn test_if_else_expr() {
        let prog = parse("fn main() { if 1 { 2 } else { 3 } }").unwrap();
        assert!(prog.funcs[0].body.tail_expr.is_some());
        match prog.funcs[0].body.tail_expr.as_ref().unwrap().as_ref() {
            Expr::If {
                cond,
                then_block,
                else_ifs,
                else_block,
                ..
            } => {
                assert!(matches!(cond.as_ref(), Expr::IntLit(1)));
                assert!(then_block.tail_expr.is_some());
                assert!(else_ifs.is_empty());
                assert!(else_block.is_some());
                if let Some(el) = else_block {
                    assert!(el.tail_expr.is_some());
                }
            }
            _ => panic!("expected If expr"),
        }
    }

    #[test]
    fn test_if_else_if_else() {
        let prog = parse("fn main() { if 1 { 2 } else if 3 { 4 } else { 5 } }").unwrap();
        assert!(prog.funcs[0].body.tail_expr.is_some());
        match prog.funcs[0].body.tail_expr.as_ref().unwrap().as_ref() {
            Expr::If {
                cond,
                else_ifs,
                else_block,
                ..
            } => {
                assert!(matches!(cond.as_ref(), Expr::IntLit(1)));
                assert_eq!(else_ifs.len(), 1);
                assert!(else_block.is_some());
            }
            _ => panic!("expected If expr"),
        }
    }

    #[test]
    fn test_loop_expr() {
        let prog = parse("fn main() { loop { 1 } }").unwrap();
        assert!(prog.funcs[0].body.tail_expr.is_some());
        match prog.funcs[0].body.tail_expr.as_ref().unwrap().as_ref() {
            Expr::Loop { body } => {
                assert!(body.tail_expr.is_some());
            }
            _ => panic!("expected Loop expr"),
        }
    }

    #[test]
    fn test_while_expr() {
        let prog = parse("fn main() { while 1 { 2 } }").unwrap();
        assert!(prog.funcs[0].body.tail_expr.is_some());
        match prog.funcs[0].body.tail_expr.as_ref().unwrap().as_ref() {
            Expr::While { cond, body } => {
                assert!(matches!(cond.as_ref(), Expr::IntLit(1)));
                assert!(body.tail_expr.is_some());
            }
            _ => panic!("expected While expr"),
        }
    }

    #[test]
    fn test_return_stmt() {
        let prog = parse("fn main() { return 42; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Return { value, .. } => {
                assert!(value.is_some());
                assert!(matches!(value.as_ref().unwrap().as_ref(), Expr::IntLit(42)));
            }
            _ => panic!("expected Return stmt"),
        }
    }

    #[test]
    fn test_return_empty_stmt() {
        let prog = parse("fn main() { return; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Return { value, .. } => {
                assert!(value.is_none());
            }
            _ => panic!("expected Return stmt"),
        }
    }

    #[test]
    fn test_implicit_return_tail_expr() {
        let prog = parse("fn main() -> i32 { 42 }").unwrap();
        assert!(prog.funcs[0].body.tail_expr.is_some());
        assert!(matches!(
            prog.funcs[0].body.tail_expr.as_ref().unwrap().as_ref(),
            Expr::IntLit(42)
        ));
    }

    #[test]
    fn test_comparison_operators() {
        let prog = parse("fn main() { 1 == 2; 3 != 4; 5 < 6; 7 > 8; 9 <= 10; 11 >= 12; }").unwrap();
        let stmts = &prog.funcs[0].body.stmts;
        assert_eq!(stmts.len(), 6);
        let ops = [
            BinOp::Eq,
            BinOp::Neq,
            BinOp::Lt,
            BinOp::Gt,
            BinOp::Le,
            BinOp::Ge,
        ];
        for (i, expected_op) in ops.iter().enumerate() {
            match &stmts[i] {
                Stmt::Expr(Expr::Binary { op, .. }) => {
                    assert_eq!(op, expected_op);
                }
                _ => panic!("expected Binary expr at index {}", i),
            }
        }
    }

    #[test]
    fn test_comparison_chaining() {
        // Comparisons are left-associative: 1 < 2 == 3 > 4
        // parses as (((1 < 2) == 3) > 4) since all comparisons are at same level
        let prog = parse("fn main() { 1 < 2 == 3 > 4; }").unwrap();
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Expr(Expr::Binary {
                op: BinOp::Gt,
                lhs,
                rhs,
            }) => {
                assert!(matches!(rhs.as_ref(), Expr::IntLit(4)));
                assert!(matches!(lhs.as_ref(), Expr::Binary { op: BinOp::Eq, .. }));
            }
            _ => panic!("expected Gt as top-level"),
        }
    }

    #[test]
    fn test_nested_control_flow() {
        let prog = parse("fn main() { if 1 { loop { if 2 { 3 } } } }").unwrap();
        assert!(prog.funcs[0].body.tail_expr.is_some());
        match prog.funcs[0].body.tail_expr.as_ref().unwrap().as_ref() {
            Expr::If { then_block, .. } => {
                assert!(then_block.tail_expr.is_some());
                match then_block.tail_expr.as_ref().unwrap().as_ref() {
                    Expr::Loop { body } => {
                        assert!(body.tail_expr.is_some());
                        match body.tail_expr.as_ref().unwrap().as_ref() {
                            Expr::If { .. } => {}
                            _ => panic!("expected nested If"),
                        }
                    }
                    _ => panic!("expected Loop"),
                }
            }
            _ => panic!("expected nested If"),
        }
    }

    #[test]
    fn test_type_alias_simple() {
        let prog = parse("type Meters = i32; fn main() {}").unwrap();
        assert_eq!(prog.type_aliases.len(), 1);
        let alias = &prog.type_aliases[0];
        assert_eq!(alias.name, "Meters");
        assert!(alias.type_params.is_empty());
        assert_eq!(alias.aliased_type, Type::I32);
    }

    #[test]
    fn test_type_alias_generic() {
        let prog = parse("type Pair<T> = (T, T); fn main() {}").unwrap();
        assert_eq!(prog.type_aliases.len(), 1);
        let alias = &prog.type_aliases[0];
        assert_eq!(alias.name, "Pair");
        assert_eq!(alias.type_params, vec!["T".to_string()]);
        assert_eq!(
            alias.aliased_type,
            Type::Tuple(vec![Type::Struct("T".into()), Type::Struct("T".into())])
        );
    }

    #[test]
    fn test_type_alias_in_type_position() {
        // After resolution, Meters should be resolved to i32 in let type annotation
        let prog = parse("type Meters = i32; fn main() { let x: Meters = 42; }").unwrap();
        assert_eq!(prog.type_aliases.len(), 1);
        // The let stmt should have type i32 (resolved from Meters)
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                type_ann: Some(ty), ..
            } => {
                assert_eq!(*ty, Type::I32);
            }
            _ => panic!("expected Let with type annotation"),
        }
    }

    #[test]
    fn test_type_alias_generic_usage() {
        let prog =
            parse("type Pair<T> = (T, T); fn main() { let x: Pair<i32> = (1, 2); }").unwrap();
        // After resolution, Pair<i32> should be resolved to (i32, i32)
        match &prog.funcs[0].body.stmts[0] {
            Stmt::Let {
                type_ann: Some(ty), ..
            } => {
                assert_eq!(*ty, Type::Tuple(vec![Type::I32, Type::I32]));
            }
            _ => panic!("expected Let with type annotation"),
        }
    }

    #[test]
    fn test_type_alias_cycle() {
        let result = parse("type A = B; type B = A; fn main() {}");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.msg.contains("cyclic"),
                "error should mention cyclic: {}",
                e.msg
            );
        }
    }

    #[test]
    fn test_type_alias_duplicate_name() {
        let result = parse("type Foo = i32; struct Foo { x: i32, } fn main() {}");
        assert!(result.is_err());
        if let Err(e) = result {
            assert!(
                e.msg.contains("conflicts"),
                "error should mention conflicts: {}",
                e.msg
            );
        }
    }

    #[test]
    fn test_type_alias_in_fn_return() {
        // Alias in return type position should be resolved
        let prog = parse("type Meters = i32; fn main() -> Meters { 42 }").unwrap();
        assert_eq!(prog.funcs[0].return_type, Some(Type::I32));
    }

    #[test]
    fn test_type_alias_in_fn_param() {
        // Alias in function parameter should be resolved
        let prog = parse("type Meters = i32; fn foo(x: Meters) {}").unwrap();
        assert_eq!(prog.funcs[0].params[0].ty, Type::I32);
    }
}
