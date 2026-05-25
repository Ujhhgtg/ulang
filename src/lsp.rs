use std::collections::HashMap;
use lsp_server::{Connection, Message, Request, Response, Notification};
use lsp_types::{
    InitializeParams, ServerCapabilities, TextDocumentSyncCapability, TextDocumentSyncKind,
    HoverProviderCapability, OneOf, Url, Diagnostic, DiagnosticSeverity, Position, Range,
    Hover, HoverParams, HoverContents, MarkupContent, MarkupKind, GotoDefinitionParams,
    GotoDefinitionResponse, Location, PublishDiagnosticsParams,
};
use crate::token::{Span, Token};
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::ast::Program;

struct DocumentState {
    source: String,
    last_valid_program: Option<Program>,
}

fn offset_to_position(source: &str, offset: usize) -> Position {
    let mut line = 0;
    let mut col_utf16 = 0;
    let mut current_offset = 0;

    for c in source.chars() {
        if current_offset >= offset {
            break;
        }
        if c == '\n' {
            line += 1;
            col_utf16 = 0;
        } else {
            col_utf16 += c.len_utf16();
        }
        current_offset += c.len_utf8();
    }
    Position::new(line as u32, col_utf16 as u32)
}

fn position_to_offset(source: &str, position: Position) -> usize {
    let mut line = 0;
    let mut col_utf16 = 0;
    let mut current_offset = 0;

    for c in source.chars() {
        if line == position.line as usize && col_utf16 >= position.character as usize {
            break;
        }
        if c == '\n' {
            if line == position.line as usize {
                break;
            }
            line += 1;
            col_utf16 = 0;
        } else {
            col_utf16 += c.len_utf16();
        }
        current_offset += c.len_utf8();
    }
    current_offset
}

fn type_to_string(ty: &crate::ast::Type) -> String {
    match ty {
        crate::ast::Type::I8 => "i8".to_string(),
        crate::ast::Type::I16 => "i16".to_string(),
        crate::ast::Type::I32 => "i32".to_string(),
        crate::ast::Type::I64 => "i64".to_string(),
        crate::ast::Type::U8 => "u8".to_string(),
        crate::ast::Type::U16 => "u16".to_string(),
        crate::ast::Type::U32 => "u32".to_string(),
        crate::ast::Type::U64 => "u64".to_string(),
        crate::ast::Type::Usize => "usize".to_string(),
        crate::ast::Type::Isize => "isize".to_string(),
        crate::ast::Type::F32 => "f32".to_string(),
        crate::ast::Type::F64 => "f64".to_string(),
        crate::ast::Type::Bool => "bool".to_string(),
        crate::ast::Type::Never => "!".to_string(),
        crate::ast::Type::Tuple(tys) => {
            let parts: Vec<String> = tys.iter().map(type_to_string).collect();
            format!("({})", parts.join(", "))
        }
        crate::ast::Type::Unit => "()".to_string(),
        crate::ast::Type::Str => "str".to_string(),
        crate::ast::Type::Ptr { inner, is_mut } => {
            let mut_str = if *is_mut { "mut" } else { "const" };
            format!("*{} {}", mut_str, type_to_string(inner))
        }
        crate::ast::Type::Ref { inner, is_mut } => {
            let mut_str = if *is_mut { "mut " } else { "" };
            format!("&{}{}", mut_str, type_to_string(inner))
        }
        crate::ast::Type::Array { inner, len } => {
            format!("[{}; {}]", type_to_string(inner), len)
        }
        crate::ast::Type::Struct(name) => name.clone(),
        crate::ast::Type::GenericInstance(name, args) => {
            let parts: Vec<String> = args.iter().map(type_to_string).collect();
            format!("{}<{}>", name, parts.join(", "))
        }
        crate::ast::Type::Alias(name, args) => {
            if args.is_empty() {
                name.clone()
            } else {
                let parts: Vec<String> = args.iter().map(type_to_string).collect();
                format!("{}<{}>", name, parts.join(", "))
            }
        }
        crate::ast::Type::SelfType => "Self".to_string(),
    }
}

fn get_hover_text_from_program(program: &Program, name: &str) -> Option<String> {
    // 1. Search functions
    if let Some(func) = program.funcs.iter().find(|f| f.name == name) {
        let params: Vec<String> = func.params.iter().map(|p| {
            format!("{}: {}", p.name, type_to_string(&p.ty))
        }).collect();
        let ret = match &func.return_type {
            Some(ty) => format!(" -> {}", type_to_string(ty)),
            None => "".to_string(),
        };
        let is_ext = if func.is_extern { "extern \"C\" " } else { "" };
        return Some(format!("```rust\n{}fn {}({}){}\n```", is_ext, name, params.join(", "), ret));
    }

    // 2. Search structs
    if let Some(st) = program.structs.iter().find(|s| s.name == name) {
        let fields: Vec<String> = st.fields.iter().map(|f| {
            format!("    {}: {}", f.name, type_to_string(&f.ty))
        }).collect();
        let type_params = if st.type_params.is_empty() {
            "".to_string()
        } else {
            format!("<{}>", st.type_params.join(", "))
        };
        return Some(format!("```rust\nstruct {}{} {{\n{}\n}}\n```", name, type_params, fields.join(",\n")));
    }

    // 3. Search enums
    if let Some(en) = program.enums.iter().find(|e| e.name == name) {
        let variants: Vec<String> = en.variants.iter().map(|v| {
            match &v.ty {
                Some(ty) => format!("    {}({})", v.name, type_to_string(ty)),
                None => format!("    {}", v.name),
            }
        }).collect();
        let type_params = if en.type_params.is_empty() {
            "".to_string()
        } else {
            format!("<{}>", en.type_params.join(", "))
        };
        return Some(format!("```rust\nenum {}{} {{\n{}\n}}\n```", name, type_params, variants.join(",\n")));
    }

    // 4. Search traits
    if let Some(tr) = program.traits.iter().find(|t| t.name == name) {
        let methods: Vec<String> = tr.methods.iter().map(|m| {
            let params: Vec<String> = m.params.iter().map(|p| {
                format!("{}: {}", p.name, type_to_string(&p.ty))
            }).collect();
            let ret = match &m.return_type {
                Some(ty) => format!(" -> {}", type_to_string(ty)),
                None => "".to_string(),
            };
            format!("    fn {}({}){};", m.name, params.join(", "), ret)
        }).collect();
        let type_params = if tr.type_params.is_empty() {
            "".to_string()
        } else {
            format!("<{}>", tr.type_params.join(", "))
        };
        return Some(format!("```rust\ntrait {}{} {{\n{}\n}}\n```", name, type_params, methods.join("\n")));
    }

    // 5. Search type aliases
    if let Some(ta) = program.type_aliases.iter().find(|t| t.name == name) {
        let type_params = if ta.type_params.is_empty() {
            "".to_string()
        } else {
            format!("<{}>", ta.type_params.join(", "))
        };
        return Some(format!("```rust\ntype {}{} = {};\n```", name, type_params, type_to_string(&ta.aliased_type)));
    }

    // 6. Search impl methods
    for imp in &program.impls {
        if let Some(method) = imp.methods.iter().find(|m| m.name == name) {
            let params: Vec<String> = method.params.iter().map(|p| {
                format!("{}: {}", p.name, type_to_string(&p.ty))
            }).collect();
            let ret = match &method.return_type {
                Some(ty) => format!(" -> {}", type_to_string(ty)),
                None => "".to_string(),
            };
            let type_name = type_to_string(&imp.impl_type);
            let trait_str = match &imp.trait_name {
                Some(t) => format!("{} for ", t),
                None => "".to_string(),
            };
            return Some(format!(
                "```rust\nimpl {} {{\n    fn {}({}){}\n}}\n```",
                format!("{}{}", trait_str, type_name),
                name,
                params.join(", "),
                ret
            ));
        }
    }

    None
}

fn find_definition_span(tokens: &[(Token, Span)], name: &str, hover_offset: usize) -> Option<Span> {
    // 1. Local backward search for variables or parameters
    let hover_index = tokens.iter().position(|(_, span)| span.lo <= hover_offset && hover_offset <= span.hi);
    if let Some(idx) = hover_index {
        let mut i = idx;
        while i > 0 {
            i -= 1;
            if i + 1 < tokens.len() {
                if let Token::Ident(ref id) = tokens[i].0 {
                    if id == name {
                        if i > 0 && (tokens[i - 1].0 == Token::Let || tokens[i - 1].0 == Token::Mut) {
                            return Some(tokens[i].1);
                        }
                    }
                }
                if let Token::Ident(ref id) = tokens[i + 1].0 {
                    if id == name && tokens[i].0 == Token::Let {
                        return Some(tokens[i + 1].1);
                    }
                }
            }
            if tokens[i].0 == Token::Fn {
                // Check parameters of this enclosing function boundary
                let mut j = i + 1;
                let mut in_params = false;
                while j < tokens.len() && tokens[j].0 != Token::LBrace && tokens[j].0 != Token::Semicolon {
                    if tokens[j].0 == Token::LParen {
                        in_params = true;
                    } else if tokens[j].0 == Token::RParen {
                        break;
                    } else if in_params {
                        if let Token::Ident(ref id) = tokens[j].0 {
                            if id == name {
                                return Some(tokens[j].1);
                            }
                        }
                    }
                    j += 1;
                }
                break; // Stop local search at the function definition boundary
            }
        }
    }

    // 2. Global search for top-level constructs
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i].0 {
            Token::Fn | Token::Struct | Token::Enum | Token::Trait | Token::Type => {
                if i + 1 < tokens.len() {
                    if let Token::Ident(ref id) = tokens[i + 1].0 {
                        if id == name {
                            return Some(tokens[i + 1].1);
                        }
                    }
                }
            }
            _ => {}
        }
        i += 1;
    }

    None
}

fn get_diagnostics(source: &str) -> Vec<Diagnostic> {
    let mut diagnostics = Vec::new();
    let mut lexer = Lexer::new(source);
    let mut tokens = Vec::new();
    loop {
        match lexer.next_token() {
            Ok((token, span)) => {
                tokens.push((token, span));
                if let Token::Eof = tokens.last().unwrap().0 {
                    break;
                }
            }
            Err(msg) => {
                let pos = lexer.pos();
                let position = offset_to_position(source, pos);
                let range = Range::new(position, position);
                diagnostics.push(Diagnostic::new(
                    range,
                    Some(DiagnosticSeverity::ERROR),
                    None,
                    Some("ulang".to_string()),
                    format!("Lex error: {}", msg),
                    None,
                    None,
                ));
                return diagnostics;
            }
        }
    }

    let mut parser = Parser::new(&tokens);
    match parser.parse_program() {
        Ok(_) => {}
        Err(e) => {
            let start = offset_to_position(source, e.span.lo);
            let end = offset_to_position(source, e.span.hi);
            let range = Range::new(start, end);
            diagnostics.push(Diagnostic::new(
                range,
                Some(DiagnosticSeverity::ERROR),
                None,
                Some("ulang".to_string()),
                format!("Parse error: {}", e.msg),
                None,
                None,
            ));
        }
    }

    diagnostics
}

fn update_document(
    documents: &mut HashMap<Url, DocumentState>,
    url: Url,
    text: String,
) -> Vec<Diagnostic> {
    let diags = get_diagnostics(&text);
    
    let mut lexer = Lexer::new(&text);
    let mut tokens = Vec::new();
    let mut parse_success = false;
    let mut valid_prog = None;
    
    loop {
        match lexer.next_token() {
            Ok((token, span)) => {
                tokens.push((token, span));
                if let Token::Eof = tokens.last().unwrap().0 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    
    if !tokens.is_empty() && matches!(tokens.last().unwrap().0, Token::Eof) {
        let mut parser = Parser::new(&tokens);
        if let Ok(prog) = parser.parse_program() {
            valid_prog = Some(prog);
            parse_success = true;
        }
    }
    
    if let Some(doc) = documents.get_mut(&url) {
        doc.source = text;
        if parse_success {
            doc.last_valid_program = valid_prog;
        }
    } else {
        documents.insert(url, DocumentState {
            source: text,
            last_valid_program: valid_prog,
        });
    }
    
    diags
}

fn publish_diagnostics(
    connection: &Connection,
    url: Url,
    diagnostics: Vec<Diagnostic>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let params = PublishDiagnosticsParams::new(url, diagnostics, None);
    let notification = Notification::new("textDocument/publishDiagnostics".to_string(), params);
    connection.sender.send(Message::Notification(notification))?;
    Ok(())
}

fn handle_hover(
    documents: &HashMap<Url, DocumentState>,
    params: HoverParams,
) -> Option<Hover> {
    let url = params.text_document_position_params.text_document.uri;
    let doc = documents.get(&url)?;
    let position = params.text_document_position_params.position;
    
    let offset = position_to_offset(&doc.source, position);
    
    let mut lexer = Lexer::new(&doc.source);
    let mut tokens = Vec::new();
    loop {
        match lexer.next_token() {
            Ok((token, span)) => {
                tokens.push((token, span));
                if let Token::Eof = tokens.last().unwrap().0 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    
    let hovered_token = tokens.iter().find(|(_, span)| {
        span.lo <= offset && offset <= span.hi
    })?;
    
    if let Token::Ident(ref name) = hovered_token.0 {
        let hover_text = if let Some(prog) = &doc.last_valid_program {
            get_hover_text_from_program(prog, name)
        } else {
            None
        };
        
        if let Some(contents) = hover_text {
            return Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: contents,
                }),
                range: Some(Range::new(
                    offset_to_position(&doc.source, hovered_token.1.lo),
                    offset_to_position(&doc.source, hovered_token.1.hi),
                )),
            });
        }
    }
    
    None
}

fn handle_definition(
    documents: &HashMap<Url, DocumentState>,
    params: GotoDefinitionParams,
) -> Option<GotoDefinitionResponse> {
    let url = params.text_document_position_params.text_document.uri;
    let doc = documents.get(&url)?;
    let position = params.text_document_position_params.position;
    
    let offset = position_to_offset(&doc.source, position);
    
    let mut lexer = Lexer::new(&doc.source);
    let mut tokens = Vec::new();
    loop {
        match lexer.next_token() {
            Ok((token, span)) => {
                tokens.push((token, span));
                if let Token::Eof = tokens.last().unwrap().0 {
                    break;
                }
            }
            Err(_) => break,
        }
    }
    
    let clicked_token = tokens.iter().find(|(_, span)| {
        span.lo <= offset && offset <= span.hi
    })?;
    
    if let Token::Ident(ref name) = clicked_token.0 {
        if let Some(def_span) = find_definition_span(&tokens, name, offset) {
            let start = offset_to_position(&doc.source, def_span.lo);
            let end = offset_to_position(&doc.source, def_span.hi);
            let range = Range::new(start, end);
            return Some(GotoDefinitionResponse::Scalar(Location::new(url, range)));
        }
    }
    
    None
}

fn cast_request<R>(req: Request) -> Result<(lsp_server::RequestId, R::Params), Box<dyn std::error::Error + Send + Sync>>
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned,
{
    req.extract(R::METHOD)
        .map_err(|e| format!("failed to extract request: {:?}", e).into())
}

fn cast_notification<N>(not: Notification) -> Result<N::Params, Box<dyn std::error::Error + Send + Sync>>
where
    N: lsp_types::notification::Notification,
    N::Params: serde::de::DeserializeOwned,
{
    not.extract(N::METHOD)
        .map_err(|e| format!("failed to extract notification: {:?}", e).into())
}

fn main_loop(connection: Connection) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let mut documents: HashMap<Url, DocumentState> = HashMap::new();

    for msg in &connection.receiver {
        match msg {
            Message::Request(req) => {
                if connection.handle_shutdown(&req)? {
                    return Ok(());
                }
                match req.method.as_str() {
                    "textDocument/hover" => {
                        let (id, params) = cast_request::<lsp_types::request::HoverRequest>(req)?;
                        let response = handle_hover(&documents, params);
                        let result = serde_json::to_value(&response)?;
                        connection.sender.send(Message::Response(Response::new_ok(id, result)))?;
                    }
                    "textDocument/definition" => {
                        let (id, params) = cast_request::<lsp_types::request::GotoDefinition>(req)?;
                        let response = handle_definition(&documents, params);
                        let result = serde_json::to_value(&response)?;
                        connection.sender.send(Message::Response(Response::new_ok(id, result)))?;
                    }
                    _ => {}
                }
            }
            Message::Notification(not) => {
                match not.method.as_str() {
                    "textDocument/didOpen" => {
                        let params = cast_notification::<lsp_types::notification::DidOpenTextDocument>(not)?;
                        let url = params.text_document.uri;
                        let text = params.text_document.text;
                        
                        let diags = update_document(&mut documents, url.clone(), text);
                        publish_diagnostics(&connection, url, diags)?;
                    }
                    "textDocument/didChange" => {
                        let params = cast_notification::<lsp_types::notification::DidChangeTextDocument>(not)?;
                        let url = params.text_document.uri;
                        if let Some(change) = params.content_changes.into_iter().next() {
                            let diags = update_document(&mut documents, url.clone(), change.text);
                            publish_diagnostics(&connection, url, diags)?;
                        }
                    }
                    "textDocument/didClose" => {
                        let params = cast_notification::<lsp_types::notification::DidCloseTextDocument>(not)?;
                        documents.remove(&params.text_document.uri);
                    }
                    _ => {}
                }
            }
            Message::Response(_) => {}
        }
    }
    Ok(())
}

pub fn run_server() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    eprintln!("starting ulang lsp server");

    let (connection, io_threads) = Connection::stdio();

    let server_capabilities = serde_json::to_value(&ServerCapabilities {
        text_document_sync: Some(TextDocumentSyncCapability::Kind(TextDocumentSyncKind::FULL)),
        hover_provider: Some(HoverProviderCapability::Simple(true)),
        definition_provider: Some(OneOf::Left(true)),
        ..Default::default()
    })?;

    let initialization_params = connection.initialize(server_capabilities)?;
    let _params: InitializeParams = serde_json::from_value(initialization_params)?;

    main_loop(connection)?;
    io_threads.join()?;

    eprintln!("ulang lsp server stopped successfully");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_offset_conversion() {
        let src = "fn main() {\n    let x = 42;\n}";
        
        let pos1 = offset_to_position(src, 0);
        assert_eq!(pos1.line, 0);
        assert_eq!(pos1.character, 0);
        
        let pos2 = offset_to_position(src, 12);
        assert_eq!(pos2.line, 1);
        assert_eq!(pos2.character, 0);
        
        let offset1 = position_to_offset(src, pos1);
        assert_eq!(offset1, 0);
        
        let offset2 = position_to_offset(src, pos2);
        assert_eq!(offset2, 12);
    }

    #[test]
    fn test_diagnostics_success() {
        let src = "fn main() {}";
        let diags = get_diagnostics(src);
        assert!(diags.is_empty());
    }

    #[test]
    fn test_diagnostics_lex_error() {
        let src = "fn main() { @ }";
        let diags = get_diagnostics(src);
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("Lex error"));
    }

    #[test]
    fn test_diagnostics_parse_error() {
        let src = "fn main(";
        let diags = get_diagnostics(src);
        assert_eq!(diags.len(), 1);
        assert!(diags[0].message.contains("Parse error"));
    }
}
