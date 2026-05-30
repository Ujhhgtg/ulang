use crate::ast::Program;
use crate::lexer::Lexer;
use crate::parser::Parser;
use crate::token::{Span, Token};
use lsp_server::{Connection, Message, Notification, Request, Response};
use lsp_types::request::Completion;
use lsp_types::{
    CompletionItem, CompletionItemKind, CompletionOptions, CompletionParams, CompletionResponse,
    InsertTextFormat, TextEdit,
};
use lsp_types::{
    Diagnostic, DiagnosticSeverity, GotoDefinitionParams, GotoDefinitionResponse, Hover,
    HoverContents, HoverParams, HoverProviderCapability, InitializeParams, Location, MarkupContent,
    MarkupKind, OneOf, Position, PublishDiagnosticsParams, Range, ServerCapabilities,
    TextDocumentSyncCapability, TextDocumentSyncKind, Uri,
};
use std::collections::HashMap;
use url::Url;

struct DocumentState {
    source: String,
    last_valid_program: Option<Program>,
}

struct StdlibEntry {
    source: String,
    program: Program,
}

struct StdlibCache {
    entries: Vec<(Url, StdlibEntry)>,
}

fn load_stdlib_cache() -> StdlibCache {
    let mut entries = Vec::new();
    let stdlib_root = find_stdlib_root();
    let core_dir = stdlib_root.join("core");
    let std_dir = stdlib_root.join("std");

    let load_dir = |dir: &std::path::Path, entries: &mut Vec<(Url, StdlibEntry)>| {
        let mod_u_path = dir.join("mod.u");
        if let Ok(mod_src) = std::fs::read_to_string(&mod_u_path) {
            let mut lexer = Lexer::new(&mod_src);
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
            if !tokens.is_empty() && matches!(tokens.last().unwrap().0, Token::Eof) {
                let mut parser = Parser::new(&tokens);
                if let Ok(stdlib_root) = parser.parse_program() {
                    for m in &stdlib_root.modules {
                        let sub_path = dir.join(format!("{}.u", m.name));
                        if let Ok(src) = std::fs::read_to_string(&sub_path) {
                            let mut lexer = Lexer::new(&src);
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
                            if !tokens.is_empty() && matches!(tokens.last().unwrap().0, Token::Eof)
                            {
                                let mut parser = Parser::new(&tokens);
                                if let Ok(program) = parser.parse_program()
                                    && let Ok(url) = Url::from_file_path(&sub_path)
                                {
                                    entries.push((
                                        url,
                                        StdlibEntry {
                                            source: src,
                                            program,
                                        },
                                    ));
                                }
                            }
                        }
                    }
                    if let Ok(url) = Url::from_file_path(&mod_u_path) {
                        entries.push((
                            url,
                            StdlibEntry {
                                source: mod_src,
                                program: stdlib_root,
                            },
                        ));
                    }
                }
            }
        }
    };

    load_dir(&core_dir, &mut entries);
    load_dir(&std_dir, &mut entries);

    StdlibCache { entries }
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

fn trait_bound_to_string(bound: &crate::ast::TraitBound) -> String {
    if bound.generic_args.is_empty() {
        bound.trait_name.clone()
    } else {
        let parts: Vec<String> = bound.generic_args.iter().map(type_to_string).collect();
        format!("{}<{}>", bound.trait_name, parts.join(", "))
    }
}

fn format_generic_params(params: &[crate::ast::GenericParam]) -> String {
    if params.is_empty() {
        "".to_string()
    } else {
        let formatted: Vec<String> = params
            .iter()
            .map(|p| {
                if p.bounds.is_empty() {
                    p.name.clone()
                } else {
                    let bounds_str: Vec<String> =
                        p.bounds.iter().map(trait_bound_to_string).collect();
                    format!("{}: {}", p.name, bounds_str.join(" + "))
                }
            })
            .collect();
        format!("<{}>", formatted.join(", "))
    }
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
        crate::ast::Type::Slice { inner } => {
            format!("[{}]", type_to_string(inner))
        }
        crate::ast::Type::GenericArray { inner, len_var } => {
            format!("[{}; {}]", type_to_string(inner), len_var)
        }
        crate::ast::Type::ImplTrait(bounds) => {
            let parts: Vec<String> = bounds.iter().map(trait_bound_to_string).collect();
            format!("impl {}", parts.join(" + "))
        }
    }
}

fn get_hover_text_from_program(
    program: &Program,
    name: &str,
    struct_context: Option<&str>,
) -> Option<String> {
    get_hover_text_recursive(program, name, struct_context)
}

fn get_hover_text_recursive(
    program: &Program,
    name: &str,
    struct_context: Option<&str>,
) -> Option<String> {
    // 1. Search functions
    if let Some(func) = program.funcs.iter().find(|f| f.name == name) {
        let params: Vec<String> = func
            .params
            .iter()
            .map(|p| format!("{}: {}", p.name, type_to_string(&p.ty)))
            .collect();
        let ret = match &func.return_type {
            Some(ty) => format!(" -> {}", type_to_string(ty)),
            None => "".to_string(),
        };
        let pub_str = if func.is_pub { "pub " } else { "" };
        let is_ext = if func.is_extern { "extern \"C\" " } else { "" };
        let type_params = format_generic_params(&func.type_params);
        return Some(format!(
            "```rust\n{}{}fn {}{}({}){}\n```",
            pub_str,
            is_ext,
            name,
            type_params,
            params.join(", "),
            ret
        ));
    }

    // 2. Search structs
    if let Some(st) = program.structs.iter().find(|s| s.name == name) {
        let fields: Vec<String> = st
            .fields
            .iter()
            .map(|f| {
                let f_pub = if f.is_pub { "pub " } else { "" };
                format!("    {}{}: {}", f_pub, f.name, type_to_string(&f.ty))
            })
            .collect();
        let pub_str = if st.is_pub { "pub " } else { "" };
        let type_params = format_generic_params(&st.type_params);
        return Some(format!(
            "```rust\n{}struct {}{} {{\n{}\n}}\n```",
            pub_str,
            name,
            type_params,
            fields.join(",\n")
        ));
    }

    // Search struct fields
    for st in &program.structs {
        let matches_context = match struct_context {
            Some(ctx) => st.name == ctx,
            None => true,
        };
        if matches_context && let Some(field) = st.fields.iter().find(|f| f.name == name) {
            let pub_str = if st.is_pub { "pub " } else { "" };
            let f_pub = if field.is_pub { "pub " } else { "" };
            return Some(format!(
                "```rust\n{}struct {} {{\n    {}{}: {}\n}}\n```",
                pub_str,
                st.name,
                f_pub,
                field.name,
                type_to_string(&field.ty)
            ));
        }
    }

    // 3. Search enums
    if let Some(en) = program.enums.iter().find(|e| e.name == name) {
        let variants: Vec<String> = en
            .variants
            .iter()
            .map(|v| match &v.ty {
                Some(ty) => format!("    {}({})", v.name, type_to_string(ty)),
                None => format!("    {}", v.name),
            })
            .collect();
        let pub_str = if en.is_pub { "pub " } else { "" };
        let type_params = format_generic_params(&en.type_params);
        return Some(format!(
            "```rust\n{}enum {}{} {{\n{}\n}}\n```",
            pub_str,
            name,
            type_params,
            variants.join(",\n")
        ));
    }

    // Search enum variants
    for en in &program.enums {
        let matches_context = match struct_context {
            Some(ctx) => en.name == ctx,
            None => true,
        };
        if matches_context && let Some(variant) = en.variants.iter().find(|v| v.name == name) {
            let variant_str = match &variant.ty {
                Some(ty) => format!("{}({})", variant.name, type_to_string(ty)),
                None => variant.name.clone(),
            };
            let pub_str = if en.is_pub { "pub " } else { "" };
            return Some(format!(
                "```rust\n{}enum {} {{\n    {}\n}}\n```",
                pub_str, en.name, variant_str
            ));
        }
    }

    // 4. Search traits
    if let Some(tr) = program.traits.iter().find(|t| t.name == name) {
        let methods: Vec<String> = tr
            .methods
            .iter()
            .map(|m| {
                let params: Vec<String> = m
                    .params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, type_to_string(&p.ty)))
                    .collect();
                let ret = match &m.return_type {
                    Some(ty) => format!(" -> {}", type_to_string(ty)),
                    None => "".to_string(),
                };
                format!("    fn {}({}){};", m.name, params.join(", "), ret)
            })
            .collect();
        let pub_str = if tr.is_pub { "pub " } else { "" };
        let type_params = format_generic_params(&tr.type_params);
        return Some(format!(
            "```rust\n{}trait {}{} {{\n{}\n}}\n```",
            pub_str,
            name,
            type_params,
            methods.join("\n")
        ));
    }

    // 5. Search type aliases
    if let Some(ta) = program.type_aliases.iter().find(|t| t.name == name) {
        let pub_str = if ta.is_pub { "pub " } else { "" };
        let type_params = format_generic_params(&ta.type_params);
        return Some(format!(
            "```rust\n{}type {}{} = {};\n```",
            pub_str,
            name,
            type_params,
            type_to_string(&ta.aliased_type)
        ));
    }

    // 6. Search impl methods
    for imp in &program.impls {
        if let Some(method) = imp.methods.iter().find(|m| m.name == name) {
            let params: Vec<String> = method
                .params
                .iter()
                .map(|p| format!("{}: {}", p.name, type_to_string(&p.ty)))
                .collect();
            let ret = match &method.return_type {
                Some(ty) => format!(" -> {}", type_to_string(ty)),
                None => "".to_string(),
            };
            let type_name = type_to_string(&imp.impl_type);
            let trait_str = match &imp.trait_name {
                Some(t) => format!("{} for ", t),
                None => "".to_string(),
            };
            let pub_str = if method.is_pub { "pub " } else { "" };
            let type_params = format_generic_params(&method.type_params);
            return Some(format!(
                "```rust\nimpl {} {{\n    {}fn {}{}({}){}\n}}\n```",
                format!("{}{}", trait_str, type_name),
                pub_str,
                name,
                type_params,
                params.join(", "),
                ret
            ));
        }
    }

    // 7. Search modules
    if let Some(m) = program.modules.iter().find(|md| md.name == name) {
        let pub_str = if m.is_pub { "pub " } else { "" };
        return Some(format!("```rust\n{}mod {};\n```", pub_str, name));
    }

    // 8. Recursively search inside submodules
    for m in &program.modules {
        if let Some(ref body) = m.body
            && let Some(res) = get_hover_text_recursive(body, name, struct_context)
        {
            return Some(res);
        }
    }

    None
}

fn get_path_prefix(tokens: &[(Token, Span)], clicked_idx: usize) -> Vec<String> {
    let mut segments = Vec::new();
    let mut idx = clicked_idx;
    while idx >= 2 && tokens[idx - 1].0 == Token::DoubleColon {
        match &tokens[idx - 2].0 {
            Token::Ident(name) => {
                segments.push(name.clone());
            }
            Token::SelfType => {
                segments.push("Self".to_string());
            }
            Token::Self_ => {
                segments.push("self".to_string());
            }
            _ => break,
        }
        idx -= 2;
    }
    segments.reverse();
    segments
}

fn get_struct_context(tokens: &[(Token, Span)], clicked_idx: usize) -> Option<String> {
    let mut brace_depth = 0;
    let mut paren_depth = 0;
    let mut bracket_depth = 0;
    let mut i = clicked_idx;
    while i > 0 {
        i -= 1;
        match &tokens[i].0 {
            Token::RBrace => brace_depth += 1,
            Token::LBrace => {
                if brace_depth == 0 {
                    // Walk backward to find the identifier before LBrace (or before Lt/Gt generic parameters)
                    let mut j = i;
                    let mut lt_depth = 0;
                    while j > 0 {
                        j -= 1;
                        match &tokens[j].0 {
                            Token::Gt => lt_depth += 1,
                            Token::Lt => {
                                if lt_depth > 0 {
                                    lt_depth -= 1;
                                }
                            }
                            Token::Ident(name) if lt_depth == 0 => {
                                return Some(name.clone());
                            }
                            Token::DoubleColon | Token::SelfType | Token::Self_ => {}
                            _ if lt_depth > 0 => {}
                            _ => break,
                        }
                    }
                    break;
                } else {
                    brace_depth -= 1;
                }
            }
            Token::RParen => paren_depth += 1,
            Token::LParen => {
                if paren_depth > 0 {
                    paren_depth -= 1;
                }
            }
            Token::RBracket => bracket_depth += 1,
            Token::LBracket if bracket_depth > 0 => {
                bracket_depth -= 1;
            }
            _ => {}
        }
    }
    None
}

fn resolve_prefix_to_full_path(uses: &[crate::ast::Use], prefix: &[String]) -> Vec<String> {
    if prefix.is_empty() {
        return Vec::new();
    }
    let first = &prefix[0];
    if first == "std" || first == "core" {
        return prefix.to_vec();
    }
    // Search for a matching import
    for u in uses {
        if let Some(last) = u.path.last()
            && last == first
        {
            let mut full_path = u.path.clone();
            full_path.extend_from_slice(&prefix[1..]);
            return full_path;
        }
    }
    prefix.to_vec()
}

fn find_stdlib_entry_by_path<'a>(
    cache: &'a StdlibCache,
    resolved_path: &[String],
) -> Option<(&'a Url, &'a StdlibEntry)> {
    if resolved_path.len() < 2 {
        return None;
    }
    let dir = &resolved_path[0];
    let mod_name = &resolved_path[1];
    if dir != "std" && dir != "core" {
        return None;
    }
    let suffix = format!("/{}/{}.u", dir, mod_name);
    for (url, entry) in &cache.entries {
        if url.path().ends_with(&suffix) {
            return Some((url, entry));
        }
    }
    None
}

#[derive(Debug)]
struct GenericParamDecl {
    name: String,
    span: Span,
}

fn find_matching_brace_or_semicolon(
    tokens: &[(Token, Span)],
    start_idx: usize,
    allow_semicolon: bool,
) -> usize {
    let mut i = start_idx;
    let mut brace_depth = 0;
    let mut has_braces = false;
    while i < tokens.len() {
        match &tokens[i].0 {
            Token::LBrace => {
                has_braces = true;
                brace_depth += 1;
            }
            Token::RBrace => {
                if brace_depth > 0 {
                    brace_depth -= 1;
                    if brace_depth == 0 {
                        return i;
                    }
                }
            }
            Token::Semicolon if allow_semicolon && brace_depth == 0 && !has_braces => {
                return i;
            }
            _ => {}
        }
        i += 1;
    }
    tokens.len().saturating_sub(1)
}

fn parse_generic_params(tokens: &[(Token, Span)], lt_idx: usize) -> (Vec<GenericParamDecl>, usize) {
    let mut params = Vec::new();
    if lt_idx >= tokens.len() || tokens[lt_idx].0 != Token::Lt {
        return (params, lt_idx);
    }
    let mut i = lt_idx + 1;
    let mut depth = 1;
    let mut new_param = true;
    while i < tokens.len() && depth > 0 {
        match &tokens[i].0 {
            Token::Lt => {
                depth += 1;
                new_param = false;
            }
            Token::Gt => {
                depth -= 1;
                if depth == 0 {
                    break;
                }
                new_param = false;
            }
            Token::Comma if depth == 1 => {
                new_param = true;
            }
            Token::Const if depth == 1 && new_param => {
                if i + 1 < tokens.len()
                    && let Token::Ident(name) = &tokens[i + 1].0
                {
                    params.push(GenericParamDecl {
                        name: name.clone(),
                        span: tokens[i + 1].1,
                    });
                }
                new_param = false;
            }
            Token::Ident(name) if depth == 1 && new_param => {
                params.push(GenericParamDecl {
                    name: name.clone(),
                    span: tokens[i].1,
                });
                new_param = false;
            }
            _ => {}
        }
        i += 1;
    }
    (params, i)
}

fn collect_generic_scopes(tokens: &[(Token, Span)]) -> Vec<(usize, usize, Vec<GenericParamDecl>)> {
    let mut scopes = Vec::new();
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i].0 {
            Token::Impl | Token::Struct | Token::Enum | Token::Trait | Token::Fn | Token::Type => {
                let start_idx = i;
                let allow_semicolon = matches!(&tokens[start_idx].0, Token::Fn | Token::Type);
                let mut lt_idx = None;
                let mut j = i + 1;
                while j < tokens.len() {
                    match &tokens[j].0 {
                        Token::Lt => {
                            lt_idx = Some(j);
                            break;
                        }
                        Token::LBrace => {
                            break;
                        }
                        Token::Semicolon if allow_semicolon => {
                            break;
                        }
                        _ => {}
                    }
                    j += 1;
                }

                let mut params = Vec::new();
                if let Some(idx) = lt_idx {
                    let (parsed, _) = parse_generic_params(tokens, idx);
                    params = parsed;
                }

                let end_idx = find_matching_brace_or_semicolon(tokens, start_idx, allow_semicolon);
                scopes.push((start_idx, end_idx, params));
            }
            _ => {}
        }
        i += 1;
    }
    scopes
}

fn find_definition_span(
    tokens: &[(Token, Span)],
    name: &str,
    hover_offset: usize,
    struct_context: Option<&str>,
) -> Option<Span> {
    // 1. Local backward search for variables or parameters
    let hover_index = tokens
        .iter()
        .position(|(_, span)| span.lo <= hover_offset && hover_offset < span.hi);
    if let Some(idx) = hover_index {
        let is_member_access = idx > 0 && tokens[idx - 1].0 == Token::Dot;
        let is_path_access = idx > 0 && tokens[idx - 1].0 == Token::DoubleColon;
        let is_struct_field_name =
            struct_context.is_some() && idx + 1 < tokens.len() && tokens[idx + 1].0 == Token::Colon;

        if !is_member_access && !is_path_access && !is_struct_field_name {
            let mut i = idx;
            while i > 0 {
                i -= 1;
                if i + 1 < tokens.len() {
                    if let Token::Ident(ref id) = tokens[i].0
                        && id == name
                        && i > 0
                        && (tokens[i - 1].0 == Token::Let || tokens[i - 1].0 == Token::Mut)
                    {
                        return Some(tokens[i].1);
                    }
                    if let Token::Ident(ref id) = tokens[i + 1].0
                        && id == name
                        && tokens[i].0 == Token::Let
                    {
                        return Some(tokens[i + 1].1);
                    }
                }
                if tokens[i].0 == Token::Fn {
                    // Check parameters of this enclosing function boundary
                    let mut j = i + 1;
                    let mut in_params = false;
                    while j < tokens.len()
                        && tokens[j].0 != Token::LBrace
                        && tokens[j].0 != Token::Semicolon
                    {
                        if tokens[j].0 == Token::LParen {
                            in_params = true;
                        } else if tokens[j].0 == Token::RParen {
                            break;
                        } else if in_params
                            && let Token::Ident(ref id) = tokens[j].0
                            && id == name
                        {
                            return Some(tokens[j].1);
                        }
                        j += 1;
                    }
                    break; // Stop local search at the function definition boundary
                }
            }
        }
    }

    // 1b. Generic parameters search in enclosing scopes
    if let Some(idx) = hover_index {
        let scopes = collect_generic_scopes(tokens);
        let mut matching_scopes = Vec::new();
        for (start, end, params) in scopes {
            if start <= idx && idx <= end {
                matching_scopes.push((start, end, params));
            }
        }
        // Sort by scope range size (end - start) ascending so innermost scopes come first
        matching_scopes.sort_by_key(|(start, end, _)| end - start);
        for (_, _, params) in matching_scopes {
            if let Some(param) = params.iter().find(|p| p.name == name) {
                return Some(param.span);
            }
        }
    }

    // 2. Global search for top-level constructs
    let mut i = 0;
    while i < tokens.len() {
        match &tokens[i].0 {
            Token::Fn | Token::Trait | Token::Type | Token::Mod => {
                if i + 1 < tokens.len()
                    && let Token::Ident(ref id) = tokens[i + 1].0
                    && id == name
                {
                    return Some(tokens[i + 1].1);
                }
            }
            Token::Struct => {
                let struct_name = if i + 1 < tokens.len() {
                    if let Token::Ident(ref id) = tokens[i + 1].0 {
                        Some(id.as_str())
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(s_name) = struct_name
                    && s_name == name
                {
                    return Some(tokens[i + 1].1);
                }

                // Scan fields of the struct only if there is no struct_context OR the struct_name matches struct_context
                let matches_context = match struct_context {
                    Some(ctx) => struct_name == Some(ctx),
                    None => true,
                };

                if matches_context {
                    // Scan fields of the struct
                    let mut j = i + 1;
                    while j < tokens.len() && tokens[j].0 != Token::LBrace {
                        j += 1;
                    }
                    if j < tokens.len() {
                        j += 1; // move past LBrace
                        let mut brace_depth = 1;
                        let mut paren_depth = 0;
                        let mut bracket_depth = 0;
                        let mut lt_depth = 0;
                        while j < tokens.len() && brace_depth > 0 {
                            match &tokens[j].0 {
                                Token::LBrace => brace_depth += 1,
                                Token::RBrace => brace_depth -= 1,
                                Token::LParen => paren_depth += 1,
                                Token::RParen => paren_depth -= 1,
                                Token::LBracket => bracket_depth += 1,
                                Token::RBracket => bracket_depth -= 1,
                                Token::Lt => lt_depth += 1,
                                Token::Gt => lt_depth -= 1,
                                Token::Ident(id)
                                    if brace_depth == 1
                                        && paren_depth == 0
                                        && bracket_depth == 0
                                        && lt_depth == 0
                                        && id == name =>
                                {
                                    return Some(tokens[j].1);
                                }
                                _ => {}
                            }
                            j += 1;
                        }
                    }
                }
            }
            Token::Enum => {
                let enum_name = if i + 1 < tokens.len() {
                    if let Token::Ident(ref id) = tokens[i + 1].0 {
                        Some(id.as_str())
                    } else {
                        None
                    }
                } else {
                    None
                };

                if let Some(e_name) = enum_name
                    && e_name == name
                {
                    return Some(tokens[i + 1].1);
                }

                // Scan variants of the enum only if there is no struct_context OR the enum_name matches struct_context
                let matches_context = match struct_context {
                    Some(ctx) => enum_name == Some(ctx),
                    None => true,
                };

                if matches_context {
                    // Scan variants of the enum
                    let mut j = i + 1;
                    while j < tokens.len() && tokens[j].0 != Token::LBrace {
                        j += 1;
                    }
                    if j < tokens.len() {
                        j += 1; // move past LBrace
                        let mut brace_depth = 1;
                        let mut paren_depth = 0;
                        let mut bracket_depth = 0;
                        let mut lt_depth = 0;
                        while j < tokens.len() && brace_depth > 0 {
                            match &tokens[j].0 {
                                Token::LBrace => brace_depth += 1,
                                Token::RBrace => brace_depth -= 1,
                                Token::LParen => paren_depth += 1,
                                Token::RParen => paren_depth -= 1,
                                Token::LBracket => bracket_depth += 1,
                                Token::RBracket => bracket_depth -= 1,
                                Token::Lt => lt_depth += 1,
                                Token::Gt => lt_depth -= 1,
                                Token::Ident(id)
                                    if brace_depth == 1
                                        && paren_depth == 0
                                        && bracket_depth == 0
                                        && lt_depth == 0
                                        && id == name =>
                                {
                                    return Some(tokens[j].1);
                                }
                                _ => {}
                            }
                            j += 1;
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
        documents.insert(
            url,
            DocumentState {
                source: text,
                last_valid_program: valid_prog,
            },
        );
    }

    diags
}

fn publish_diagnostics(
    connection: &Connection,
    url: Url,
    diagnostics: Vec<Diagnostic>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let uri = url.to_string().parse::<Uri>().unwrap();
    let params = PublishDiagnosticsParams::new(uri, diagnostics, None);
    let notification = Notification::new("textDocument/publishDiagnostics".to_string(), params);
    connection
        .sender
        .send(Message::Notification(notification))?;
    Ok(())
}

fn handle_hover(
    documents: &HashMap<Url, DocumentState>,
    stdlib_cache: &StdlibCache,
    params: HoverParams,
) -> Option<Hover> {
    let uri = params.text_document_position_params.text_document.uri;
    let url = Url::parse(uri.as_str()).unwrap();
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

    let hovered_idx = tokens
        .iter()
        .position(|(_, span)| span.lo <= offset && offset < span.hi)?;
    let hovered_token = &tokens[hovered_idx];

    if let Token::Ident(ref name) = hovered_token.0 {
        let struct_context = get_struct_context(&tokens, hovered_idx);

        // 1. Try to resolve path prefix if one exists
        let path_prefix = get_path_prefix(&tokens, hovered_idx);
        if !path_prefix.is_empty() {
            let mut resolved_path = path_prefix.clone();
            if let Some(prog) = &doc.last_valid_program {
                resolved_path = resolve_prefix_to_full_path(&prog.uses, &path_prefix);
            }
            if let Some((_, target_entry)) = find_stdlib_entry_by_path(stdlib_cache, &resolved_path)
                && let Some(contents) = get_hover_text_from_program(
                    &target_entry.program,
                    name,
                    struct_context.as_deref(),
                )
            {
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

        // 2. Fallback: Search locally in current file
        let hover_text = if let Some(prog) = &doc.last_valid_program {
            get_hover_text_from_program(prog, name, struct_context.as_deref())
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

        // 3. Fallback: Search globally in the standard library cache
        if let Some(stdlib_text) =
            get_hover_text_from_stdlib(stdlib_cache, name, struct_context.as_deref())
        {
            return Some(Hover {
                contents: HoverContents::Markup(MarkupContent {
                    kind: MarkupKind::Markdown,
                    value: stdlib_text,
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

fn find_stdlib_root() -> std::path::PathBuf {
    let check_dir = |path: std::path::PathBuf| -> Option<std::path::PathBuf> {
        if path.is_dir() {
            Some(std::fs::canonicalize(&path).unwrap_or(path))
        } else {
            None
        }
    };

    // 1. Try $ULANG_ROOT
    if let Ok(val) = std::env::var("ULANG_ROOT") {
        let root_path = std::path::PathBuf::from(val);
        if let Some(p) = check_dir(root_path.join("stdlib")) {
            return p;
        }
        if let Some(p) = check_dir(root_path.join("root").join("stdlib")) {
            return p;
        }
        if let Some(p) = check_dir(root_path.clone()) {
            return p;
        }
    }

    // 2. Try ./root
    let root_path = std::path::PathBuf::from("root");
    if let Some(p) = check_dir(root_path.join("stdlib")) {
        return p;
    }
    if let Some(p) = check_dir(root_path) {
        return p;
    }

    // 3. Try /usr/share/ulang/
    let share_path = std::path::PathBuf::from("/usr/share/ulang");
    if let Some(p) = check_dir(share_path.join("stdlib")) {
        return p;
    }
    if let Some(p) = check_dir(share_path) {
        return p;
    }

    // 4. Fails
    eprintln!("error: standard library (stdlib) not found.");
    eprintln!("Please specify the location using the ULANG_ROOT environment variable,");
    eprintln!("or ensure standard library is present at './root' or '/usr/share/ulang/'.");
    std::process::exit(1);
}

/// Search for a definition in the stdlib cache.
fn find_stdlib_definition(
    cache: &StdlibCache,
    name: &str,
    struct_context: Option<&str>,
) -> Option<(Url, Range)> {
    for (url, entry) in &cache.entries {
        if get_hover_text_from_program(&entry.program, name, struct_context).is_some() {
            // Token-based search for exact span
            let mut lexer = Lexer::new(&entry.source);
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
            if let Some(def_span) = find_definition_span(&tokens, name, 0, struct_context) {
                let start = offset_to_position(&entry.source, def_span.lo);
                let end = offset_to_position(&entry.source, def_span.hi);
                let range = Range::new(start, end);
                return Some((url.clone(), range));
            }
        }
    }
    None
}

/// Search for hover text in the stdlib cache.
fn get_hover_text_from_stdlib(
    cache: &StdlibCache,
    name: &str,
    struct_context: Option<&str>,
) -> Option<String> {
    for (_, entry) in &cache.entries {
        if let Some(text) = get_hover_text_recursive(&entry.program, name, struct_context) {
            return Some(text);
        }
    }
    None
}

fn handle_definition(
    documents: &HashMap<Url, DocumentState>,
    stdlib_cache: &StdlibCache,
    params: GotoDefinitionParams,
) -> Option<GotoDefinitionResponse> {
    let uri = params.text_document_position_params.text_document.uri;
    let url = Url::parse(uri.as_str()).unwrap();
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

    let clicked_idx = tokens
        .iter()
        .position(|(_, span)| span.lo <= offset && offset < span.hi)?;
    let clicked_token = &tokens[clicked_idx];

    if let Token::Ident(ref name) = clicked_token.0 {
        let struct_context = get_struct_context(&tokens, clicked_idx);

        // 1. Try to resolve path prefix if one exists
        let path_prefix = get_path_prefix(&tokens, clicked_idx);
        if !path_prefix.is_empty() {
            let mut resolved_path = path_prefix.clone();
            if let Some(prog) = &doc.last_valid_program {
                resolved_path = resolve_prefix_to_full_path(&prog.uses, &path_prefix);
            }
            if let Some((target_url, target_entry)) =
                find_stdlib_entry_by_path(stdlib_cache, &resolved_path)
            {
                // Token-based search in target stdlib entry
                let mut target_lexer = Lexer::new(&target_entry.source);
                let mut target_tokens = Vec::new();
                loop {
                    match target_lexer.next_token() {
                        Ok((token, span)) => {
                            target_tokens.push((token, span));
                            if let Token::Eof = target_tokens.last().unwrap().0 {
                                break;
                            }
                        }
                        Err(_) => break,
                    }
                }
                if let Some(def_span) =
                    find_definition_span(&target_tokens, name, 0, struct_context.as_deref())
                {
                    let start = offset_to_position(&target_entry.source, def_span.lo);
                    let end = offset_to_position(&target_entry.source, def_span.hi);
                    let range = Range::new(start, end);
                    let file_uri = target_url.to_string().parse::<Uri>().unwrap();
                    return Some(GotoDefinitionResponse::Scalar(Location::new(
                        file_uri, range,
                    )));
                }
            }
        }

        // 2. Fallback: Search locally in the current file
        if let Some(def_span) =
            find_definition_span(&tokens, name, offset, struct_context.as_deref())
        {
            let start = offset_to_position(&doc.source, def_span.lo);
            let end = offset_to_position(&doc.source, def_span.hi);
            let range = Range::new(start, end);
            return Some(GotoDefinitionResponse::Scalar(Location::new(uri, range)));
        }

        // 3. Fallback: Search globally in the standard library cache
        if let Some((file_url, range)) =
            find_stdlib_definition(stdlib_cache, name, struct_context.as_deref())
        {
            let file_uri = file_url.to_string().parse::<Uri>().unwrap();
            return Some(GotoDefinitionResponse::Scalar(Location::new(
                file_uri, range,
            )));
        }
    }

    None
}

fn cast_request<R>(
    req: Request,
) -> Result<(lsp_server::RequestId, R::Params), Box<dyn std::error::Error + Send + Sync>>
where
    R: lsp_types::request::Request,
    R::Params: serde::de::DeserializeOwned,
{
    req.extract(R::METHOD)
        .map_err(|e| format!("failed to extract request: {:?}", e).into())
}

fn cast_notification<N>(
    not: Notification,
) -> Result<N::Params, Box<dyn std::error::Error + Send + Sync>>
where
    N: lsp_types::notification::Notification,
    N::Params: serde::de::DeserializeOwned,
{
    not.extract(N::METHOD)
        .map_err(|e| format!("failed to extract notification: {:?}", e).into())
}

// ---------------------------------------------------------------------------
// Completion infrastructure
// ---------------------------------------------------------------------------

/// Cached parsed project files for cross-file completion.
struct ProjectCache {
    entries: Vec<(Url, StdlibEntry)>,
}

fn build_project_cache(
    workspace_root: &std::path::Path,
    documents: &HashMap<Url, DocumentState>,
) -> ProjectCache {
    let mut entries = Vec::new();
    let mut paths = Vec::new();
    collect_u_files(workspace_root, workspace_root, &mut paths);

    // Compute stdlib root to exclude stdlib files from project cache
    let stdlib_root = std::fs::canonicalize(find_stdlib_root()).ok();

    for path in &paths {
        // Skip files that belong to the standard library
        if let Some(ref stdlib) = stdlib_root
            && let Ok(canonical_path) = std::fs::canonicalize(path)
            && canonical_path.starts_with(stdlib)
        {
            continue;
        }

        if let Ok(url) = Url::from_file_path(path) {
            // If the file is open in the editor, use the editor's source.
            // Otherwise read from disk.
            let source = if let Some(doc) = documents.get(&url) {
                doc.source.clone()
            } else if let Ok(src) = std::fs::read_to_string(path) {
                src
            } else {
                continue;
            };

            let mut lexer = Lexer::new(&source);
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
            if !tokens.is_empty() && matches!(tokens.last().unwrap().0, Token::Eof) {
                let mut parser = Parser::new(&tokens);
                if let Ok(program) = parser.parse_program() {
                    entries.push((url, StdlibEntry { source, program }));
                }
            }
        }
    }

    ProjectCache { entries }
}

fn collect_u_files(
    root: &std::path::Path,
    dir: &std::path::Path,
    out: &mut Vec<std::path::PathBuf>,
) {
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                // Skip target/ directories (build artifacts)
                if let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && (name == "target" || name.starts_with('.'))
                {
                    continue;
                }
                collect_u_files(root, &path, out);
            } else if path.extension().and_then(|e| e.to_str()) == Some("u") {
                out.push(path);
            }
        }
    }
}

/// Extract the module path for a project file URL relative to the workspace root.
/// E.g., `src/foo.u` -> `["foo"]`, `src/sub/bar.u` -> `["sub", "bar"]`.
fn project_file_module_path(workspace_root: &std::path::Path, url: &Url) -> Vec<String> {
    let file_path = match url.to_file_path() {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };
    let relative = match file_path.strip_prefix(workspace_root) {
        Ok(p) => p,
        Err(_) => return Vec::new(),
    };

    let mut segments: Vec<String> = relative
        .components()
        .map(|c| c.as_os_str().to_string_lossy().to_string())
        .collect();

    // Strip `.u` extension from last segment
    if let Some(last) = segments.last_mut()
        && last.ends_with(".u")
    {
        *last = last[..last.len() - 2].to_string();
    }

    segments
}

/// Extract the stdlib module path from a cache entry URL.
/// E.g., `/path/to/stdlib/core/option.u` -> `["core", "option"]`.
fn stdlib_module_path(url: &Url) -> Vec<String> {
    let path = url.path();
    let segments: Vec<&str> = path.split('/').filter(|s| !s.is_empty()).collect();

    // Find "core" or "std" in the path and take from there
    for (i, seg) in segments.iter().enumerate() {
        if *seg == "core" || *seg == "std" {
            let mut result = Vec::new();
            for j in i..segments.len() {
                let s = segments[j];
                if s.ends_with(".u") {
                    result.push(s[..s.len() - 2].to_string());
                } else {
                    result.push(s.to_string());
                }
            }
            return result;
        }
    }
    Vec::new()
}

/// Get the word prefix at a given byte offset (what the user has typed so far).
fn get_word_prefix_at_offset(tokens: &[(Token, Span)], source: &str, offset: usize) -> String {
    for (tok, span) in tokens {
        if span.lo <= offset && offset <= span.hi {
            if let Token::Ident(_name) = tok {
                let byte_end = offset.min(span.hi).min(source.len());
                let byte_start = span.lo.min(source.len());
                if byte_end >= byte_start {
                    return source[byte_start..byte_end].to_string();
                }
            }
            return String::new();
        }
    }
    String::new()
}

/// Get the path prefix at a given byte offset (segments before `::`).
fn get_path_prefix_at_offset(tokens: &[(Token, Span)], offset: usize) -> Vec<String> {
    // Find the token that contains the cursor, or the last token before cursor
    let mut cursor_idx = tokens.len();
    for (i, (_, span)) in tokens.iter().enumerate() {
        if span.lo <= offset && offset <= span.hi {
            cursor_idx = i;
            break;
        }
        if span.lo > offset {
            cursor_idx = i;
            break;
        }
    }
    if cursor_idx == 0 {
        return Vec::new();
    }

    // Check if the token just before cursor is `::` (path separator)
    let before_cursor = if cursor_idx >= tokens.len() {
        tokens.last().map(|_| tokens.len() - 1)
    } else if cursor_idx > 0 {
        Some(cursor_idx - 1)
    } else {
        None
    };

    // Determine the last identifier index in the path
    let ident_idx = match before_cursor {
        Some(i) if tokens[i].0 == Token::DoubleColon && i > 0 => {
            // Cursor is after `::`, the path includes the segment before `::`
            match &tokens[i - 1].0 {
                Token::Ident(_) | Token::SelfType | Token::Self_ => i - 1,
                _ => return Vec::new(),
            }
        }
        Some(i)
            if matches!(
                &tokens[i].0,
                Token::Ident(_) | Token::SelfType | Token::Self_
            ) =>
        {
            // Check if there's a `::` before this ident
            if i > 0 && tokens[i - 1].0 == Token::DoubleColon {
                i
            } else {
                return Vec::new();
            }
        }
        _ => return Vec::new(),
    };

    // Walk backward collecting path segments, including the ident at ident_idx
    let mut segments = Vec::new();
    let mut i = ident_idx;
    loop {
        match &tokens[i].0 {
            Token::Ident(name) => segments.push(name.clone()),
            Token::SelfType => segments.push("Self".to_string()),
            Token::Self_ => segments.push("self".to_string()),
            _ => break,
        }
        if i >= 2 && tokens[i - 1].0 == Token::DoubleColon {
            i -= 2;
        } else {
            break;
        }
    }
    segments.reverse();
    segments
}

/// Find the position to insert a `use` statement in a document.
/// Returns a Position (0-indexed line/col) where the `use` should be inserted.
fn find_use_insertion_position(doc: &DocumentState) -> Position {
    if let Some(ref prog) = doc.last_valid_program
        && let Some(last_use) = prog.uses.last()
    {
        // Insert after the newline following the last use statement
        let mut offset = last_use.span.hi;
        let source_bytes = doc.source.as_bytes();
        while offset < source_bytes.len() && source_bytes[offset] == b'\n' {
            offset += 1;
        }
        return offset_to_position(&doc.source, offset);
    }
    // No uses: insert at the beginning of the file
    Position::new(0, 0)
}

/// Check if a symbol is already imported via `use` declarations.
fn is_already_imported(
    uses: &[crate::ast::Use],
    module_path: &[String],
    symbol_name: &str,
) -> bool {
    for u in uses {
        if u.path.last().map(|s| s.as_str()) == Some(symbol_name) {
            // For simple imports like `use std::option::Option;`, the path
            // includes the symbol name. Check that the module path matches.
            if u.path.len() >= 2 {
                let import_module = &u.path[..u.path.len() - 1];
                if import_module == module_path {
                    return true;
                }
            }
        }
    }
    false
}

/// Check if a core module is re-exported by `std`.
fn is_core_reexported_by_std(stdlib_cache: &StdlibCache, core_name: &str) -> bool {
    for (url, entry) in &stdlib_cache.entries {
        if url.path().ends_with("/std/mod.u") {
            for u in &entry.program.uses {
                if u.path.len() == 2 && u.path[0] == "core" && u.path[1] == core_name {
                    return true;
                }
            }
            return false;
        }
    }
    false
}

/// Build a CompletionItem for a symbol from the stdlib or project cache.
fn build_imported_completion_item(
    name: &str,
    kind: CompletionItemKind,
    detail: String,
    sort_prefix: &str,
    insert_use: bool,
    use_text: &str,
) -> CompletionItem {
    let mut item = CompletionItem {
        label: name.to_string(),
        kind: Some(kind),
        detail: Some(detail),
        sort_text: Some(format!("{}{}", sort_prefix, name)),
        filter_text: Some(name.to_string()),
        ..Default::default()
    };
    if insert_use {
        item.additional_text_edits = Some(vec![TextEdit {
            range: Range::new(Position::new(0, 0), Position::new(0, 0)), // placeholder, replaced later
            new_text: use_text.to_string(),
        }]);
    }
    item
}

fn collect_local_symbols(
    items: &mut Vec<CompletionItem>,
    doc: &DocumentState,
    word_prefix: &str,
    cursor_offset: usize,
) {
    let Some(ref prog) = doc.last_valid_program else {
        return;
    };

    let prefix_lower = word_prefix.to_lowercase();

    // Lex the source for variable/param scanning
    let mut tokens = Vec::new();
    let mut lexer = Lexer::new(&doc.source);
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

    // 1. Functions (top-level and methods)
    let mut seen_funcs = std::collections::HashSet::new();
    for func in &prog.funcs {
        if func.name.to_lowercase().starts_with(&prefix_lower) && seen_funcs.insert(&func.name) {
            let params: Vec<String> = func
                .params
                .iter()
                .map(|p| format!("{}: {}", p.name, type_to_string(&p.ty)))
                .collect();
            let ret = match &func.return_type {
                Some(ty) => format!(" -> {}", type_to_string(ty)),
                None => String::new(),
            };
            items.push(CompletionItem {
                label: func.name.clone(),
                kind: Some(CompletionItemKind::FUNCTION),
                detail: Some(format!("fn({}){}", params.join(", "), ret)),
                sort_text: Some(format!("1_{}", func.name)),
                insert_text: Some(format!("{}()", func.name)),
                insert_text_format: Some(InsertTextFormat::SNIPPET),
                ..Default::default()
            });
        }
    }

    // Methods from impl blocks
    let mut seen_methods = std::collections::HashSet::new();
    for imp in &prog.impls {
        for method in &imp.methods {
            if method.name.to_lowercase().starts_with(&prefix_lower)
                && seen_methods.insert(&method.name)
            {
                let params: Vec<String> = method
                    .params
                    .iter()
                    .map(|p| format!("{}: {}", p.name, type_to_string(&p.ty)))
                    .collect();
                let ret = match &method.return_type {
                    Some(ty) => format!(" -> {}", type_to_string(ty)),
                    None => String::new(),
                };
                items.push(CompletionItem {
                    label: method.name.clone(),
                    kind: Some(CompletionItemKind::METHOD),
                    detail: Some(format!("fn({}){}", params.join(", "), ret)),
                    sort_text: Some(format!("1_{}", method.name)),
                    ..Default::default()
                });
            }
        }
    }

    // 2. Structs
    for st in &prog.structs {
        if st.name.to_lowercase().starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: st.name.clone(),
                kind: Some(CompletionItemKind::STRUCT),
                detail: Some(format!("struct {}", st.name)),
                sort_text: Some(format!("1_{}", st.name)),
                ..Default::default()
            });
        }
    }

    // 3. Enums
    for en in &prog.enums {
        if en.name.to_lowercase().starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: en.name.clone(),
                kind: Some(CompletionItemKind::ENUM),
                detail: Some(format!("enum {}", en.name)),
                sort_text: Some(format!("1_{}", en.name)),
                ..Default::default()
            });
        }
    }

    // 4. Traits
    for tr in &prog.traits {
        if tr.name.to_lowercase().starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: tr.name.clone(),
                kind: Some(CompletionItemKind::INTERFACE),
                detail: Some(format!("trait {}", tr.name)),
                sort_text: Some(format!("1_{}", tr.name)),
                ..Default::default()
            });
        }
    }

    // 5. Type aliases
    for ta in &prog.type_aliases {
        if ta.name.to_lowercase().starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: ta.name.clone(),
                kind: Some(CompletionItemKind::TYPE_PARAMETER),
                detail: Some(format!(
                    "type {} = {}",
                    ta.name,
                    type_to_string(&ta.aliased_type)
                )),
                sort_text: Some(format!("1_{}", ta.name)),
                ..Default::default()
            });
        }
    }

    // 6. Modules
    for m in &prog.modules {
        if m.name.to_lowercase().starts_with(&prefix_lower) {
            items.push(CompletionItem {
                label: m.name.clone(),
                kind: Some(CompletionItemKind::MODULE),
                detail: Some(format!("mod {}", m.name)),
                sort_text: Some(format!("1_{}", m.name)),
                ..Default::default()
            });
        }
    }

    // 7. Let/Const bindings and function params in scope
    // Scan backward from cursor to find variables in scope
    let cursor_idx = tokens
        .iter()
        .position(|(_, span)| span.lo <= cursor_offset && cursor_offset < span.hi)
        .or_else(|| {
            // If cursor is not on a token, find the last token before cursor
            tokens
                .iter()
                .rposition(|(_, span)| span.hi <= cursor_offset)
        });

    if let Some(end_idx) = cursor_idx {
        let mut seen = std::collections::HashSet::new();
        let mut i = end_idx;

        while i > 0 {
            i -= 1;
            match &tokens[i].0 {
                Token::Let | Token::Const => {
                    // Find the name after Let/Const (skip `mut` if present)
                    let mut j = i + 1;
                    while j < tokens.len() && j <= end_idx {
                        match &tokens[j].0 {
                            Token::Mut => {
                                j += 1;
                                continue;
                            }
                            Token::Ident(name) => {
                                if name != "_"
                                    && name.to_lowercase().starts_with(&prefix_lower)
                                    && seen.insert(name.clone())
                                {
                                    items.push(CompletionItem {
                                        label: name.clone(),
                                        kind: Some(CompletionItemKind::VARIABLE),
                                        detail: Some(format!("let {}", name)),
                                        sort_text: Some(format!("1_{}", name)),
                                        ..Default::default()
                                    });
                                }
                                break;
                            }
                            Token::Underscore => break,
                            _ => break,
                        }
                    }
                }
                Token::Fn => {
                    // Collect function parameters
                    let mut j = i + 1;
                    let mut in_params = false;
                    while j < tokens.len() && j <= end_idx {
                        match &tokens[j].0 {
                            Token::LParen => in_params = true,
                            Token::RParen => break,
                            Token::Ident(name)
                                if in_params
                                    && name != "self"
                                    && name.to_lowercase().starts_with(&prefix_lower)
                                    && seen.insert(name.clone()) =>
                            {
                                items.push(CompletionItem {
                                    label: name.clone(),
                                    kind: Some(CompletionItemKind::VARIABLE),
                                    detail: Some(format!("param {}", name)),
                                    sort_text: Some(format!("1_{}", name)),
                                    ..Default::default()
                                });
                            }
                            _ => {}
                        }
                        j += 1;
                    }
                    break; // Stop at function boundary
                }
                _ => {}
            }
        }
    }
}

fn collect_project_symbols(
    items: &mut Vec<CompletionItem>,
    project_cache: &ProjectCache,
    uses: &[crate::ast::Use],
    workspace_root: &std::path::Path,
    word_prefix: &str,
    path_prefix: &[String],
) {
    let prefix_lower = word_prefix.to_lowercase();

    for (url, entry) in &project_cache.entries {
        let module_path = project_file_module_path(workspace_root, url);
        if module_path.is_empty() {
            continue;
        }

        // If path_prefix is non-empty, only process entries whose module path
        // starts with the resolved prefix.
        if !path_prefix.is_empty()
            && (module_path.len() < path_prefix.len()
                || module_path[..path_prefix.len()] != *path_prefix)
        {
            continue;
        }

        // Collect pub symbols
        let mut seen_funcs = std::collections::HashSet::new();
        for func in &entry.program.funcs {
            if !func.is_pub
                || !func.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_funcs.insert(&func.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &module_path, &func.name);
            let use_decl = format!("use {}::{};\n", module_path.join("::"), func.name);
            items.push(build_imported_completion_item(
                &func.name,
                CompletionItemKind::FUNCTION,
                module_path.join("::"),
                "2_",
                !already_imported,
                &use_decl,
            ));
        }

        let mut seen_structs = std::collections::HashSet::new();
        for st in &entry.program.structs {
            if !st.is_pub
                || !st.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_structs.insert(&st.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &module_path, &st.name);
            let use_decl = format!("use {}::{};\n", module_path.join("::"), st.name);
            items.push(build_imported_completion_item(
                &st.name,
                CompletionItemKind::STRUCT,
                module_path.join("::"),
                "2_",
                !already_imported,
                &use_decl,
            ));
        }

        let mut seen_enums = std::collections::HashSet::new();
        for en in &entry.program.enums {
            if !en.is_pub
                || !en.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_enums.insert(&en.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &module_path, &en.name);
            let use_decl = format!("use {}::{};\n", module_path.join("::"), en.name);
            items.push(build_imported_completion_item(
                &en.name,
                CompletionItemKind::ENUM,
                module_path.join("::"),
                "2_",
                !already_imported,
                &use_decl,
            ));
        }

        let mut seen_traits = std::collections::HashSet::new();
        for tr in &entry.program.traits {
            if !tr.is_pub
                || !tr.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_traits.insert(&tr.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &module_path, &tr.name);
            let use_decl = format!("use {}::{};\n", module_path.join("::"), tr.name);
            items.push(build_imported_completion_item(
                &tr.name,
                CompletionItemKind::INTERFACE,
                module_path.join("::"),
                "2_",
                !already_imported,
                &use_decl,
            ));
        }

        let mut seen_aliases = std::collections::HashSet::new();
        for ta in &entry.program.type_aliases {
            if !ta.is_pub
                || !ta.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_aliases.insert(&ta.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &module_path, &ta.name);
            let use_decl = format!("use {}::{};\n", module_path.join("::"), ta.name);
            items.push(build_imported_completion_item(
                &ta.name,
                CompletionItemKind::TYPE_PARAMETER,
                module_path.join("::"),
                "2_",
                !already_imported,
                &use_decl,
            ));
        }
    }
}

fn collect_stdlib_symbols(
    items: &mut Vec<CompletionItem>,
    stdlib_cache: &StdlibCache,
    uses: &[crate::ast::Use],
    word_prefix: &str,
    path_prefix: &[String],
) {
    let prefix_lower = word_prefix.to_lowercase();

    for (url, entry) in &stdlib_cache.entries {
        // Skip mod.u files (module index files)
        if url.path().ends_with("/mod.u") {
            continue;
        }

        let module_path = stdlib_module_path(url);
        if module_path.len() < 2 {
            continue;
        }

        // If path_prefix is non-empty, only process entries whose module path starts with it
        if !path_prefix.is_empty()
            && (module_path.len() < path_prefix.len()
                || module_path[..path_prefix.len()] != *path_prefix)
        {
            continue;
        }

        // Determine the `use` prefix: for core modules re-exported by std, use `std::`
        let use_prefix = if module_path[0] == "core"
            && is_core_reexported_by_std(stdlib_cache, &module_path[1])
        {
            let mut p = vec!["std".to_string()];
            p.extend_from_slice(&module_path[1..]);
            p
        } else {
            module_path.clone()
        };

        // Collect pub symbols
        let mut seen_funcs = std::collections::HashSet::new();
        for func in &entry.program.funcs {
            if !func.is_pub
                || !func.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_funcs.insert(&func.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &use_prefix, &func.name);
            let use_decl = format!("use {}::{};\n", use_prefix.join("::"), func.name);
            items.push(build_imported_completion_item(
                &func.name,
                CompletionItemKind::FUNCTION,
                use_prefix.join("::"),
                "3_",
                !already_imported,
                &use_decl,
            ));
        }

        let mut seen_structs = std::collections::HashSet::new();
        for st in &entry.program.structs {
            if !st.is_pub
                || !st.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_structs.insert(&st.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &use_prefix, &st.name);
            let use_decl = format!("use {}::{};\n", use_prefix.join("::"), st.name);
            items.push(build_imported_completion_item(
                &st.name,
                CompletionItemKind::STRUCT,
                use_prefix.join("::"),
                "3_",
                !already_imported,
                &use_decl,
            ));
        }

        let mut seen_enums = std::collections::HashSet::new();
        for en in &entry.program.enums {
            if !en.is_pub
                || !en.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_enums.insert(&en.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &use_prefix, &en.name);
            let use_decl = format!("use {}::{};\n", use_prefix.join("::"), en.name);
            items.push(build_imported_completion_item(
                &en.name,
                CompletionItemKind::ENUM,
                use_prefix.join("::"),
                "3_",
                !already_imported,
                &use_decl,
            ));
        }

        let mut seen_traits = std::collections::HashSet::new();
        for tr in &entry.program.traits {
            if !tr.is_pub
                || !tr.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_traits.insert(&tr.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &use_prefix, &tr.name);
            let use_decl = format!("use {}::{};\n", use_prefix.join("::"), tr.name);
            items.push(build_imported_completion_item(
                &tr.name,
                CompletionItemKind::INTERFACE,
                use_prefix.join("::"),
                "3_",
                !already_imported,
                &use_decl,
            ));
        }

        let mut seen_aliases = std::collections::HashSet::new();
        for ta in &entry.program.type_aliases {
            if !ta.is_pub
                || !ta.name.to_lowercase().starts_with(&prefix_lower)
                || !seen_aliases.insert(&ta.name)
            {
                continue;
            }
            let already_imported = is_already_imported(uses, &use_prefix, &ta.name);
            let use_decl = format!("use {}::{};\n", use_prefix.join("::"), ta.name);
            items.push(build_imported_completion_item(
                &ta.name,
                CompletionItemKind::TYPE_PARAMETER,
                use_prefix.join("::"),
                "3_",
                !already_imported,
                &use_decl,
            ));
        }
    }
}

fn handle_completion(
    documents: &HashMap<Url, DocumentState>,
    stdlib_cache: &StdlibCache,
    workspace_root: &Option<std::path::PathBuf>,
    params: CompletionParams,
) -> Option<CompletionResponse> {
    let uri = params.text_document_position.text_document.uri;
    let url = Url::parse(uri.as_str()).ok()?;
    let doc = documents.get(&url)?;
    let position = params.text_document_position.position;

    let offset = position_to_offset(&doc.source, position);

    let mut tokens = Vec::new();
    let mut lexer = Lexer::new(&doc.source);
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

    let word_prefix = get_word_prefix_at_offset(&tokens, &doc.source, offset);
    let path_prefix = get_path_prefix_at_offset(&tokens, offset);

    // Lazy project cache build
    let project_cache: Option<ProjectCache> = workspace_root
        .as_ref()
        .map(|root| build_project_cache(root, documents));

    // If path prefix, attempt to resolve it
    let resolved_prefix = if !path_prefix.is_empty() {
        if let Some(ref prog) = doc.last_valid_program {
            resolve_prefix_to_full_path(&prog.uses, &path_prefix)
        } else {
            path_prefix.clone()
        }
    } else {
        Vec::new()
    };

    let mut items: Vec<CompletionItem> = Vec::new();

    if resolved_prefix.is_empty() && path_prefix.is_empty() {
        // No path prefix: collect from all sources
        collect_local_symbols(&mut items, doc, &word_prefix, offset);

        let uses: &[crate::ast::Use] = doc
            .last_valid_program
            .as_ref()
            .map(|p| p.uses.as_slice())
            .unwrap_or(&[]);

        if let Some(ref pc) = project_cache {
            collect_project_symbols(
                &mut items,
                pc,
                uses,
                workspace_root.as_ref().unwrap(),
                &word_prefix,
                &[],
            );
        }

        collect_stdlib_symbols(&mut items, stdlib_cache, uses, &word_prefix, &[]);
    } else {
        let uses: &[crate::ast::Use] = doc
            .last_valid_program
            .as_ref()
            .map(|p| p.uses.as_slice())
            .unwrap_or(&[]);

        // Path prefix: only collect from matching modules
        if let Some(ref pc) = project_cache {
            collect_project_symbols(
                &mut items,
                pc,
                uses,
                workspace_root.as_ref().unwrap(),
                &word_prefix,
                &resolved_prefix,
            );
        }

        collect_stdlib_symbols(
            &mut items,
            stdlib_cache,
            uses,
            &word_prefix,
            &resolved_prefix,
        );
    }

    // Filter by word prefix (case-insensitive) — already done in collectors,
    // but ensure it's applied for any edge cases
    items.retain(|item| {
        item.label
            .to_lowercase()
            .starts_with(&word_prefix.to_lowercase())
    });

    // Sort: local first (1_), project (2_), stdlib (3_), alphabetically within each
    items.sort_by(|a, b| {
        let sa = a.sort_text.as_deref().unwrap_or("9_");
        let sb = b.sort_text.as_deref().unwrap_or("9_");
        sa.cmp(sb)
    });

    // Populate additional_text_edits with correct insertion positions
    let insertion_pos = find_use_insertion_position(doc);
    for item in &mut items {
        if let Some(ref mut edits) = item.additional_text_edits {
            for edit in edits {
                edit.range = Range::new(insertion_pos, insertion_pos);
            }
        }
    }

    // Append `()` for function and method completions
    for item in &mut items {
        if matches!(
            item.kind,
            Some(CompletionItemKind::FUNCTION) | Some(CompletionItemKind::METHOD)
        ) {
            let name = item.label.clone();
            item.insert_text = Some(format!("{}($0)", name));
            item.insert_text_format = Some(InsertTextFormat::SNIPPET);
        }
    }

    if items.is_empty() {
        None
    } else {
        Some(CompletionResponse::Array(items))
    }
}

fn main_loop(
    connection: Connection,
    stdlib_cache: StdlibCache,
    workspace_root: Option<std::path::PathBuf>,
) -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
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
                        let response = handle_hover(&documents, &stdlib_cache, params);
                        let result = serde_json::to_value(&response)?;
                        connection
                            .sender
                            .send(Message::Response(Response::new_ok(id, result)))?;
                    }
                    "textDocument/definition" => {
                        let (id, params) = cast_request::<lsp_types::request::GotoDefinition>(req)?;
                        let response = handle_definition(&documents, &stdlib_cache, params);
                        let result = serde_json::to_value(&response)?;
                        connection
                            .sender
                            .send(Message::Response(Response::new_ok(id, result)))?;
                    }
                    "textDocument/completion" => {
                        let (id, params) = cast_request::<Completion>(req)?;
                        let response =
                            handle_completion(&documents, &stdlib_cache, &workspace_root, params);
                        let result = serde_json::to_value(&response)?;
                        connection
                            .sender
                            .send(Message::Response(Response::new_ok(id, result)))?;
                    }
                    _ => {}
                }
            }
            Message::Notification(not) => match not.method.as_str() {
                "textDocument/didOpen" => {
                    let params =
                        cast_notification::<lsp_types::notification::DidOpenTextDocument>(not)?;
                    let uri = params.text_document.uri;
                    let url = Url::parse(uri.as_str()).unwrap();
                    let text = params.text_document.text;

                    let diags = update_document(&mut documents, url.clone(), text);
                    publish_diagnostics(&connection, url, diags)?;
                }
                "textDocument/didChange" => {
                    let params =
                        cast_notification::<lsp_types::notification::DidChangeTextDocument>(not)?;
                    let uri = params.text_document.uri;
                    let url = Url::parse(uri.as_str()).unwrap();
                    if let Some(change) = params.content_changes.into_iter().next() {
                        let diags = update_document(&mut documents, url.clone(), change.text);
                        publish_diagnostics(&connection, url, diags)?;
                    }
                }
                "textDocument/didClose" => {
                    let params =
                        cast_notification::<lsp_types::notification::DidCloseTextDocument>(not)?;
                    let url = Url::parse(params.text_document.uri.as_str()).unwrap();
                    documents.remove(&url);
                }
                _ => {}
            },
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
        completion_provider: Some(CompletionOptions {
            resolve_provider: None,
            trigger_characters: None,
            work_done_progress_options: Default::default(),
            all_commit_characters: None,
            completion_item: None,
        }),
        ..Default::default()
    })?;

    let initialization_params = connection.initialize(server_capabilities)?;
    let init_params: InitializeParams = serde_json::from_value(initialization_params)?;

    let workspace_root = init_params
        .workspace_folders
        .as_ref()
        .and_then(|f| {
            let first_uri = f.first()?.uri.clone();
            Url::parse(first_uri.as_str()).ok()
        })
        .or_else(|| {
            #[allow(deprecated)]
            init_params
                .root_uri
                .as_ref()
                .and_then(|uri| Url::parse(uri.as_str()).ok())
        })
        .and_then(|url| url.to_file_path().ok());
    if let Some(ref root) = workspace_root {
        eprintln!("ulang lsp workspace root: {}", root.display());
    }

    let stdlib_cache = load_stdlib_cache();
    eprintln!(
        "ulang lsp server loaded {} stdlib modules",
        stdlib_cache.entries.len()
    );

    main_loop(connection, stdlib_cache, workspace_root)?;
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

    #[test]
    fn test_lsp_pub_and_mod_hover() {
        let src = r#"
            pub mod mymod {
                pub fn myfn() {}
            }
        "#;
        let mut tokens = Vec::new();
        let mut lexer = Lexer::new(src);
        loop {
            let (t, s) = lexer.next_token().unwrap();
            tokens.push((t.clone(), s));
            if tokens.last().unwrap().0 == Token::Eof {
                break;
            }
        }
        let mut parser = Parser::new(&tokens);
        let prog = parser.parse_program().unwrap();

        // 1. Check mod hover text
        let mod_hover = get_hover_text_from_program(&prog, "mymod", None);
        assert!(mod_hover.is_some());
        let mod_text = mod_hover.unwrap();
        assert!(mod_text.contains("pub mod mymod;"));

        // 2. Check pub fn hover text inside submodules (recursive)
        let fn_hover = get_hover_text_from_program(&prog, "myfn", None);
        assert!(fn_hover.is_some());
        let fn_text = fn_hover.unwrap();
        assert!(fn_text.contains("pub fn myfn()"));

        // 3. Check definition span for mod in find_definition_span
        let def_span = find_definition_span(&tokens, "mymod", 0, None);
        assert!(def_span.is_some());
    }

    #[test]
    fn test_lsp_single_extern_hover() {
        let src = r#"
            pub extern "C" fn fork() -> i32;
        "#;
        let mut tokens = Vec::new();
        let mut lexer = Lexer::new(src);
        loop {
            let (t, s) = lexer.next_token().unwrap();
            tokens.push((t.clone(), s));
            if tokens.last().unwrap().0 == Token::Eof {
                break;
            }
        }
        let mut parser = Parser::new(&tokens);
        let prog = parser.parse_program().unwrap();

        // 1. Check single extern hover text
        let ext_hover = get_hover_text_from_program(&prog, "fork", None);
        assert!(ext_hover.is_some());
        let ext_text = ext_hover.unwrap();
        assert!(ext_text.contains("pub extern \"C\" fn fork() -> i32"));

        // 2. Check definition span for single extern
        let def_span = find_definition_span(&tokens, "fork", 0, None);
        assert!(def_span.is_some());
    }

    #[test]
    fn test_find_definition_span_method_in_impl() {
        // Method definitions inside impl blocks must be discoverable
        let src = r#"
            struct Foo { x: i32 }
            impl Foo {
                pub fn bar(&self) -> i32 { self.x }
                fn baz(&self) -> i32 { self.x }
            }
        "#;
        let mut tokens = Vec::new();
        let mut lexer = Lexer::new(src);
        loop {
            let (t, s) = lexer.next_token().unwrap();
            tokens.push((t.clone(), s));
            if tokens.last().unwrap().0 == Token::Eof {
                break;
            }
        }

        // Find method 'bar' inside impl block
        let span_bar = find_definition_span(&tokens, "bar", 0, None);
        assert!(
            span_bar.is_some(),
            "should find 'bar' method definition inside impl"
        );

        // Find method 'baz' inside impl block (no pub)
        let span_baz = find_definition_span(&tokens, "baz", 0, None);
        assert!(
            span_baz.is_some(),
            "should find 'baz' method definition inside impl"
        );

        // Find struct 'Foo' definition alongside impl
        let span_foo = find_definition_span(&tokens, "Foo", 0, None);
        assert!(span_foo.is_some(), "should find 'Foo' struct definition");
    }

    #[test]
    fn test_find_definition_span_backward_search_local_var() {
        // Local backward search for let-binding
        let src = "fn main() { let x = 42; let y = x; }";
        let mut tokens = Vec::new();
        let mut lexer = Lexer::new(src);
        loop {
            let (t, s) = lexer.next_token().unwrap();
            tokens.push((t.clone(), s));
            if tokens.last().unwrap().0 == Token::Eof {
                break;
            }
        }

        // Hover over the second 'x' (the usage) and find its let-binding definition
        let second_x_offset = tokens
            .iter()
            .rposition(|(tok, _)| matches!(tok, Token::Ident(id) if id == "x"))
            .map(|i| tokens[i].1.lo)
            .unwrap_or(0);

        let def_span = find_definition_span(&tokens, "x", second_x_offset, None);
        assert!(
            def_span.is_some(),
            "should find 'x' let-binding definition from usage"
        );
    }

    #[test]
    fn test_stdlib_cache_loads_modules() {
        // Load stdlib cache from the actual filesystem
        let cache = load_stdlib_cache();
        // At minimum we should have the stdlib modules we know about
        assert!(
            !cache.entries.is_empty(),
            "stdlib cache should have entries"
        );

        // Verify we get expected modules by checking hover text
        // io.u should export `print` and `println`
        let print_hover = get_hover_text_from_stdlib(&cache, "print", None);
        assert!(print_hover.is_some(), "should find 'print' in stdlib");
        let print_text = print_hover.unwrap();
        assert!(
            print_text.contains("fn print"),
            "hover for print should be a function"
        );

        // option.u should export `Option` enum
        let option_hover = get_hover_text_from_stdlib(&cache, "Option", None);
        assert!(option_hover.is_some(), "should find 'Option' in stdlib");
        let option_text = option_hover.unwrap();
        assert!(
            option_text.contains("enum Option"),
            "hover for Option should be an enum"
        );

        // panic.u should export `panic` function
        let panic_hover = get_hover_text_from_stdlib(&cache, "panic", None);
        assert!(panic_hover.is_some(), "should find 'panic' in stdlib");
        let panic_text = panic_hover.unwrap();
        assert!(
            panic_text.contains("fn panic"),
            "hover for panic should be a function"
        );

        // alloc.u should export `malloc`, `free`, `realloc`, `memcpy`
        let malloc_hover = get_hover_text_from_stdlib(&cache, "malloc", None);
        assert!(malloc_hover.is_some(), "should find 'malloc' in stdlib");
        assert!(malloc_hover.unwrap().contains("fn malloc"));

        let free_hover = get_hover_text_from_stdlib(&cache, "free", None);
        assert!(free_hover.is_some(), "should find 'free' in stdlib");
        assert!(free_hover.unwrap().contains("fn free"));

        // process.u should export `exit`, `Command`
        let exit_hover = get_hover_text_from_stdlib(&cache, "exit", None);
        assert!(exit_hover.is_some(), "should find 'exit' in stdlib");

        let command_hover = get_hover_text_from_stdlib(&cache, "Command", None);
        assert!(command_hover.is_some(), "should find 'Command' in stdlib");
    }

    #[test]
    fn test_stdlib_definition_find_function() {
        // Verify we can find a definition span in stdlib for a function
        let cache = load_stdlib_cache();

        // Try to find definition of 'print' in stdlib
        let def = find_stdlib_definition(&cache, "print", None);
        assert!(
            def.is_some(),
            "should find definition for 'print' in stdlib"
        );

        let (url, range) = def.unwrap();
        // The URL should point to io.u
        let path = url.to_file_path().unwrap();
        assert!(
            path.to_string_lossy().contains("io.u"),
            "print definition should be in io.u, got {:?}",
            path
        );
        // Range should be meaningful (non-zero)
        assert!(
            range.start.line > 0 || range.start.character > 0 || range.end.line > 0,
            "definition range should be non-trivial"
        );

        // Find definition of 'Option' enum
        let option_def = find_stdlib_definition(&cache, "Option", None);
        assert!(
            option_def.is_some(),
            "should find definition for 'Option' in stdlib"
        );
        let (opt_url, _) = option_def.unwrap();
        let opt_path = opt_url.to_file_path().unwrap();
        assert!(
            opt_path.to_string_lossy().contains("option.u"),
            "Option definition should be in option.u, got {:?}",
            opt_path
        );

        // Find definition of 'malloc' which is extern "C"
        let malloc_def = find_stdlib_definition(&cache, "malloc", None);
        assert!(
            malloc_def.is_some(),
            "should find definition for 'malloc' in stdlib"
        );
        let (m_url, _) = malloc_def.unwrap();
        let m_path = m_url.to_file_path().unwrap();
        assert!(
            m_path.to_string_lossy().contains("libc.u"),
            "malloc definition should be in libc.u, got {:?}",
            m_path
        );
    }

    #[test]
    fn test_stdlib_definition_method_in_impl() {
        // Methods inside impl blocks should be findable via stdlib search
        let cache = load_stdlib_cache();

        // Option::unwrap is defined inside impl<T> Option<T> { ... }
        let unwrap_def = find_stdlib_definition(&cache, "unwrap", None);
        assert!(
            unwrap_def.is_some(),
            "should find 'unwrap' method inside impl block in stdlib"
        );

        let (url, _) = unwrap_def.unwrap();
        let path = url.to_file_path().unwrap();
        let path_str = path.to_string_lossy();
        assert!(
            path_str.contains("option.u") || path_str.contains("result.u"),
            "unwrap definition should be in option.u or result.u, got {:?}",
            path
        );

        // Vec::push is defined inside impl<T> Vec<T> { ... }
        let push_def = find_stdlib_definition(&cache, "push", None);
        assert!(
            push_def.is_some(),
            "should find 'push' method inside impl block in stdlib"
        );
        let (push_url, _) = push_def.unwrap();
        let push_path = push_url.to_file_path().unwrap();
        assert!(
            push_path.to_string_lossy().contains("vec.u"),
            "push definition should be in vec.u, got {:?}",
            push_path
        );
    }

    #[test]
    fn test_get_path_prefix() {
        let src = "std::vec::Vec::with_capacity(cap)";
        let mut tokens = Vec::new();
        let mut lexer = Lexer::new(src);
        loop {
            let (t, s) = lexer.next_token().unwrap();
            tokens.push((t.clone(), s));
            if tokens.last().unwrap().0 == Token::Eof {
                break;
            }
        }
        let with_cap_idx = tokens
            .iter()
            .position(|(tok, _)| matches!(tok, Token::Ident(id) if id == "with_capacity"))
            .unwrap();
        let prefix = get_path_prefix(&tokens, with_cap_idx);
        assert_eq!(
            prefix,
            vec!["std".to_string(), "vec".to_string(), "Vec".to_string()]
        );
    }

    #[test]
    fn test_resolve_prefix_to_full_path() {
        let uses = vec![crate::ast::Use {
            path: vec!["std".to_string(), "vec".to_string(), "Vec".to_string()],
            is_pub: false,
            module_path: Vec::new(),
            span: Span::new(0, 0),
        }];
        let resolved = resolve_prefix_to_full_path(&uses, &["Vec".to_string()]);
        assert_eq!(
            resolved,
            vec!["std".to_string(), "vec".to_string(), "Vec".to_string()]
        );
    }

    #[test]
    fn test_path_prefixed_definition_navigation() {
        let cache = load_stdlib_cache();
        let mut documents = HashMap::new();
        let doc_url = Url::parse("file:///home/user/project/main.u").unwrap();
        let src = "use std::vec::Vec;\nfn main() {\n    Vec::with_capacity(10);\n}";
        let diags = update_document(&mut documents, doc_url.clone(), src.to_string());
        assert!(
            diags.is_empty(),
            "Document should compile without diagnostics, but got: {:?}",
            diags
        );
        let doc = documents.get(&doc_url).unwrap();
        assert!(
            doc.last_valid_program.is_some(),
            "last_valid_program should be Some"
        );

        let position = Position::new(2, 9); // line 2 (0-indexed), char 9
        let params = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position,
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let response = handle_definition(&documents, &cache, params);
        assert!(
            response.is_some(),
            "Should find definition for Vec::with_capacity"
        );
        if let Some(GotoDefinitionResponse::Scalar(location)) = response {
            let url = Url::parse(location.uri.as_str()).unwrap();
            let path = url.to_file_path().unwrap();
            assert!(
                path.to_string_lossy().contains("vec.u"),
                "Should navigate to vec.u, got {:?}",
                path
            );
        } else {
            panic!("Expected Scalar response");
        }
    }

    #[test]
    fn test_enum_variant_definition_navigation() {
        let cache = load_stdlib_cache();
        let mut documents = HashMap::new();
        let doc_url = Url::parse("file:///home/user/project/main.u").unwrap();
        let src = "use std::option::Option;\nfn main() {\n    let val = Option::Some(42);\n}";
        update_document(&mut documents, doc_url.clone(), src.to_string());

        let position = Position::new(2, 22);
        let params = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position,
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let response = handle_definition(&documents, &cache, params);
        assert!(
            response.is_some(),
            "Should find definition for Option::Some"
        );
        if let Some(GotoDefinitionResponse::Scalar(location)) = response {
            let url = Url::parse(location.uri.as_str()).unwrap();
            let path = url.to_file_path().unwrap();
            assert!(
                path.to_string_lossy().contains("option.u"),
                "Should navigate to option.u, got {:?}",
                path
            );
        } else {
            panic!("Expected Scalar response");
        }

        let hover_params = HoverParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position,
            },
            work_done_progress_params: Default::default(),
        };
        let hover_res = handle_hover(&documents, &cache, hover_params);
        assert!(hover_res.is_some(), "Should find hover for Option::Some");
        if let Some(hover) = hover_res {
            if let HoverContents::Markup(markup) = hover.contents {
                assert!(
                    markup.value.contains("Some(T)") || markup.value.contains("Some"),
                    "Hover text should contain Some(T), got: {}",
                    markup.value
                );
            } else {
                panic!("Expected Markup contents");
            }
        }
    }

    #[test]
    fn test_member_method_navigation() {
        let cache = load_stdlib_cache();
        let mut documents = HashMap::new();
        let doc_url = Url::parse("file:///home/user/project/main.u").unwrap();
        let src = "use std::slice;\nfn foo(args: &[i32]) {\n    let len = args.len();\n}";
        update_document(&mut documents, doc_url.clone(), src.to_string());

        let position = Position::new(2, 19);
        let params = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position,
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let response = handle_definition(&documents, &cache, params);
        assert!(response.is_some(), "Should find definition for .len()");
        if let Some(GotoDefinitionResponse::Scalar(location)) = response {
            let url = Url::parse(location.uri.as_str()).unwrap();
            let path = url.to_file_path().unwrap();
            assert!(
                path.to_string_lossy().contains("slice.u"),
                "Should navigate to slice.u, got {:?}",
                path
            );
        } else {
            panic!("Expected Scalar response");
        }
    }

    #[test]
    fn test_struct_field_navigation_and_hover() {
        let cache = load_stdlib_cache();
        let mut documents = HashMap::new();
        let doc_url = Url::parse("file:///home/user/project/main.u").unwrap();
        let src = "struct Point { x: i32, y: i32 }\nfn main() {\n    let p = Point { x: 10, y: 20 };\n    let val = p.x;\n}";
        update_document(&mut documents, doc_url.clone(), src.to_string());

        let position = Position::new(3, 16);
        let params = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position,
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let response = handle_definition(&documents, &cache, params);
        assert!(
            response.is_some(),
            "Should find definition for struct field x"
        );
        if let Some(GotoDefinitionResponse::Scalar(location)) = response {
            let url = Url::parse(location.uri.as_str()).unwrap();
            assert_eq!(url, doc_url);
            assert_eq!(location.range.start.line, 0);
            assert_eq!(location.range.start.character, 15);
        } else {
            panic!("Expected Scalar response");
        }

        let hover_params = HoverParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position,
            },
            work_done_progress_params: Default::default(),
        };
        let hover_res = handle_hover(&documents, &cache, hover_params);
        assert!(hover_res.is_some(), "Should find hover for struct field x");
        if let Some(hover) = hover_res {
            if let HoverContents::Markup(markup) = hover.contents {
                assert!(
                    markup.value.contains("struct Point") && markup.value.contains("x: i32"),
                    "Hover text should contain struct Point and x: i32, got: {}",
                    markup.value
                );
            } else {
                panic!("Expected Markup contents");
            }
        }
    }

    #[test]
    fn test_struct_field_context_navigation() {
        let cache = load_stdlib_cache();
        let mut documents = HashMap::new();
        let doc_url = Url::parse("file:///home/user/project/main.u").unwrap();
        let src = r#"
            struct Iter { ptr: *mut u8, end: *mut u8 }
            struct IterMut { ptr: *mut u8, end: *mut u8 }
            fn main() {
                let it = IterMut { ptr: 0 as *mut u8, end: 0 as *mut u8 };
            }
        "#;
        update_document(&mut documents, doc_url.clone(), src.to_string());

        let position = Position::new(4, 35);
        let params = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position,
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let response = handle_definition(&documents, &cache, params);
        assert!(
            response.is_some(),
            "Should find definition for struct field ptr"
        );
        if let Some(GotoDefinitionResponse::Scalar(location)) = response {
            let url = Url::parse(location.uri.as_str()).unwrap();
            assert_eq!(url, doc_url);
            assert_eq!(location.range.start.line, 2);
            assert_eq!(location.range.start.character, 29);
        } else {
            panic!("Expected Scalar response");
        }
    }

    #[test]
    fn test_struct_field_vs_local_variable_navigation() {
        let cache = load_stdlib_cache();
        let mut documents = HashMap::new();
        let doc_url = Url::parse("file:///home/user/project/main.u").unwrap();
        let src = r#"
            struct Iter { ptr: *mut u8, end: *mut u8 }
            fn main() {
                let end = 0 as *mut u8;
                let it = Iter { ptr: end, end: end };
            }
        "#;
        update_document(&mut documents, doc_url.clone(), src.to_string());

        // 1. Click on the first `end` (the field name) at line 4, character 42
        let params_field = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position: Position::new(4, 42),
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let response_field = handle_definition(&documents, &cache, params_field);
        assert!(
            response_field.is_some(),
            "Should find definition for struct field end"
        );
        if let Some(GotoDefinitionResponse::Scalar(location)) = response_field {
            assert_eq!(location.range.start.line, 1);
            assert_eq!(location.range.start.character, 40);
        } else {
            panic!("Expected Scalar response for field");
        }

        // 2. Click on the second `end` (the variable) at line 4, character 47
        let params_var = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position: Position::new(4, 47),
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let response_var = handle_definition(&documents, &cache, params_var);
        assert!(
            response_var.is_some(),
            "Should find definition for local variable end"
        );
        if let Some(GotoDefinitionResponse::Scalar(location)) = response_var {
            assert_eq!(location.range.start.line, 3);
            assert_eq!(location.range.start.character, 20);
        } else {
            panic!("Expected Scalar response for variable");
        }
    }

    #[test]
    fn test_generic_param_navigation() {
        let cache = load_stdlib_cache();
        let mut documents = HashMap::new();
        let doc_url = Url::parse("file:///home/user/project/main.u").unwrap();
        let src = r#"
            pub enum Option<T> {
                Some(T),
                None,
            }
            impl<T> Option<T> {
                pub fn unwrap(&self) -> T {
                    match *self {
                        Option::Some(val) => val,
                        Option::None => {
                            panic("error");
                        }
                    }
                }
            }
        "#;
        update_document(&mut documents, doc_url.clone(), src.to_string());

        // 1. Click on generic T inside Option<T> definition on line 2 (Some(T)):
        // "                Some(T),"
        // 16 spaces + `Some(` (5 chars) = 21. Let's click at line 2, character 21.
        let params_enum = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position: Position::new(2, 21),
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let response_enum = handle_definition(&documents, &cache, params_enum);
        assert!(
            response_enum.is_some(),
            "Should find definition for generic T in enum"
        );
        if let Some(GotoDefinitionResponse::Scalar(location)) = response_enum {
            assert_eq!(location.range.start.line, 1); // pub enum Option<T>
            assert_eq!(location.range.start.character, 28); // generic T is at character index 28 (12 spaces + `pub enum Option<` is 28)
        } else {
            panic!("Expected Scalar response for generic T in enum");
        }

        // 2. Click on generic T inside unwrap return type on line 6 (-> T):
        // "                pub fn unwrap(&self) -> T {"
        // 16 spaces + `pub fn unwrap(&self) -> ` (24 chars) = 40.
        let params_impl = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position: Position::new(6, 40),
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let response_impl = handle_definition(&documents, &cache, params_impl);
        assert!(
            response_impl.is_some(),
            "Should find definition for generic T in impl block"
        );
        if let Some(GotoDefinitionResponse::Scalar(location)) = response_impl {
            assert_eq!(location.range.start.line, 5); // impl<T> Option<T>
            assert_eq!(location.range.start.character, 17); // generic T is at character index 17 (12 spaces + `impl<` is 17)
        } else {
            panic!("Expected Scalar response for generic T in impl block");
        }
    }

    #[test]
    fn test_const_generic_param_navigation() {
        let cache = load_stdlib_cache();
        let mut documents = HashMap::new();
        let doc_url = Url::parse("file:///home/user/project/main.u").unwrap();
        let src = r#"
            impl<T, const L: usize> [T; L] {
                fn len(&self) -> usize {
                    L
                }
            }
        "#;
        update_document(&mut documents, doc_url.clone(), src.to_string());

        // Click on const generic param L inside the function body on line 3:
        // "                    L"
        // 20 spaces. Click at line 3, character 20.
        let params = GotoDefinitionParams {
            text_document_position_params: lsp_types::TextDocumentPositionParams {
                text_document: lsp_types::TextDocumentIdentifier {
                    uri: doc_url.to_string().parse().unwrap(),
                },
                position: Position::new(3, 20),
            },
            work_done_progress_params: Default::default(),
            partial_result_params: Default::default(),
        };

        let mut lexer = Lexer::new(&src);
        let mut tokens = Vec::new();
        while let Ok((tok, span)) = lexer.next_token() {
            tokens.push((tok.clone(), span));
            if matches!(tok, Token::Eof) {
                break;
            }
        }
        let scopes = collect_generic_scopes(&tokens);
        let response = handle_definition(&documents, &cache, params);
        assert!(
            response.is_some(),
            "Should find definition for const generic L\nTokens: {:#?}\nScopes: {:#?}",
            tokens,
            scopes
        );
        if let Some(GotoDefinitionResponse::Scalar(location)) = response {
            assert_eq!(location.range.start.line, 1); // impl<T, const L: usize>
            assert_eq!(location.range.start.character, 26); // L is at character index 26 (12 spaces + `impl<T, const ` is 26)
        } else {
            panic!("Expected Scalar response for const generic L");
        }
    }

    // ---- Completion tests ----

    #[test]
    fn test_completion_word_prefix_at_offset() {
        let src = "fn main() { let hello_world = 42; hel";
        let mut tokens = Vec::new();
        let mut lexer = Lexer::new(src);
        loop {
            let (t, s) = lexer.next_token().unwrap();
            tokens.push((t.clone(), s));
            if matches!(t, Token::Eof) {
                break;
            }
        }
        // Cursor on 'hel' inside the partial identifier
        let hel_span = tokens
            .iter()
            .find(|(t, _)| matches!(t, Token::Ident(id) if id == "hel"))
            .unwrap()
            .1;
        // Place cursor at the end of "hel" (full prefix)
        let prefix = get_word_prefix_at_offset(&tokens, src, hel_span.hi);
        assert_eq!(prefix, "hel");

        // Cursor after whitespace, nothing
        let prefix_empty = get_word_prefix_at_offset(&tokens, src, 0);
        assert_eq!(prefix_empty, "");
    }

    #[test]
    fn test_completion_path_prefix_at_offset() {
        let src = "std::vec::Vec";
        let mut tokens = Vec::new();
        let mut lexer = Lexer::new(src);
        loop {
            let (t, s) = lexer.next_token().unwrap();
            tokens.push((t.clone(), s));
            if matches!(t, Token::Eof) {
                break;
            }
        }
        // Cursor at the end (after Vec)
        let offset = src.len();
        let prefix = get_path_prefix_at_offset(&tokens, offset);
        assert_eq!(prefix, vec!["std".to_string(), "vec".to_string()]);

        // Cursor in the middle (no path prefix)
        let prefix_none = get_path_prefix_at_offset(&tokens, 0);
        assert!(prefix_none.is_empty());
    }

    #[test]
    fn test_completion_use_insertion_position_no_uses() {
        let doc = DocumentState {
            source: "fn main() {}\n".to_string(),
            last_valid_program: {
                let mut lexer = Lexer::new("fn main() {}");
                let mut tokens = Vec::new();
                loop {
                    let (t, s) = lexer.next_token().unwrap();
                    tokens.push((t.clone(), s));
                    if matches!(t, Token::Eof) {
                        break;
                    }
                }
                let mut parser = Parser::new(&tokens);
                parser.parse_program().ok()
            },
        };
        let pos = find_use_insertion_position(&doc);
        assert_eq!(pos.line, 0);
        assert_eq!(pos.character, 0);
    }

    #[test]
    fn test_completion_use_insertion_position_with_uses() {
        let src = "use std::io::print;\nfn main() {}\n";
        let doc = DocumentState {
            source: src.to_string(),
            last_valid_program: {
                let mut lexer = Lexer::new(src);
                let mut tokens = Vec::new();
                loop {
                    let (t, s) = lexer.next_token().unwrap();
                    tokens.push((t.clone(), s));
                    if matches!(t, Token::Eof) {
                        break;
                    }
                }
                let mut parser = Parser::new(&tokens);
                parser.parse_program().ok()
            },
        };
        let pos = find_use_insertion_position(&doc);
        // Should be after the first \n (line 1, col 0)
        assert_eq!(pos.line, 1);
        assert_eq!(pos.character, 0);
    }

    #[test]
    fn test_completion_rexported_by_std() {
        let cache = load_stdlib_cache();
        // option is re-exported by std
        assert!(is_core_reexported_by_std(&cache, "option"));
        // alloc is NOT re-exported by std (it's a direct std module)
        assert!(!is_core_reexported_by_std(&cache, "alloc"));
    }

    #[test]
    fn test_completion_stdlib_module_path() {
        let url = Url::parse("file:///usr/share/ulang/stdlib/core/option.u").unwrap();
        let path = stdlib_module_path(&url);
        assert_eq!(path, vec!["core".to_string(), "option".to_string()]);

        let url2 = Url::parse("file:///usr/share/ulang/stdlib/std/vec.u").unwrap();
        let path2 = stdlib_module_path(&url2);
        assert_eq!(path2, vec!["std".to_string(), "vec".to_string()]);
    }

    #[test]
    fn test_completion_project_file_module_path() {
        let root = std::path::PathBuf::from("/home/user/project");
        let url = Url::parse("file:///home/user/project/src/foo.u").unwrap();
        let path = project_file_module_path(&root, &url);
        assert_eq!(path, vec!["src".to_string(), "foo".to_string()]);

        let url2 = Url::parse("file:///home/user/project/src/sub/bar.u").unwrap();
        let path2 = project_file_module_path(&root, &url2);
        assert_eq!(
            path2,
            vec!["src".to_string(), "sub".to_string(), "bar".to_string()]
        );
    }

    #[test]
    fn test_completion_already_imported_detection() {
        let uses = vec![crate::ast::Use {
            path: vec![
                "std".to_string(),
                "option".to_string(),
                "Option".to_string(),
            ],
            is_pub: false,
            module_path: Vec::new(),
            span: Span::new(0, 0),
        }];
        assert!(is_already_imported(
            &uses,
            &["std".to_string(), "option".to_string()],
            "Option"
        ));
        assert!(!is_already_imported(
            &uses,
            &["std".to_string(), "option".to_string()],
            "None"
        ));
        assert!(!is_already_imported(
            &uses,
            &["std".to_string(), "result".to_string()],
            "Option"
        ));
    }

    #[test]
    fn test_completion_collect_local_symbols() {
        let src = r#"
            struct Foo { x: i32 }
            enum Bar { A, B }
            trait Baz { fn method(&self); }
            type MyInt = i32;
            mod mymod {}
            fn my_func(a: i32) -> i32 { a }
        "#;
        let doc = DocumentState {
            source: src.to_string(),
            last_valid_program: {
                let mut tokens = Vec::new();
                let mut lexer = Lexer::new(src);
                loop {
                    let (t, s) = lexer.next_token().unwrap();
                    tokens.push((t.clone(), s));
                    if matches!(t, Token::Eof) {
                        break;
                    }
                }
                let mut parser = Parser::new(&tokens);
                parser.parse_program().ok()
            },
        };

        let mut items = Vec::new();
        let offset = src.len(); // cursor at end
        collect_local_symbols(&mut items, &doc, "", offset);

        // Should have Foo, Bar, Baz, MyInt, mymod, my_func
        let names: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(names.contains(&"Foo"));
        assert!(names.contains(&"Bar"));
        assert!(names.contains(&"Baz"));
        assert!(names.contains(&"MyInt"));
        assert!(names.contains(&"mymod"));
        assert!(names.contains(&"my_func"));
    }

    #[test]
    fn test_completion_collect_local_variables_in_scope() {
        let src = "fn main() { let x = 1; let mut y = 2; let z = x; }";
        let doc = DocumentState {
            source: src.to_string(),
            last_valid_program: {
                let mut tokens = Vec::new();
                let mut lexer = Lexer::new(src);
                loop {
                    let (t, s) = lexer.next_token().unwrap();
                    tokens.push((t.clone(), s));
                    if matches!(t, Token::Eof) {
                        break;
                    }
                }
                let mut parser = Parser::new(&tokens);
                parser.parse_program().ok()
            },
        };

        let mut items = Vec::new();
        let offset = src.len(); // cursor at end
        collect_local_symbols(&mut items, &doc, "", offset);

        let names: Vec<&str> = items.iter().map(|i| i.label.as_str()).collect();
        assert!(names.contains(&"x"));
        assert!(names.contains(&"y"));
        assert!(names.contains(&"z"));
    }

    #[test]
    fn test_completion_collect_stdlib_symbols() {
        let cache = load_stdlib_cache();
        let uses: Vec<crate::ast::Use> = Vec::new();

        let mut items = Vec::new();
        collect_stdlib_symbols(&mut items, &cache, &uses, "Option", &[]);
        assert!(!items.is_empty(), "Should find Option in stdlib");
        assert!(
            items.iter().any(|i| i.label == "Option"),
            "Option should be in completions"
        );

        // Option should have additionalTextEdits since nothing is imported
        let option_item = items.iter().find(|i| i.label == "Option").unwrap();
        assert!(
            option_item.additional_text_edits.is_some(),
            "Option should have use insertion"
        );
    }

    #[test]
    fn test_completion_existing_import_no_use_insertion() {
        let cache = load_stdlib_cache();
        let src = "use std::option::Option;\nfn main() {}";
        let doc = DocumentState {
            source: src.to_string(),
            last_valid_program: {
                let mut tokens = Vec::new();
                let mut lexer = Lexer::new(src);
                loop {
                    let (t, s) = lexer.next_token().unwrap();
                    tokens.push((t.clone(), s));
                    if matches!(t, Token::Eof) {
                        break;
                    }
                }
                let mut parser = Parser::new(&tokens);
                parser.parse_program().ok()
            },
        };

        let mut items = Vec::new();
        let uses: &[crate::ast::Use] = doc
            .last_valid_program
            .as_ref()
            .map(|p| p.uses.as_slice())
            .unwrap_or(&[]);
        collect_stdlib_symbols(&mut items, &cache, uses, "Option", &[]);

        let option_item = items.iter().find(|i| i.label == "Option");
        assert!(option_item.is_some(), "Option should appear in completions");
        assert!(
            option_item.unwrap().additional_text_edits.is_none(),
            "Option should NOT have use insertion (already imported)"
        );
    }

    #[test]
    fn test_completion_path_prefix_stdlib() {
        let cache = load_stdlib_cache();
        let uses: Vec<crate::ast::Use> = Vec::new();

        let mut items = Vec::new();
        collect_stdlib_symbols(
            &mut items,
            &cache,
            &uses,
            "",
            &["std".to_string(), "vec".to_string()],
        );
        assert!(!items.is_empty(), "Should find vec symbols");
        // Vec is defined in std/vec.u
        assert!(
            items.iter().any(|i| i.label == "Vec"),
            "Vec should be in vec module completions"
        );
    }
}
