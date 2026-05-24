mod ast;
mod codegen;
mod error;
mod lexer;
mod parser;
mod token;

use annotate_snippets::renderer::{AnsiColor, Effects};
use clap::{Parser, Subcommand, builder::Styles};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::process;

use crate::ast::{EnumDecl, Function, ImplDecl, Program, StructDecl, Type};

type OverloadMap = HashMap<String, Vec<(String, Vec<Type>)>>;
use crate::token::{Span, Token};
use inkwell::targets::TargetTriple;

#[derive(Clone, Copy, Debug, Default)]
enum OptLevel {
    None,
    #[default]
    Default,
    Less,
    Aggressive,
}

fn parse_opt_level(s: &str) -> Result<OptLevel, String> {
    match s {
        "0" | "none" => Ok(OptLevel::None),
        "1" | "less" => Ok(OptLevel::Less),
        "2" | "default" => Ok(OptLevel::Default),
        "3" | "aggressive" => Ok(OptLevel::Aggressive),
        _ => Err(format!(
            "invalid optimization level '{s}': use 0|none, 1|less, 2|default, 3|aggressive"
        )),
    }
}

impl std::fmt::Display for OptLevel {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            OptLevel::None => write!(f, "0"),
            OptLevel::Less => write!(f, "1"),
            OptLevel::Default => write!(f, "2"),
            OptLevel::Aggressive => write!(f, "3"),
        }
    }
}

impl From<OptLevel> for inkwell::OptimizationLevel {
    fn from(val: OptLevel) -> Self {
        match val {
            OptLevel::None => inkwell::OptimizationLevel::None,
            OptLevel::Less => inkwell::OptimizationLevel::Less,
            OptLevel::Default => inkwell::OptimizationLevel::Default,
            OptLevel::Aggressive => inkwell::OptimizationLevel::Aggressive,
        }
    }
}

#[derive(Clone, Copy, Debug, Default)]
enum Cc {
    #[default]
    Gcc,
    Clang,
    Cosmocc,
    Zig,
    Tcc,
}

fn parse_cc(s: &str) -> Result<Cc, String> {
    match s {
        "gcc" => Ok(Cc::Gcc),
        "clang" => Ok(Cc::Clang),
        "cosmocc" => Ok(Cc::Cosmocc),
        "zig" => Ok(Cc::Zig),
        "tcc" => Ok(Cc::Tcc),
        _ => Err(format!(
            "invalid C compiler '{s}': use gcc, clang, cosmocc, zig, or tcc"
        )),
    }
}

impl std::fmt::Display for Cc {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Cc::Gcc => write!(f, "gcc"),
            Cc::Clang => write!(f, "clang"),
            Cc::Cosmocc => write!(f, "cosmocc"),
            Cc::Zig => write!(f, "zig"),
            Cc::Tcc => write!(f, "tcc"),
        }
    }
}

#[derive(Parser)]
#[command(name = "ulang", version, about = "A tiny compiled language",
    styles = Styles::styled()
        .header(AnsiColor::BrightGreen.on_default() | Effects::BOLD | Effects::UNDERLINE)
        .usage(AnsiColor::Cyan.on_default() | Effects::BOLD)
        .literal(AnsiColor::BrightCyan.on_default() | Effects::BOLD)
        .placeholder(AnsiColor::Cyan.on_default()))]
struct Cli {
    #[command(subcommand)]
    command: Command,
    /// Disable loading of the standard library
    #[arg(long = "no-std", global = true)]
    no_std: bool,
}

#[derive(Subcommand)]
enum Command {
    /// Compile and run the source file via JIT
    Run {
        /// Path to .u source file
        file: String,
        /// Optimization level (0|none, 1|less, 2|default, 3|aggressive)
        #[arg(short = 'o', long = "opt", default_value_t = OptLevel::Default, value_parser = parse_opt_level)]
        opt: OptLevel,
    },
    /// Compile to a native executable
    Build {
        /// Path to .u source file
        file: String,
        /// Output executable path [default: a.out]
        #[arg(long = "output")]
        output: Option<String>,
        /// Optimization level (0|none, 1|less, 2|default, 3|aggressive)
        #[arg(short = 'o', long = "opt", default_value_t = OptLevel::Default, value_parser = parse_opt_level)]
        opt: OptLevel,
        /// C compiler to use for linking (gcc, clang, cosmocc, zig, tcc)
        #[arg(long = "cc", default_value_t = Cc::Gcc, value_parser = parse_cc)]
        cc: Cc,
    },
    /// Compile to a native executable and run it
    #[command(name = "build-run")]
    BuildRun {
        /// Path to .u source file
        file: String,
        /// Output executable path [default: a.out]
        #[arg(long = "output")]
        output: Option<String>,
        /// Optimization level (0|none, 1|less, 2|default, 3|aggressive)
        #[arg(short = 'o', long = "opt", default_value_t = OptLevel::Default, value_parser = parse_opt_level)]
        opt: OptLevel,
        /// C compiler to use for linking (gcc, clang, cosmocc, zig, tcc)
        #[arg(long = "cc", default_value_t = Cc::Gcc, value_parser = parse_cc)]
        cc: Cc,
    },
}

enum Mode {
    Run {
        opt: OptLevel,
    },
    Build {
        output: Option<String>,
        opt: OptLevel,
        cc: Cc,
    },
    BuildRun {
        output: Option<String>,
        opt: OptLevel,
        cc: Cc,
    },
}

/// Compile the module to an object file, link it, and return the executable path.
/// Exits the process on failure.
fn do_build(codegen: &mut codegen::CodeGen<'_>, output: Option<String>, cc: Cc) -> String {
    let exe_path = output.unwrap_or_else(|| "a.out".to_string());
    let obj_path = format!("{}.o", exe_path);

    match cc {
        Cc::Cosmocc => {
            if let Err(msg) = codegen.compile_to_object_for_triple(
                &TargetTriple::create("x86_64-pc-linux-gnu"),
                Path::new(&obj_path),
            ) {
                eprintln!("codegen error: {}", msg);
                process::exit(1);
            }

            let obj_dir = Path::new(&obj_path)
                .parent()
                .unwrap_or_else(|| Path::new("."));
            let aarch64_dir = obj_dir.join(".aarch64");
            std::fs::create_dir_all(&aarch64_dir).unwrap_or_else(|e| {
                eprintln!("failed to create '{}': {}", aarch64_dir.display(), e);
                process::exit(1);
            });
            let aarch64_obj =
                Path::new(&aarch64_dir).join(Path::new(&obj_path).file_name().unwrap());
            if let Err(msg) = codegen.compile_to_object_for_triple(
                &TargetTriple::create("aarch64-linux-gnu"),
                &aarch64_obj,
            ) {
                eprintln!("codegen error: {}", msg);
                let _ = std::fs::remove_dir_all(&aarch64_dir);
                let _ = fs::remove_file(&obj_path);
                process::exit(1);
            }

            let cc_args: &[&str] = &["cosmocc"];
            if let Err(msg) = codegen::CodeGen::link_executable(
                cc_args,
                Path::new(&obj_path),
                Path::new(&exe_path),
            ) {
                eprintln!("link error: {}", msg);
                let _ = std::fs::remove_dir_all(&aarch64_dir);
                let _ = fs::remove_file(&obj_path);
                process::exit(1);
            }

            let _ = std::fs::remove_dir_all(&aarch64_dir);
            let _ = fs::remove_file(&obj_path);
        }
        _ => {
            if let Err(msg) = codegen.compile_to_object(Path::new(&obj_path)) {
                eprintln!("codegen error: {}", msg);
                process::exit(1);
            }

            let cc_str = cc.to_string();
            let cc_args: &[&str] = match cc {
                Cc::Zig => &["zig", "cc"],
                _ => &[&cc_str],
            };
            if let Err(msg) = codegen::CodeGen::link_executable(
                cc_args,
                Path::new(&obj_path),
                Path::new(&exe_path),
            ) {
                eprintln!("link error: {}", msg);
                let _ = fs::remove_file(&obj_path);
                process::exit(1);
            }

            let _ = fs::remove_file(&obj_path);
        }
    }

    exe_path
}

/// Lex and parse a .u source file, returning the Program.
fn lex_and_parse(source: &str, path: &str) -> Program {
    let mut lexer = lexer::Lexer::new(source);
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
                error::emit_error(source, path, Span::new(pos, pos), "lex error", &msg);
                process::exit(1);
            }
        }
    }

    let mut parser = parser::Parser::new(&tokens);
    match parser.parse_program() {
        Ok(prog) => prog,
        Err(e) => {
            error::emit_error(source, path, e.span, "parse error", &e.msg);
            process::exit(1);
        }
    }
}

/// Process a stdlib module's functions, detecting duplicates and building overload map.
/// Returns (processed_functions, overload_entries) where:
/// - processed_functions has duplicates mangled with `$N` suffix
/// - overload_entries maps base name -> { arg_count -> mangled_name }
fn process_stdlib_functions(funcs: Vec<Function>) -> (Vec<Function>, OverloadMap) {
    let mut name_counts: HashMap<String, usize> = HashMap::new();
    for func in &funcs {
        *name_counts.entry(func.name.clone()).or_insert(0) += 1;
    }

    let mut overloads: OverloadMap = HashMap::new();
    let mut processed = Vec::with_capacity(funcs.len());
    // Counter for generating unique suffixes per base name
    let mut counters: HashMap<String, usize> = HashMap::new();

    for mut func in funcs {
        if *name_counts.get(&func.name).unwrap_or(&0) > 1 {
            let idx = counters.entry(func.name.clone()).or_insert(0);
            let mangled = format!("{}${}", func.name, idx);
            *idx += 1;
            let param_types: Vec<Type> = func.params.iter().map(|p| p.ty.clone()).collect();
            overloads
                .entry(func.name.clone())
                .or_default()
                .push((mangled.clone(), param_types));
            func.name = mangled;
        }
        processed.push(func);
    }

    (processed, overloads)
}

/// Process a stdlib module's internal use declarations recursively.
fn resolve_module_uses(
    program: &Program,
    all_stdlib_progs: &HashMap<String, Program>,
    all_structs: &mut HashMap<String, StructDecl>,
    all_enums: &mut HashMap<String, EnumDecl>,
    all_impls: &mut Vec<ImplDecl>,
    all_funcs: &mut HashMap<String, Function>,
    all_overloads: &mut OverloadMap,
    canonical_struct_sources: &HashMap<String, String>,
    resolving: &mut HashSet<String>,
) {
    for use_decl in &program.uses {
        let path = &use_decl.path;
        if path.len() < 3 || path[0] != "std" {
            continue;
        }
        let module_name = &path[1];
        if !resolving.insert(module_name.clone()) {
            continue; // already resolving/processed
        }
        if let Some(prog) = all_stdlib_progs.get(module_name) {
            // First, resolve the dependency's own uses (recursive)
            resolve_module_uses(
                prog,
                all_stdlib_progs,
                all_structs,
                all_enums,
                all_impls,
                all_funcs,
                all_overloads,
                canonical_struct_sources,
                resolving,
            );
            // Import all structs from this dependency
            for decl in &prog.structs {
                all_structs
                    .entry(decl.name.clone())
                    .or_insert_with(|| decl.clone());
            }
            // Import all enums from this dependency
            for decl in &prog.enums {
                all_enums
                    .entry(decl.name.clone())
                    .or_insert_with(|| decl.clone());
            }
            // Import all impls
            for decl in &prog.impls {
                all_impls.push(decl.clone());
            }
            // Import all functions (including extern "C" declarations)
            for func in &prog.funcs {
                if !all_funcs.contains_key(&func.name) {
                    all_funcs.insert(func.name.clone(), func.clone());
                }
            }
            // Import overloads
            if path.len() >= 3 {
                let _target_name = &path[2];
                let (_module_funcs, module_overloads) =
                    process_stdlib_functions(prog.funcs.clone());
                for (base_name, overloads_list) in &module_overloads {
                    let qualified_base = format!("{}::{}", module_name, base_name);
                    let qualified_list: Vec<(String, Vec<Type>)> = overloads_list
                        .iter()
                        .map(|(mangled, params)| {
                            (format!("{}::{}", module_name, mangled), params.clone())
                        })
                        .collect();
                    all_overloads.insert(qualified_base, qualified_list);
                }
            }
            // Import impls that match this module's structs
            for impl_decl in &prog.impls {
                let impl_type_name = match &impl_decl.impl_type {
                    Type::Struct(name) => name.clone(),
                    Type::GenericInstance(name, _) => name.clone(),
                    _ => continue,
                };
                if all_structs.contains_key(&impl_type_name) {
                    // Already imported above, skip
                }
            }
        }
    }
}

/// Collect all struct type names referenced by a type (including nested/generic).
fn collect_struct_deps(ty: &Type) -> Vec<String> {
    match ty {
        Type::Struct(name) => vec![name.clone()],
        Type::GenericInstance(name, args) => {
            let mut deps = vec![name.clone()];
            deps.extend(args.iter().flat_map(collect_struct_deps));
            deps
        }
        Type::Tuple(elems) => elems.iter().flat_map(collect_struct_deps).collect(),
        Type::Ref { inner, .. } | Type::Ptr { inner, .. } => collect_struct_deps(inner),
        Type::Alias(_, args) => args.iter().flat_map(collect_struct_deps).collect(),
        _ => Vec::new(),
    }
}

/// Resolve `use` directives, loading standard library modules as needed.
/// Returns a new Program with empty uses and a merged function list,
/// plus a map of function name overloads.
fn resolve_uses(program: Program, no_std: bool, _source_path: &str) -> (Program, OverloadMap) {
    let mut all_funcs: HashMap<String, Function> = HashMap::new();
    let mut all_overloads: OverloadMap = HashMap::new();
    let mut all_structs: HashMap<String, StructDecl> = HashMap::new();
    let mut all_enums: HashMap<String, EnumDecl> = HashMap::new();
    let mut all_impls: Vec<ImplDecl> = Vec::new();
    // Collect user-defined functions, structs, impls first
    for func in program.funcs {
        all_funcs.insert(func.name.clone(), func);
    }
    for decl in program.structs {
        all_structs.insert(decl.name.clone(), decl);
    }
    for decl in program.enums {
        all_enums.insert(decl.name.clone(), decl);
    }
    for decl in program.impls {
        all_impls.push(decl.clone());
    }

    // Cache all parsed stdlib modules for cross-module struct resolution.
    // Pre-load all stdlib modules so cross-module type references resolve.
    let mut all_stdlib_progs: HashMap<String, Program> = HashMap::new();
    // Map from struct name to canonical source module (e.g., "String" → "string")
    let mut canonical_struct_sources: HashMap<String, String> = HashMap::new();
    if let Ok(entries) = std::fs::read_dir("stdlib") {
        for entry in entries.flatten() {
            let path = entry.path();
            if path.extension().map_or(false, |e| e == "u") {
                if let Some(name) = path.file_stem().and_then(|s| s.to_str()) {
                    if let Ok(src) = std::fs::read_to_string(&path) {
                        let prog = lex_and_parse(&src, &path.to_string_lossy());
                        // Register canonical struct sources
                        for decl in &prog.structs {
                            canonical_struct_sources
                                .entry(decl.name.clone())
                                .or_insert_with(|| name.to_string());
                        }
                        all_stdlib_progs.entry(name.to_string()).or_insert(prog);
                    }
                }
            }
        }
    }

    // Resolve each use declaration
    for use_decl in &program.uses {
        let path = &use_decl.path;
        if path[0] == "std" {
            if no_std {
                eprintln!(
                    "error: use of std::{} requires the standard library (use --no-std)",
                    path[1..].join("::")
                );
                process::exit(1);
            }

            // Resolve stdlib/module_name.u
            let module_name = &path[1];
            let stdlib_path = Path::new("stdlib").join(format!("{}.u", module_name));

            let stdlib_src = match fs::read_to_string(&stdlib_path) {
                Ok(s) => s,
                Err(_) => {
                    eprintln!(
                        "error: cannot find std::{} module at '{}'",
                        module_name,
                        stdlib_path.display()
                    );
                    process::exit(1);
                }
            };

            let stdlib_prog = lex_and_parse(&stdlib_src, &stdlib_path.to_string_lossy());
            all_stdlib_progs
                .entry(module_name.clone())
                .or_insert_with(|| stdlib_prog.clone());

            // Recursively resolve this module's internal use declarations
            // (e.g., io.u uses string.u/String which uses vec.u/Vec which uses option.u/Option)
            let mut resolving_modules = HashSet::new();
            resolving_modules.insert(module_name.clone());
            resolve_module_uses(
                &stdlib_prog,
                &all_stdlib_progs,
                &mut all_structs,
                &mut all_enums,
                &mut all_impls,
                &mut all_funcs,
                &mut all_overloads,
                &canonical_struct_sources,
                &mut resolving_modules,
            );

            // Process duplicates and build overload map for this module
            let (module_funcs, module_overloads) = process_stdlib_functions(stdlib_prog.funcs);

            if path.len() == 2 {
                // Namespace import: use std::string;
                // Import all functions, structs, traits, impls
                for func in &module_funcs {
                    let qualified_name = format!("{}::{}", module_name, func.name);
                    all_funcs.entry(qualified_name).or_insert(func.clone());
                }
                // Register qualified overload names
                for (base_name, overloads_list) in &module_overloads {
                    let qualified_base = format!("{}::{}", module_name, base_name);
                    let qualified_list: Vec<(String, Vec<Type>)> = overloads_list
                        .iter()
                        .map(|(mangled, params)| {
                            (format!("{}::{}", module_name, mangled), params.clone())
                        })
                        .collect();
                    all_overloads.insert(qualified_base, qualified_list);
                }
                // Import all structs from the module
                for decl in stdlib_prog.structs {
                    all_structs.entry(decl.name.clone()).or_insert(decl);
                }
                // Import all enums from the module
                for decl in stdlib_prog.enums {
                    all_enums.entry(decl.name.clone()).or_insert(decl);
                }
                // Import all impls from the module
                for decl in stdlib_prog.impls {
                    all_impls.push(decl);
                }
                // Import all traits from the module
                for _trait_decl in stdlib_prog.traits {
                    // TODO: merge traits (not currently needed for basic String)
                }
                // Import all extern "C" functions from the module
                for func in &module_funcs {
                    if func.is_extern && !all_funcs.contains_key(&func.name) {
                        all_funcs.insert(func.name.clone(), func.clone());
                    }
                }
            } else {
                // Direct import: use std::string::String or std::io::println
                let target_name = &path[2];
                let qualified_name = format!("{}::{}", module_name, target_name);

                // Check if target is a struct or enum in the module
                let is_struct = stdlib_prog.structs.iter().any(|s| s.name == *target_name);
                let is_enum = stdlib_prog.enums.iter().any(|e| e.name == *target_name);

                if is_struct {
                    // Import the struct
                    for decl in stdlib_prog.structs {
                        if decl.name == *target_name {
                            all_structs.entry(decl.name.clone()).or_insert(decl);
                            break;
                        }
                    }
                    // Recursively import struct dependencies (e.g., String depends on Vec)
                    let mut dep_stack: Vec<String> = Vec::new();
                    if let Some(decl) = all_structs.get(target_name) {
                        for field in &decl.fields {
                            dep_stack.extend(collect_struct_deps(&field.ty));
                        }
                    }
                    // Also scan impl blocks for the target struct for additional deps
                    for impl_decl in &all_impls {
                        let impl_type_name = match &impl_decl.impl_type {
                            Type::Struct(name) => name.clone(),
                            Type::GenericInstance(name, _) => name.clone(),
                            _ => continue,
                        };
                        if impl_type_name == *target_name {
                            for method in &impl_decl.methods {
                                for param in &method.params {
                                    dep_stack.extend(collect_struct_deps(&param.ty));
                                }
                                if let Some(ref ret) = method.return_type {
                                    dep_stack.extend(collect_struct_deps(ret));
                                }
                            }
                        }
                    }
                    while let Some(dep_name) = dep_stack.pop() {
                        if all_structs.contains_key(&dep_name) {
                            continue;
                        }
                        // Look up canonical source module for this struct
                        if let Some(mod_name) = canonical_struct_sources.get(&dep_name) {
                            if let Some(candidate_prog) = all_stdlib_progs.get(mod_name) {
                                if let Some(dep_decl) =
                                    candidate_prog.structs.iter().find(|s| s.name == dep_name)
                                {
                                    all_structs
                                        .entry(dep_name.clone())
                                        .or_insert_with(|| dep_decl.clone());
                                    // Add its own field deps
                                    for field in &dep_decl.fields {
                                        dep_stack.extend(collect_struct_deps(&field.ty));
                                    }
                                    // Import impls for this dependency
                                    for impl_decl in &candidate_prog.impls {
                                        let impl_type_name = match &impl_decl.impl_type {
                                            Type::Struct(name) => name.clone(),
                                            Type::GenericInstance(name, _) => name.clone(),
                                            _ => continue,
                                        };
                                        if impl_type_name == dep_name {
                                            all_impls.push(impl_decl.clone());
                                        }
                                    }
                                    // Also scan the dependency's impl methods for further deps
                                    for impl_decl in &candidate_prog.impls {
                                        let impl_type_name = match &impl_decl.impl_type {
                                            Type::Struct(name) => name.clone(),
                                            Type::GenericInstance(name, _) => name.clone(),
                                            _ => continue,
                                        };
                                        if impl_type_name == dep_name {
                                            for method in &impl_decl.methods {
                                                for param in &method.params {
                                                    dep_stack
                                                        .extend(collect_struct_deps(&param.ty));
                                                }
                                                if let Some(ref ret) = method.return_type {
                                                    dep_stack.extend(collect_struct_deps(ret));
                                                }
                                            }
                                        }
                                    }
                                    // Import ALL functions from the dependency module
                                    for func in &candidate_prog.funcs {
                                        if func.is_extern && !all_funcs.contains_key(&func.name) {
                                            all_funcs.insert(func.name.clone(), func.clone());
                                        }
                                    }
                                    break;
                                }
                            }
                        }
                    }
                    // Import all impl blocks for this struct
                    for decl in stdlib_prog.impls {
                        let impl_type_name = match &decl.impl_type {
                            Type::Struct(name) => name.clone(),
                            _ => continue,
                        };
                        if impl_type_name == *target_name {
                            all_impls.push(decl);
                        }
                    }
                    // Import extern "C" declarations from the same module
                    for func in &module_funcs {
                        if func.is_extern && !all_funcs.contains_key(&func.name) {
                            all_funcs.insert(func.name.clone(), func.clone());
                        }
                    }
                } else if is_enum {
                    // Import the enum
                    for decl in stdlib_prog.enums {
                        if decl.name == *target_name {
                            all_enums.entry(decl.name.clone()).or_insert(decl);
                            break;
                        }
                    }
                    // Import all impl blocks for this enum
                    for decl in stdlib_prog.impls {
                        let impl_type_name = match &decl.impl_type {
                            Type::Struct(name) => name.clone(),
                            Type::GenericInstance(name, _) => name.clone(),
                            _ => continue,
                        };
                        if impl_type_name == *target_name {
                            all_impls.push(decl);
                        }
                    }
                    // Import extern "C" declarations from the same module
                    for func in &module_funcs {
                        if func.is_extern && !all_funcs.contains_key(&func.name) {
                            all_funcs.insert(func.name.clone(), func.clone());
                        }
                    }
                } else if let Some(overload_map) = module_overloads.get(target_name) {
                    // Overloaded function — import all variants under mangled names
                    // and register the overload map entry
                    for func in &module_funcs {
                        let base_end = func.name.find('$').unwrap_or(func.name.len());
                        if func.name[..base_end] == *target_name {
                            all_funcs
                                .entry(func.name.clone())
                                .or_insert_with(|| func.clone());
                            let qualified_mangled = format!("{}::{}", module_name, func.name);
                            all_funcs
                                .entry(qualified_mangled)
                                .or_insert_with(|| func.clone());
                        }
                        if func.is_extern {
                            let en = func.name.clone();
                            all_funcs.entry(en).or_insert_with(|| func.clone());
                        }
                    }
                    all_overloads.insert(target_name.clone(), overload_map.clone());

                    // Import struct types referenced by the overloaded function parameters.
                    // Resolve them from any stdlib module (they may be defined in another module).
                    let mut needed_structs: HashSet<String> = HashSet::new();
                    let mut dep_modules: HashSet<String> = HashSet::new();
                    for func in &module_funcs {
                        let base_end = func.name.find('$').unwrap_or(func.name.len());
                        if func.name[..base_end] == *target_name {
                            for param in &func.params {
                                needed_structs.extend(collect_struct_deps(&param.ty));
                            }
                            if let Some(ret) = &func.return_type {
                                needed_structs.extend(collect_struct_deps(ret));
                            }
                        }
                    }
                    for struct_name in &needed_structs {
                        if all_structs.contains_key(struct_name) {
                            continue;
                        }
                        // Look up canonical source module for this struct
                        if let Some(mod_name) = canonical_struct_sources.get(struct_name) {
                            if let Some(prog) = all_stdlib_progs.get(mod_name) {
                                if let Some(decl) =
                                    prog.structs.iter().find(|s| &s.name == struct_name)
                                {
                                    all_structs
                                        .entry(struct_name.clone())
                                        .or_insert_with(|| decl.clone());
                                    dep_modules.insert(mod_name.clone());
                                }
                            }
                        }
                    }
                    // Import extern "C" functions and impl blocks from dependency modules
                    for mod_name in &dep_modules {
                        if let Some(prog) = all_stdlib_progs.get(mod_name) {
                            // Extern functions
                            let (dep_funcs, _) = process_stdlib_functions(prog.funcs.clone());
                            for func in &dep_funcs {
                                if func.is_extern && !all_funcs.contains_key(&func.name) {
                                    all_funcs.insert(func.name.clone(), func.clone());
                                }
                            }
                            // Impl blocks
                            for decl in &prog.impls {
                                all_impls.push(decl.clone());
                            }
                        }
                    }
                } else {
                    // Single function import
                    let found = module_funcs.iter().find(|f| f.name == *target_name);
                    match found {
                        Some(func) => {
                            if !all_funcs.contains_key(target_name) {
                                all_funcs.insert(
                                    target_name.clone(),
                                    Function {
                                        name: target_name.clone(),
                                        ..func.clone()
                                    },
                                );
                                all_funcs.insert(qualified_name, func.clone());
                            }
                            for other in &module_funcs {
                                if other.name != *target_name
                                    && other.is_extern
                                    && !all_funcs.contains_key(&other.name)
                                {
                                    all_funcs.insert(other.name.clone(), other.clone());
                                }
                            }
                        }
                        None => {
                            eprintln!(
                                "error: '{}' not found in std::{}{}",
                                target_name,
                                module_name,
                                if target_name.chars().next().unwrap_or(' ').is_uppercase() {
                                    format!(
                                        " (struct '{}' has impl methods, try 'use std::{}::{}{}')",
                                        target_name,
                                        module_name,
                                        target_name,
                                        " or 'use std::{}' for all items",
                                    )
                                } else {
                                    String::new()
                                }
                            );
                            process::exit(1);
                        }
                    }
                }
            }
        }
    }

    (
        Program {
            uses: Vec::new(), // cleared after resolution
            funcs: all_funcs.into_values().collect(),
            structs: all_structs.into_values().collect(),
            enums: all_enums.into_values().collect(),
            traits: program.traits,
            impls: all_impls,
            type_aliases: Vec::new(),
        },
        all_overloads,
    )
}

fn main() {
    let cli = Cli::parse();

    let (mode, path) = match cli.command {
        Command::Run { file, opt } => (Mode::Run { opt }, file),
        Command::Build {
            file,
            output,
            opt,
            cc,
        } => (Mode::Build { output, opt, cc }, file),
        Command::BuildRun {
            file,
            output,
            opt,
            cc,
        } => (Mode::BuildRun { output, opt, cc }, file),
    };

    let source = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("error reading '{}': {}", path, e);
        process::exit(1);
    });

    // --- Lex & Parse ---
    let program = lex_and_parse(&source, &path);

    // --- Resolve uses (stdlib) ---
    let (program, overloads) = resolve_uses(program, cli.no_std, &path);

    let context = inkwell::context::Context::create();

    match mode {
        Mode::Run { opt } => {
            let opt_level: inkwell::OptimizationLevel = opt.into();
            let mut codegen = match codegen::CodeGen::new_jit(&context, opt_level) {
                Ok(cg) => cg,
                Err(msg) => {
                    eprintln!("codegen error: {}", msg);
                    process::exit(1);
                }
            };
            codegen.overloads = overloads;
            if let Err(msg) = codegen.jit_run(&program) {
                eprintln!("runtime error: {}", msg);
                process::exit(1);
            }
            println!("program executed successfully");
        }
        Mode::Build { output, opt, cc } => {
            let opt_level: inkwell::OptimizationLevel = opt.into();
            let mut codegen = codegen::CodeGen::new_native(&context, opt_level);
            codegen.overloads = overloads;

            if let Err(msg) = codegen.compile_module(&program) {
                eprintln!("codegen error: {}", msg);
                process::exit(1);
            }

            let exe_path = do_build(&mut codegen, output, cc);
            println!("compiled successfully: {exe_path}");
        }
        Mode::BuildRun { output, opt, cc } => {
            let opt_level: inkwell::OptimizationLevel = opt.into();
            let mut codegen = codegen::CodeGen::new_native(&context, opt_level);
            codegen.overloads = overloads;

            if let Err(msg) = codegen.compile_module(&program) {
                eprintln!("codegen error: {}", msg);
                process::exit(1);
            }

            let exe_path = do_build(&mut codegen, output, cc);

            let run_path = if exe_path.contains('/') {
                exe_path.clone()
            } else {
                format!("./{}", exe_path)
            };

            let status = std::process::Command::new(&run_path)
                .status()
                .unwrap_or_else(|e| {
                    eprintln!("failed to run '{}': {}", exe_path, e);
                    process::exit(1);
                });

            if !status.success() {
                process::exit(status.code().unwrap_or(1));
            }
        }
    }
}
