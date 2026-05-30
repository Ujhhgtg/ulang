mod ast;
mod codegen;
mod error;
mod lexer;
mod lsp;
mod parser;
mod token;

use annotate_snippets::renderer::{AnsiColor, Effects};
use clap::{Parser, Subcommand, builder::Styles};
use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::Path;
use std::process;

use crate::ast::{
    EnumDecl, Function, ImplDecl, Program, StructDecl, TraitDecl, Type, TypeAliasDecl,
};

type OverloadMap = HashMap<String, Vec<(String, Vec<Type>)>>;
use crate::token::{Span, Token};
use inkwell::targets::TargetTriple;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
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

use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct UlangToml {
    package: PackageConfig,
    #[serde(default)]
    build: BuildConfig,
    #[serde(default)]
    profile: ProfileConfig,
}

#[derive(Debug, Deserialize)]
struct PackageConfig {
    name: String,
}

#[derive(Debug, Deserialize, Default)]
struct BuildConfig {
    #[serde(default, deserialize_with = "deserialize_cc_opt")]
    cc: Option<Cc>,
}

fn deserialize_cc_opt<'de, D>(deserializer: D) -> Result<Option<Cc>, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct CcOptVisitor;
    impl<'de> serde::de::Visitor<'de> for CcOptVisitor {
        type Value = Option<Cc>;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("a string representation of Cc or null")
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            parse_cc(v).map(Some).map_err(serde::de::Error::custom)
        }
        fn visit_none<E>(self) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            Ok(None)
        }
        fn visit_some<D>(self, deserializer: D) -> Result<Self::Value, D::Error>
        where
            D: serde::Deserializer<'de>,
        {
            deserializer.deserialize_str(self)
        }
    }
    deserializer.deserialize_option(CcOptVisitor)
}

#[derive(Debug, Deserialize, Default)]
struct ProfileConfig {
    #[serde(default)]
    dev: ProfileDevConfig,
    #[serde(default)]
    release: ProfileReleaseConfig,
}

#[derive(Debug, Deserialize)]
struct ProfileDevConfig {
    #[serde(default = "default_dev_opt", deserialize_with = "deserialize_toml_opt")]
    opt_level: OptLevel,
}

impl Default for ProfileDevConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::None,
        }
    }
}

fn default_dev_opt() -> OptLevel {
    OptLevel::None
}

#[derive(Debug, Deserialize)]
struct ProfileReleaseConfig {
    #[serde(
        default = "default_release_opt",
        deserialize_with = "deserialize_toml_opt"
    )]
    opt_level: OptLevel,
}

impl Default for ProfileReleaseConfig {
    fn default() -> Self {
        Self {
            opt_level: OptLevel::Aggressive,
        }
    }
}

fn default_release_opt() -> OptLevel {
    OptLevel::Aggressive
}

fn deserialize_toml_opt<'de, D>(deserializer: D) -> Result<OptLevel, D::Error>
where
    D: serde::Deserializer<'de>,
{
    struct OptVisitor;
    impl<'de> serde::de::Visitor<'de> for OptVisitor {
        type Value = OptLevel;
        fn expecting(&self, formatter: &mut std::fmt::Formatter) -> std::fmt::Result {
            formatter.write_str("an integer (0-3) or string representing OptLevel")
        }
        fn visit_i64<E>(self, v: i64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            match v {
                0 => Ok(OptLevel::None),
                1 => Ok(OptLevel::Less),
                2 => Ok(OptLevel::Default),
                3 => Ok(OptLevel::Aggressive),
                _ => Err(serde::de::Error::custom(format!(
                    "invalid opt-level: {v}. Use 0, 1, 2, or 3"
                ))),
            }
        }
        fn visit_u64<E>(self, v: u64) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            self.visit_i64(v as i64)
        }
        fn visit_str<E>(self, v: &str) -> Result<Self::Value, E>
        where
            E: serde::de::Error,
        {
            parse_opt_level(v).map_err(serde::de::Error::custom)
        }
    }
    deserializer.deserialize_any(OptVisitor)
}

fn find_project_root() -> Option<std::path::PathBuf> {
    let mut dir = std::env::current_dir().ok()?;
    loop {
        let toml_path = dir.join("ulang.toml");
        if toml_path.is_file() {
            return Some(dir);
        }
        if !dir.pop() {
            break;
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

fn do_new_project(name: &str) {
    let project_dir = Path::new(name);
    if project_dir.exists() {
        eprintln!("error: directory '{}' already exists", name);
        process::exit(1);
    }

    if let Err(e) = fs::create_dir_all(project_dir.join("src")) {
        eprintln!("error: failed to create project directory structure: {}", e);
        process::exit(1);
    }

    let toml_content = format!(
        r#"[package]
name = "{}"

[profile.dev]
opt-level = "0"

[profile.release]
opt-level = "3"

[build]
cc = "gcc"
"#,
        name
    );

    let toml_path = project_dir.join("ulang.toml");
    if let Err(e) = fs::write(&toml_path, toml_content) {
        eprintln!("error: failed to write '{}': {}", toml_path.display(), e);
        process::exit(1);
    }

    let main_content = r#"use std::io::println;

fn main() {
    println("Hello, World!");
}
"#;

    let main_path = project_dir.join("src/main.u");
    if let Err(e) = fs::write(&main_path, main_content) {
        eprintln!("error: failed to write '{}': {}", main_path.display(), e);
        process::exit(1);
    }

    println!("Created binary project '{}'", name);
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
        /// Path to .u source file (optional in project mode)
        file: Option<String>,
        /// Optimization level (0|none, 1|less, 2|default, 3|aggressive)
        #[arg(short = 'o', long = "opt", value_parser = parse_opt_level)]
        opt: Option<OptLevel>,
        /// C compiler to use for linking (in project mode)
        #[arg(long = "cc", value_parser = parse_cc)]
        cc: Option<Cc>,
        /// Run as a single file script, even if inside a project
        #[arg(long = "script")]
        script: bool,
        /// Build and run in release mode
        #[arg(long = "release")]
        release: bool,
    },
    /// Compile to a native executable
    Build {
        /// Path to .u source file (optional in project mode)
        file: Option<String>,
        /// Output executable path [default: project name or a.out]
        #[arg(long = "output")]
        output: Option<String>,
        /// Optimization level (0|none, 1|less, 2|default, 3|aggressive)
        #[arg(short = 'o', long = "opt", value_parser = parse_opt_level)]
        opt: Option<OptLevel>,
        /// C compiler to use for linking (gcc, clang, cosmocc, zig, tcc)
        #[arg(long = "cc", value_parser = parse_cc)]
        cc: Option<Cc>,
        /// Run as a single file script, even if inside a project
        #[arg(long = "script")]
        script: bool,
        /// Build in release mode
        #[arg(long = "release")]
        release: bool,
    },
    /// Compile to a native executable and run it
    #[command(name = "build-run")]
    BuildRun {
        /// Path to .u source file (optional in project mode)
        file: Option<String>,
        /// Output executable path [default: a.out]
        #[arg(long = "output")]
        output: Option<String>,
        /// Optimization level (0|none, 1|less, 2|default, 3|aggressive)
        #[arg(short = 'o', long = "opt", default_value_t = OptLevel::Default, value_parser = parse_opt_level)]
        opt: OptLevel,
        /// C compiler to use for linking (gcc, clang, cosmocc, zig, tcc)
        #[arg(long = "cc", default_value_t = Cc::Gcc, value_parser = parse_cc)]
        cc: Cc,
        /// Run as a single file script, even if inside a project
        #[arg(long = "script")]
        script: bool,
    },
    /// Create a new project
    New {
        /// Name of the project
        name: String,
    },
    /// Emit LLVM IR for the source file
    #[command(name = "emit-ir")]
    EmitIr {
        /// Path to .u source file
        file: String,
        /// Optimization level (0|none, 1|less, 2|default, 3|aggressive)
        #[arg(short = 'o', long = "opt", value_parser = parse_opt_level)]
        opt: Option<OptLevel>,
    },
    /// Start the language server (LSP)
    Lsp,
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
    EmitIr {
        opt: OptLevel,
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
                error::emit_error_opt(
                    &codegen.source,
                    &codegen.path,
                    msg.span,
                    "codegen error",
                    &msg.msg,
                );
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
                error::emit_error_opt(
                    &codegen.source,
                    &codegen.path,
                    msg.span,
                    "codegen error",
                    &msg.msg,
                );
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
                error::emit_error_opt(
                    &codegen.source,
                    &codegen.path,
                    msg.span,
                    "link error",
                    &msg.msg,
                );
                let _ = std::fs::remove_dir_all(&aarch64_dir);
                let _ = fs::remove_file(&obj_path);
                process::exit(1);
            }

            let _ = std::fs::remove_dir_all(&aarch64_dir);
            let _ = fs::remove_file(&obj_path);
        }
        _ => {
            if let Err(msg) = codegen.compile_to_object(Path::new(&obj_path)) {
                error::emit_error_opt(
                    &codegen.source,
                    &codegen.path,
                    msg.span,
                    "codegen error",
                    &msg.msg,
                );
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
                error::emit_error_opt(
                    &codegen.source,
                    &codegen.path,
                    msg.span,
                    "link error",
                    &msg.msg,
                );
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
                let pos: usize = lexer.pos();
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
        if path.len() < 3 || !(path[0] == "std" || path[0] == "core") {
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
        Type::Ref { inner, .. }
        | Type::Ptr { inner, .. }
        | Type::Array { inner, .. }
        | Type::Slice { inner }
        | Type::GenericArray { inner, .. } => collect_struct_deps(inner),
        Type::Alias(_, args) => args.iter().flat_map(collect_struct_deps).collect(),
        Type::ImplTrait(bounds) => bounds
            .iter()
            .flat_map(|b| b.generic_args.iter().flat_map(collect_struct_deps))
            .collect(),
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
    let mut all_traits: HashMap<String, TraitDecl> = HashMap::new();
    let mut all_aliases: HashMap<String, TypeAliasDecl> = HashMap::new();

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
    for decl in program.traits {
        all_traits.entry(decl.name.clone()).or_insert(decl);
    }
    for decl in program.type_aliases {
        all_aliases.insert(decl.name.clone(), decl);
    }

    // Cache all parsed stdlib modules
    // Pre-load all stdlib modules so cross-module type references resolve.
    let mut all_stdlib_progs: HashMap<String, Program> = HashMap::new();
    // Map from struct name to canonical source module (e.g., "String" → "string")
    let mut canonical_struct_sources: HashMap<String, String> = HashMap::new();
    let stdlib_root_dir = find_stdlib_root();
    let std_dir = stdlib_root_dir.join("std");
    let core_dir = stdlib_root_dir.join("core");

    // Helper closure to load a root mod and its submodules
    let load_stdlib_dir = |dir: &std::path::Path,
                           all_progs: &mut HashMap<String, Program>,
                           sources: &mut HashMap<String, String>| {
        let mod_path = dir.join("mod.u");
        if let Ok(mod_src) = fs::read_to_string(&mod_path) {
            let stdlib_root = lex_and_parse(&mod_src, &mod_path.to_string_lossy());
            for m in &stdlib_root.modules {
                let name = &m.name;
                let sub_path = dir.join(format!("{}.u", name));
                match fs::read_to_string(&sub_path) {
                    Ok(src) => {
                        let prog = lex_and_parse(&src, &sub_path.to_string_lossy());
                        for decl in &prog.structs {
                            sources
                                .entry(decl.name.clone())
                                .or_insert_with(|| name.clone());
                        }
                        all_progs.insert(name.clone(), prog);
                    }
                    Err(_) => {
                        eprintln!(
                            "error: standard library module '{}' declared in 'mod.u' but not found at '{}'",
                            name,
                            sub_path.display()
                        );
                        process::exit(1);
                    }
                }
            }
        }
    };

    load_stdlib_dir(
        &core_dir,
        &mut all_stdlib_progs,
        &mut canonical_struct_sources,
    );
    load_stdlib_dir(
        &std_dir,
        &mut all_stdlib_progs,
        &mut canonical_struct_sources,
    );

    // Resolve each use declaration using a fixed-point loop to support transitive re-exports and nested uses
    let mut pending = program.uses.clone();
    let mut progress = true;
    while !pending.is_empty() && progress {
        progress = false;
        let mut unresolved = Vec::new();

        for use_decl in pending {
            let path = &use_decl.path;
            if path.is_empty() {
                continue;
            }

            let is_std = path[0] == "std";
            let is_core = path[0] == "core";
            if is_std || is_core {
                if is_std && no_std {
                    eprintln!(
                        "error: use of std::{} requires the standard library (use --no-std)",
                        path[1..].join("::")
                    );
                    process::exit(1);
                }

                // Resolve stdlib/module_name.u
                let module_name = &path[1];
                let stdlib_path = if is_core {
                    core_dir.join(format!("{}.u", module_name))
                } else {
                    let std_path = std_dir.join(format!("{}.u", module_name));
                    if std_path.exists() {
                        std_path
                    } else {
                        core_dir.join(format!("{}.u", module_name))
                    }
                };

                let stdlib_src = match fs::read_to_string(&stdlib_path) {
                    Ok(s) => s,
                    Err(_) => {
                        eprintln!(
                            "error: cannot find {}::{} module at '{}'",
                            path[0],
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
                let (module_funcs, module_overloads) =
                    process_stdlib_functions(stdlib_prog.funcs.clone());

                let prefix = if !use_decl.module_path.is_empty() {
                    format!("{}::", use_decl.module_path.join("::"))
                } else {
                    String::new()
                };

                if path.len() == 2 {
                    // Namespace import: use std::string;
                    let imported_module_name = if prefix.is_empty() {
                        module_name.clone()
                    } else {
                        format!("{}{}", prefix, module_name)
                    };

                    // Import all functions, structs, traits, impls
                    for func in &module_funcs {
                        let qualified_name = format!("{}::{}", imported_module_name, func.name);
                        let mut new_func = func.clone();
                        new_func.name = qualified_name.clone();
                        new_func.is_pub = use_decl.is_pub;
                        all_funcs.insert(qualified_name, new_func);
                    }
                    // Register qualified overload names
                    for (base_name, overloads_list) in &module_overloads {
                        let qualified_base = format!("{}::{}", imported_module_name, base_name);
                        let qualified_list: Vec<(String, Vec<Type>)> = overloads_list
                            .iter()
                            .map(|(mangled, params)| {
                                let base_end = mangled.find('$').unwrap_or(mangled.len());
                                let suffix = &mangled[base_end..];
                                (format!("{}{}", qualified_base, suffix), params.clone())
                            })
                            .collect();
                        all_overloads.insert(qualified_base, qualified_list);
                    }
                    // Import all structs from the module
                    for decl in &stdlib_prog.structs {
                        let new_name = if prefix.is_empty() {
                            format!("{}::{}", imported_module_name, decl.name)
                        } else {
                            format!("{}{}", prefix, decl.name)
                        };
                        let mut new_decl = decl.clone();
                        new_decl.name = new_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_structs.entry(new_name).or_insert(new_decl);
                    }
                    // Import all enums from the module
                    for decl in &stdlib_prog.enums {
                        let new_name = if prefix.is_empty() {
                            format!("{}::{}", imported_module_name, decl.name)
                        } else {
                            format!("{}{}", prefix, decl.name)
                        };
                        let mut new_decl = decl.clone();
                        new_decl.name = new_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_enums.entry(new_name).or_insert(new_decl);
                    }
                    // Import all impls from the module
                    for decl in &stdlib_prog.impls {
                        let mut new_decl = decl.clone();
                        match &mut new_decl.impl_type {
                            Type::Struct(name) => {
                                *name = format!("{}{}", prefix, name);
                            }
                            Type::GenericInstance(name, _) => {
                                *name = format!("{}{}", prefix, name);
                            }
                            _ => {}
                        }
                        all_impls.push(new_decl);
                    }
                    // Import all traits from the module
                    for decl in &stdlib_prog.traits {
                        let new_name = if prefix.is_empty() {
                            format!("{}::{}", imported_module_name, decl.name)
                        } else {
                            format!("{}{}", prefix, decl.name)
                        };
                        let mut new_decl = decl.clone();
                        new_decl.name = new_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_traits.entry(new_name).or_insert(new_decl);
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

                    // Check if target is a struct or enum in the module
                    let is_struct = stdlib_prog.structs.iter().any(|s| s.name == *target_name);
                    let is_enum = stdlib_prog.enums.iter().any(|e| e.name == *target_name);

                    if is_struct {
                        // Import the struct
                        for decl in &stdlib_prog.structs {
                            if decl.name == *target_name {
                                let new_name = if prefix.is_empty() {
                                    decl.name.clone()
                                } else {
                                    format!("{}{}", prefix, decl.name)
                                };
                                let mut new_decl = decl.clone();
                                new_decl.name = new_name.clone();
                                new_decl.is_pub = use_decl.is_pub;
                                all_structs.entry(new_name).or_insert(new_decl);
                                break;
                            }
                        }
                        // Recursively import struct dependencies (e.g., String depends on Vec)
                        let mut dep_stack: Vec<String> = Vec::new();
                        if let Some(decl) =
                            stdlib_prog.structs.iter().find(|s| s.name == *target_name)
                        {
                            for field in &decl.fields {
                                dep_stack.extend(collect_struct_deps(&field.ty));
                            }
                        }
                        // Also scan impl blocks for the target struct for additional deps
                        for impl_decl in &stdlib_prog.impls {
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
                            if let Some(mod_name) = canonical_struct_sources.get(&dep_name)
                                && let Some(candidate_prog) = all_stdlib_progs.get(mod_name)
                                && let Some(dep_decl) =
                                    candidate_prog.structs.iter().find(|s| s.name == dep_name)
                            {
                                all_structs
                                    .entry(dep_name.clone())
                                    .or_insert_with(|| dep_decl.clone());
                                for field in &dep_decl.fields {
                                    dep_stack.extend(collect_struct_deps(&field.ty));
                                }
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
                                for impl_decl in &candidate_prog.impls {
                                    let impl_type_name = match &impl_decl.impl_type {
                                        Type::Struct(name) => name.clone(),
                                        Type::GenericInstance(name, _) => name.clone(),
                                        _ => continue,
                                    };
                                    if impl_type_name == dep_name {
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
                                for func in &candidate_prog.funcs {
                                    if func.is_extern && !all_funcs.contains_key(&func.name) {
                                        all_funcs.insert(func.name.clone(), func.clone());
                                    }
                                }
                                break;
                            }
                        }
                        // Import all impl blocks for this struct
                        for decl in &stdlib_prog.impls {
                            let impl_type_name = match &decl.impl_type {
                                Type::Struct(name) => name.clone(),
                                Type::GenericInstance(name, _) => name.clone(),
                                _ => continue,
                            };
                            if impl_type_name == *target_name {
                                let mut new_decl = decl.clone();
                                let new_name = if prefix.is_empty() {
                                    impl_type_name.clone()
                                } else {
                                    format!("{}{}", prefix, impl_type_name)
                                };
                                match &mut new_decl.impl_type {
                                    Type::Struct(name) => *name = new_name,
                                    Type::GenericInstance(name, _) => *name = new_name,
                                    _ => {}
                                }
                                all_impls.push(new_decl);
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
                        for decl in &stdlib_prog.enums {
                            if decl.name == *target_name {
                                let new_name = if prefix.is_empty() {
                                    decl.name.clone()
                                } else {
                                    format!("{}{}", prefix, decl.name)
                                };
                                let mut new_decl = decl.clone();
                                new_decl.name = new_name.clone();
                                new_decl.is_pub = use_decl.is_pub;
                                all_enums.entry(new_name).or_insert(new_decl);
                                break;
                            }
                        }
                        // Import all impl blocks for this enum
                        for decl in &stdlib_prog.impls {
                            let impl_type_name = match &decl.impl_type {
                                Type::Struct(name) => name.clone(),
                                Type::GenericInstance(name, _) => name.clone(),
                                _ => continue,
                            };
                            if impl_type_name == *target_name {
                                let mut new_decl = decl.clone();
                                let new_name = if prefix.is_empty() {
                                    impl_type_name.clone()
                                } else {
                                    format!("{}{}", prefix, impl_type_name)
                                };
                                match &mut new_decl.impl_type {
                                    Type::Struct(name) => *name = new_name,
                                    Type::GenericInstance(name, _) => *name = new_name,
                                    _ => {}
                                }
                                all_impls.push(new_decl);
                            }
                        }
                        // Import extern "C" declarations from the same module
                        for func in &module_funcs {
                            if func.is_extern && !all_funcs.contains_key(&func.name) {
                                all_funcs.insert(func.name.clone(), func.clone());
                            }
                        }
                    } else if stdlib_prog.traits.iter().any(|t| t.name == *target_name) {
                        // Import the trait
                        for decl in &stdlib_prog.traits {
                            if decl.name == *target_name {
                                let new_name = if prefix.is_empty() {
                                    decl.name.clone()
                                } else {
                                    format!("{}{}", prefix, decl.name)
                                };
                                let mut new_decl = decl.clone();
                                new_decl.name = new_name.clone();
                                new_decl.is_pub = use_decl.is_pub;
                                all_traits.entry(new_name).or_insert(new_decl);
                                break;
                            }
                        }
                    } else if let Some(overload_map) = module_overloads.get(target_name) {
                        let new_name = if prefix.is_empty() {
                            target_name.clone()
                        } else {
                            format!("{}{}", prefix, target_name)
                        };
                        for func in &module_funcs {
                            let base_end = func.name.find('$').unwrap_or(func.name.len());
                            if func.name[..base_end] == *target_name {
                                let suffix = &func.name[base_end..];
                                let mangled_name = format!("{}{}", new_name, suffix);
                                let mut new_func = func.clone();
                                new_func.name = mangled_name.clone();
                                new_func.is_pub = use_decl.is_pub;
                                all_funcs.insert(mangled_name, new_func);
                            }
                            if func.is_extern {
                                let en = func.name.clone();
                                all_funcs.entry(en).or_insert_with(|| func.clone());
                            }
                        }
                        let mapped_list: Vec<(String, Vec<Type>)> = overload_map
                            .iter()
                            .map(|(mangled, params)| {
                                let base_end = mangled.find('$').unwrap_or(mangled.len());
                                let suffix = &mangled[base_end..];
                                (format!("{}{}", new_name, suffix), params.clone())
                            })
                            .collect();
                        all_overloads.insert(new_name, mapped_list);

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
                            if let Some(mod_name) = canonical_struct_sources.get(struct_name)
                                && let Some(prog) = all_stdlib_progs.get(mod_name)
                                && let Some(decl) =
                                    prog.structs.iter().find(|s| &s.name == struct_name)
                            {
                                all_structs
                                    .entry(struct_name.clone())
                                    .or_insert_with(|| decl.clone());
                                dep_modules.insert(mod_name.clone());
                            }
                        }
                        for mod_name in &dep_modules {
                            if let Some(prog) = all_stdlib_progs.get(mod_name) {
                                let (dep_funcs, _) = process_stdlib_functions(prog.funcs.clone());
                                for func in &dep_funcs {
                                    if func.is_extern && !all_funcs.contains_key(&func.name) {
                                        all_funcs.insert(func.name.clone(), func.clone());
                                    }
                                }
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
                                let new_name = if prefix.is_empty() {
                                    target_name.clone()
                                } else {
                                    format!("{}{}", prefix, target_name)
                                };
                                let mut new_func = func.clone();
                                new_func.name = new_name.clone();
                                new_func.is_pub = use_decl.is_pub;
                                all_funcs.insert(new_name, new_func);

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
                progress = true;
            } else {
                // Local import!
                let source_name_segs = if path[0] == "crate" {
                    path[1..].to_vec()
                } else if !use_decl.module_path.is_empty() {
                    let mut segs = use_decl.module_path.clone();
                    segs.extend(path.clone());
                    segs
                } else {
                    path.clone()
                };
                let source_name = source_name_segs.join("::");

                let last_seg = path.last().unwrap();
                let imported_name = if !use_decl.module_path.is_empty() {
                    format!("{}::{}", use_decl.module_path.join("::"), last_seg)
                } else {
                    last_seg.clone()
                };

                let has_overloads = all_overloads.contains_key(&source_name);
                let has_funcs = all_funcs.contains_key(&source_name)
                    || all_funcs
                        .keys()
                        .any(|k| k.starts_with(&format!("{}$", source_name)));
                let has_structs = all_structs.contains_key(&source_name);
                let has_enums = all_enums.contains_key(&source_name);
                let has_traits = all_traits.contains_key(&source_name);
                let has_aliases = all_aliases.contains_key(&source_name);

                let prefix_match = format!("{}::", source_name);
                let is_namespace = all_funcs.keys().any(|k| k.starts_with(&prefix_match))
                    || all_structs.keys().any(|k| k.starts_with(&prefix_match))
                    || all_enums.keys().any(|k| k.starts_with(&prefix_match))
                    || all_traits.keys().any(|k| k.starts_with(&prefix_match))
                    || all_aliases.keys().any(|k| k.starts_with(&prefix_match));

                let mut found_any = false;

                // 1. Copy functions
                if has_overloads || has_funcs {
                    found_any = true;
                    if let Some(overloads_list) = all_overloads.get(&source_name).cloned() {
                        let new_list: Vec<(String, Vec<Type>)> = overloads_list
                            .iter()
                            .map(|(mangled, params)| {
                                let suffix = mangled.split('$').next_back().unwrap();
                                let new_mangled = format!("{}${}", imported_name, suffix);
                                (new_mangled, params.clone())
                            })
                            .collect();
                        all_overloads.insert(imported_name.clone(), new_list);

                        for (mangled, _) in &overloads_list {
                            if let Some(func) = all_funcs.get(mangled).cloned() {
                                let suffix = mangled.split('$').next_back().unwrap();
                                let new_mangled = format!("{}${}", imported_name, suffix);
                                let mut new_func = func;
                                new_func.name = new_mangled.clone();
                                new_func.is_pub = use_decl.is_pub;
                                all_funcs.insert(new_mangled, new_func);
                            }
                        }
                    } else if let Some(func) = all_funcs.get(&source_name).cloned() {
                        let mut new_func = func;
                        new_func.name = imported_name.clone();
                        new_func.is_pub = use_decl.is_pub;
                        all_funcs.insert(imported_name.clone(), new_func);
                    }
                }

                // 2. Copy structs
                if has_structs {
                    found_any = true;
                    if let Some(decl) = all_structs.get(&source_name).cloned() {
                        let mut new_decl = decl;
                        new_decl.name = imported_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_structs.insert(imported_name.clone(), new_decl);
                    }
                }

                // 3. Copy enums
                if has_enums {
                    found_any = true;
                    if let Some(decl) = all_enums.get(&source_name).cloned() {
                        let mut new_decl = decl;
                        new_decl.name = imported_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_enums.insert(imported_name.clone(), new_decl);
                    }
                }

                // 4. Copy traits
                if has_traits {
                    found_any = true;
                    if let Some(decl) = all_traits.get(&source_name).cloned() {
                        let mut new_decl = decl;
                        new_decl.name = imported_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_traits.insert(imported_name.clone(), new_decl);
                    }
                }

                // 5. Copy type aliases
                if has_aliases {
                    found_any = true;
                    if let Some(decl) = all_aliases.get(&source_name).cloned() {
                        let mut new_decl = decl;
                        new_decl.name = imported_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_aliases.insert(imported_name.clone(), new_decl);
                    }
                }

                // 6. Copy namespace/module children
                if is_namespace {
                    found_any = true;

                    // Functions
                    let matching_funcs: Vec<(String, Function)> = all_funcs
                        .iter()
                        .filter(|(k, _)| k.starts_with(&prefix_match))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    for (k, func) in matching_funcs {
                        let relative = k.strip_prefix(&prefix_match).unwrap();
                        let new_name = format!("{}::{}", imported_name, relative);
                        let mut new_func = func;
                        new_func.name = new_name.clone();
                        new_func.is_pub = use_decl.is_pub;
                        all_funcs.insert(new_name, new_func);
                    }

                    // Overloads
                    let matching_overloads: Vec<(String, Vec<(String, Vec<Type>)>)> = all_overloads
                        .iter()
                        .filter(|(k, _)| k.starts_with(&prefix_match))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    for (k, overloads_list) in matching_overloads {
                        let relative = k.strip_prefix(&prefix_match).unwrap();
                        let new_name = format!("{}::{}", imported_name, relative);
                        let new_list: Vec<(String, Vec<Type>)> = overloads_list
                            .iter()
                            .map(|(mangled, params)| {
                                let rel_mangled = mangled.strip_prefix(&prefix_match).unwrap();
                                let new_mangled = format!("{}::{}", imported_name, rel_mangled);
                                (new_mangled, params.clone())
                            })
                            .collect();
                        all_overloads.insert(new_name, new_list);
                    }

                    // Structs
                    let matching_structs: Vec<(String, StructDecl)> = all_structs
                        .iter()
                        .filter(|(k, _)| k.starts_with(&prefix_match))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    for (k, decl) in matching_structs {
                        let relative = k.strip_prefix(&prefix_match).unwrap();
                        let new_name = format!("{}::{}", imported_name, relative);
                        let mut new_decl = decl;
                        new_decl.name = new_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_structs.insert(new_name, new_decl);
                    }

                    // Enums
                    let matching_enums: Vec<(String, EnumDecl)> = all_enums
                        .iter()
                        .filter(|(k, _)| k.starts_with(&prefix_match))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    for (k, decl) in matching_enums {
                        let relative = k.strip_prefix(&prefix_match).unwrap();
                        let new_name = format!("{}::{}", imported_name, relative);
                        let mut new_decl = decl;
                        new_decl.name = new_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_enums.insert(new_name, new_decl);
                    }

                    // Traits
                    let matching_traits: Vec<(String, TraitDecl)> = all_traits
                        .iter()
                        .filter(|(k, _)| k.starts_with(&prefix_match))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    for (k, decl) in matching_traits {
                        let relative = k.strip_prefix(&prefix_match).unwrap();
                        let new_name = format!("{}::{}", imported_name, relative);
                        let mut new_decl = decl;
                        new_decl.name = new_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_traits.insert(new_name, new_decl);
                    }

                    // Aliases
                    let matching_aliases: Vec<(String, TypeAliasDecl)> = all_aliases
                        .iter()
                        .filter(|(k, _)| k.starts_with(&prefix_match))
                        .map(|(k, v)| (k.clone(), v.clone()))
                        .collect();
                    for (k, decl) in matching_aliases {
                        let relative = k.strip_prefix(&prefix_match).unwrap();
                        let new_name = format!("{}::{}", imported_name, relative);
                        let mut new_decl = decl;
                        new_decl.name = new_name.clone();
                        new_decl.is_pub = use_decl.is_pub;
                        all_aliases.insert(new_name, new_decl);
                    }
                }

                if found_any {
                    progress = true;
                } else {
                    unresolved.push(use_decl);
                }
            }
        }
        pending = unresolved;
    }

    (
        Program {
            uses: Vec::new(), // cleared after resolution
            modules: Vec::new(),
            funcs: all_funcs.into_values().collect(),
            structs: all_structs.into_values().collect(),
            enums: all_enums.into_values().collect(),
            traits: all_traits.into_values().collect(),
            impls: all_impls,
            type_aliases: all_aliases.into_values().collect(),
        },
        all_overloads,
    )
}

fn main() {
    let cli = Cli::parse();

    if let Command::New { name } = &cli.command {
        do_new_project(name);
        return;
    }

    if let Command::Lsp = cli.command {
        if let Err(e) = lsp::run_server() {
            eprintln!("LSP server error: {}", e);
            process::exit(1);
        }
        return;
    }

    let project_root = find_project_root();
    let is_script_mode = match &cli.command {
        Command::Run { script, .. } => *script,
        Command::Build { script, .. } => *script,
        Command::BuildRun { script, .. } => *script,
        Command::EmitIr { .. } => true,
        _ => false,
    };

    let (mode, path) = if let Some(root_path) = project_root {
        if is_script_mode {
            // Script / Single-file Mode inside a project folder
            let file_opt = match &cli.command {
                Command::Run { file, .. } => file.clone(),
                Command::Build { file, .. } => file.clone(),
                Command::BuildRun { file, .. } => file.clone(),
                Command::EmitIr { file, .. } => Some(file.clone()),
                _ => None,
            };
            let path = match file_opt {
                Some(f) => f,
                None => {
                    eprintln!("error: script mode requires a source file path");
                    process::exit(1);
                }
            };

            let mode = match &cli.command {
                Command::Run { opt, .. } => Mode::Run {
                    opt: opt.unwrap_or(OptLevel::Default),
                },
                Command::Build {
                    output, opt, cc, ..
                } => Mode::Build {
                    output: output.clone(),
                    opt: opt.unwrap_or(OptLevel::Default),
                    cc: cc.unwrap_or(Cc::Gcc),
                },
                Command::BuildRun {
                    output, opt, cc, ..
                } => Mode::BuildRun {
                    output: output.clone(),
                    opt: *opt,
                    cc: *cc,
                },
                Command::EmitIr { opt, .. } => Mode::EmitIr {
                    opt: opt.unwrap_or(OptLevel::Default),
                },
                _ => unreachable!(),
            };

            (mode, path)
        } else {
            // Project Mode
            let toml_path = root_path.join("ulang.toml");
            let toml_str = fs::read_to_string(&toml_path).unwrap_or_else(|e| {
                eprintln!("error: failed to read ulang.toml: {}", e);
                process::exit(1);
            });
            let config: UlangToml = toml::from_str(&toml_str).unwrap_or_else(|e| {
                eprintln!("error: failed to parse '{}': {}", toml_path.display(), e);
                process::exit(1);
            });

            if let Command::BuildRun { .. } = &cli.command {
                eprintln!("error: `build-run` is disabled in projects, use `run` instead");
                process::exit(1);
            }

            let file_opt = match &cli.command {
                Command::Run { file, .. } => file.clone(),
                Command::Build { file, .. } => file.clone(),
                _ => None,
            };
            if file_opt.is_some() {
                eprintln!(
                    "error: found a project but a source file was specified. Use '--script' to run in script mode, or run without a file argument for project mode."
                );
                process::exit(1);
            }

            let release = match &cli.command {
                Command::Run { release, .. } => *release,
                Command::Build { release, .. } => *release,
                _ => false,
            };

            let opt = match &cli.command {
                Command::Run { opt: Some(o), .. } | Command::Build { opt: Some(o), .. } => *o,
                _ => {
                    if release {
                        config.profile.release.opt_level
                    } else {
                        config.profile.dev.opt_level
                    }
                }
            };

            let cc = match &cli.command {
                Command::Run { cc: Some(c), .. } | Command::Build { cc: Some(c), .. } => *c,
                _ => config.build.cc.unwrap_or(Cc::Gcc),
            };

            let exe_name = match &cli.command {
                Command::Build {
                    output: Some(o), ..
                } => o.clone(),
                _ => config.package.name.clone(),
            };

            let source_path_raw = root_path.join("src/main.u");
            let source_path = fs::canonicalize(&source_path_raw).unwrap_or_else(|e| {
                eprintln!("error: cannot find root module 'src/main.u': {}", e);
                process::exit(1);
            });

            let target_dir = if release {
                root_path.join("target/release")
            } else {
                root_path.join("target/debug")
            };
            fs::create_dir_all(&target_dir).unwrap_or_else(|e| {
                eprintln!("error: failed to create target directory: {}", e);
                process::exit(1);
            });
            let canonical_target_dir = fs::canonicalize(&target_dir).unwrap_or_else(|e| {
                eprintln!("error: failed to canonicalize target directory: {}", e);
                process::exit(1);
            });
            std::env::set_current_dir(&canonical_target_dir).unwrap_or_else(|e| {
                eprintln!(
                    "error: failed to set target directory as working dir: {}",
                    e
                );
                process::exit(1);
            });

            let mode = match &cli.command {
                Command::Build { .. } => Mode::Build {
                    output: Some(exe_name),
                    opt,
                    cc,
                },
                Command::Run { .. } => Mode::BuildRun {
                    output: Some(exe_name),
                    opt,
                    cc,
                },
                _ => unreachable!(),
            };

            (mode, source_path.to_string_lossy().into_owned())
        }
    } else {
        // Script / Single-file Mode
        let file_opt = match &cli.command {
            Command::Run { file, .. } => file.clone(),
            Command::Build { file, .. } => file.clone(),
            Command::BuildRun { file, .. } => file.clone(),
            Command::EmitIr { file, .. } => Some(file.clone()),
            _ => None,
        };
        let path = match file_opt {
            Some(f) => f,
            None => {
                eprintln!(
                    "error: no ulang.toml found and no source file specified. Use 'ulang new <name>' to create a project or specify a file to compile."
                );
                process::exit(1);
            }
        };

        let mode = match &cli.command {
            Command::Run { opt, .. } => Mode::Run {
                opt: opt.unwrap_or(OptLevel::Default),
            },
            Command::Build {
                output, opt, cc, ..
            } => Mode::Build {
                output: output.clone(),
                opt: opt.unwrap_or(OptLevel::Default),
                cc: cc.unwrap_or(Cc::Gcc),
            },
            Command::BuildRun {
                output, opt, cc, ..
            } => Mode::BuildRun {
                output: output.clone(),
                opt: *opt,
                cc: *cc,
            },
            Command::EmitIr { opt, .. } => Mode::EmitIr {
                opt: opt.unwrap_or(OptLevel::Default),
            },
            _ => unreachable!(),
        };

        (mode, path)
    };

    let source = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("error reading '{}': {}", path, e);
        process::exit(1);
    });

    // --- Lex & Parse ---
    let program = lex_and_parse(&source, &path);

    // --- Resolve Modules & Flatten ---
    let program = resolve_and_flatten_modules(program, &path);

    // --- Resolve uses (stdlib) ---
    let (program, overloads) = resolve_uses(program, cli.no_std, &path);

    let context = inkwell::context::Context::create();

    match mode {
        Mode::Run { opt } => {
            let opt_level: inkwell::OptimizationLevel = opt.into();
            let mut codegen = match codegen::CodeGen::new_jit(
                &context,
                opt_level,
                source.clone(),
                path.clone(),
            ) {
                Ok(cg) => cg,
                Err(e) => {
                    error::emit_error_opt(&source, &path, e.span, "codegen error", &e.msg);
                    process::exit(1);
                }
            };
            codegen.overloads = overloads;
            if let Err(e) = codegen.jit_run(&program) {
                error::emit_error_opt(&source, &path, e.span, "jit runtime error", &e.msg);
                process::exit(1);
            }
            println!("program executed successfully");
        }
        Mode::Build { output, opt, cc } => {
            let opt_level: inkwell::OptimizationLevel = opt.into();
            let mut codegen =
                codegen::CodeGen::new_native(&context, opt_level, source.clone(), path.clone());
            codegen.overloads = overloads;

            if let Err(e) = codegen.compile_module(&program) {
                error::emit_error_opt(&source, &path, e.span, "codegen error", &e.msg);
                process::exit(1);
            }

            let exe_path = do_build(&mut codegen, output, cc);
            println!("compiled successfully: {exe_path}");
        }
        Mode::BuildRun { output, opt, cc } => {
            let opt_level: inkwell::OptimizationLevel = opt.into();
            let mut codegen =
                codegen::CodeGen::new_native(&context, opt_level, source.clone(), path.clone());
            codegen.overloads = overloads;

            if let Err(e) = codegen.compile_module(&program) {
                error::emit_error_opt(&source, &path, e.span, "codegen error", &e.msg);
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
        Mode::EmitIr { opt } => {
            let opt_level: inkwell::OptimizationLevel = opt.into();
            let mut codegen =
                codegen::CodeGen::new_native(&context, opt_level, source.clone(), path.clone());
            codegen.overloads = overloads;

            if let Err(e) = codegen.compile_module(&program) {
                error::emit_error_opt(&source, &path, e.span, "codegen error", &e.msg);
                process::exit(1);
            }

            let ir = codegen.emit_ir();
            print!("{}", ir);
        }
    }
}

fn resolve_modules(program: &mut Program, source_path: &str) {
    for m in &mut program.modules {
        if m.body.is_none() {
            let parent_dir = Path::new(source_path).parent().unwrap_or(Path::new("."));
            let mod_path_file = parent_dir.join(format!("{}.u", m.name));
            let mod_path_dir = parent_dir.join(&m.name).join("mod.u");

            let (mod_path, mod_src) = if let Ok(s) = fs::read_to_string(&mod_path_file) {
                (mod_path_file, s)
            } else if let Ok(s) = fs::read_to_string(&mod_path_dir) {
                (mod_path_dir, s)
            } else {
                eprintln!(
                    "error: cannot find module file for '{}' under '{}' or '{}'",
                    m.name,
                    mod_path_file.display(),
                    mod_path_dir.display()
                );
                process::exit(1);
            };

            let mut mod_prog = lex_and_parse(&mod_src, &mod_path.to_string_lossy());
            resolve_modules(&mut mod_prog, &mod_path.to_string_lossy());
            m.body = Some(mod_prog);
        } else if let Some(ref mut body) = m.body {
            resolve_modules(body, source_path);
        }
    }
}

fn qualify_type(ty: &mut Type, local_types: &HashSet<String>, prefix: &str) {
    if prefix.is_empty() {
        return;
    }
    match ty {
        Type::Tuple(elems) => {
            for elem in elems {
                qualify_type(elem, local_types, prefix);
            }
        }
        Type::Ptr { inner, .. }
        | Type::Ref { inner, .. }
        | Type::Array { inner, .. }
        | Type::Slice { inner }
        | Type::GenericArray { inner, .. } => {
            qualify_type(inner, local_types, prefix);
        }
        Type::Struct(name) => {
            if local_types.contains(name) {
                *name = format!("{}::{}", prefix, name);
            }
        }
        Type::GenericInstance(name, args) => {
            if local_types.contains(name) {
                *name = format!("{}::{}", prefix, name);
            }
            for arg in args {
                qualify_type(arg, local_types, prefix);
            }
        }
        Type::Alias(name, args) => {
            if local_types.contains(name) {
                *name = format!("{}::{}", prefix, name);
            }
            for arg in args {
                qualify_type(arg, local_types, prefix);
            }
        }
        Type::ImplTrait(bounds) => {
            for bound in bounds {
                if local_types.contains(&bound.trait_name) {
                    bound.trait_name = format!("{}::{}", prefix, bound.trait_name);
                }
                for arg in &mut bound.generic_args {
                    qualify_type(arg, local_types, prefix);
                }
            }
        }
        _ => {}
    }
}

fn qualify_pattern(
    pattern: &mut crate::ast::Pattern,
    local_types: &HashSet<String>,
    prefix: &str,
    current_path: &[String],
    submodules: &HashSet<String>,
    top_level_modules: &HashSet<String>,
) {
    if let crate::ast::Pattern::EnumVariant {
        enum_name, payload, ..
    } = pattern
    {
        if let Some(name) = enum_name {
            if !prefix.is_empty() && local_types.contains(name) {
                *name = format!("{}::{}", prefix, name);
            }
            let segments: Vec<String> = name.split("::").map(|s| s.to_string()).collect();
            if segments.len() > 1 {
                let resolved_path = if segments[0] == "crate" {
                    segments[1..].to_vec()
                } else if submodules.contains(&segments[0]) {
                    let mut path = current_path.to_vec();
                    path.extend(segments);
                    path
                } else if top_level_modules.contains(&segments[0]) {
                    segments
                } else {
                    segments
                };
                *name = resolved_path.join("::");
            }
        }
        if let Some(p) = payload {
            qualify_pattern(
                p,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
    }
}

fn qualify_expr(
    expr: &mut crate::ast::Expr,
    local_funcs: &HashSet<String>,
    local_types: &HashSet<String>,
    prefix: &str,
    current_path: &[String],
    submodules: &HashSet<String>,
    top_level_modules: &HashSet<String>,
) {
    match expr {
        crate::ast::Expr::Call { callee, args, .. } => {
            if !prefix.is_empty() && local_funcs.contains(callee) {
                *callee = format!("{}::{}", prefix, callee);
            }
            for arg in args {
                qualify_expr(
                    arg,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        crate::ast::Expr::QualifiedCall {
            module,
            callee: _,
            args,
            ..
        } => {
            let segments: Vec<String> = module.split("::").map(|s| s.to_string()).collect();
            if !segments.is_empty() {
                let resolved_path = if segments[0] == "crate" {
                    segments[1..].to_vec()
                } else if submodules.contains(&segments[0]) {
                    let mut path = current_path.to_vec();
                    path.extend(segments);
                    path
                } else if top_level_modules.contains(&segments[0]) {
                    segments
                } else {
                    segments
                };
                if !resolved_path.is_empty() {
                    *module = resolved_path.join("::");
                }
            }
            for arg in args {
                qualify_expr(
                    arg,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        crate::ast::Expr::StructLit {
            struct_name,
            fields,
            ..
        } => {
            if !prefix.is_empty() && local_types.contains(struct_name) {
                *struct_name = format!("{}::{}", prefix, struct_name);
            }
            let segments: Vec<String> = struct_name.split("::").map(|s| s.to_string()).collect();
            if segments.len() > 1 {
                let resolved_path = if segments[0] == "crate" {
                    segments[1..].to_vec()
                } else if submodules.contains(&segments[0]) {
                    let mut path = current_path.to_vec();
                    path.extend(segments);
                    path
                } else if top_level_modules.contains(&segments[0]) {
                    segments
                } else {
                    segments
                };
                *struct_name = resolved_path.join("::");
            }
            for (_, val) in fields {
                qualify_expr(
                    val,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        crate::ast::Expr::EnumLit {
            enum_name, payload, ..
        } => {
            if !prefix.is_empty() && local_types.contains(enum_name) {
                *enum_name = format!("{}::{}", prefix, enum_name);
            }
            let segments: Vec<String> = enum_name.split("::").map(|s| s.to_string()).collect();
            if segments.len() > 1 {
                let resolved_path = if segments[0] == "crate" {
                    segments[1..].to_vec()
                } else if submodules.contains(&segments[0]) {
                    let mut path = current_path.to_vec();
                    path.extend(segments);
                    path
                } else if top_level_modules.contains(&segments[0]) {
                    segments
                } else {
                    segments
                };
                *enum_name = resolved_path.join("::");
            }
            if let Some(payload_expr) = payload {
                qualify_expr(
                    payload_expr,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        crate::ast::Expr::Binary { lhs, rhs, .. } => {
            qualify_expr(
                lhs,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            qualify_expr(
                rhs,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::Assign { target, value, .. } => {
            qualify_expr(
                target,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            qualify_expr(
                value,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::Ref { expr, .. } => {
            qualify_expr(
                expr,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::UnaryNot(expr, ..) => {
            qualify_expr(
                expr,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::UnaryMinus(expr, ..) => {
            qualify_expr(
                expr,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::Deref(expr, ..) => {
            qualify_expr(
                expr,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::Cast { expr, to_type, .. } => {
            qualify_expr(
                expr,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            qualify_type(to_type, local_types, prefix);
        }
        crate::ast::Expr::Tuple(elems, ..) => {
            for elem in elems {
                qualify_expr(
                    elem,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        crate::ast::Expr::Member { expr, .. } => {
            qualify_expr(
                expr,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::MethodCall { expr, args, .. } => {
            qualify_expr(
                expr,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            for arg in args {
                qualify_expr(
                    arg,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        crate::ast::Expr::If {
            cond,
            then_block,
            else_ifs,
            else_block,
            ..
        } => {
            qualify_expr(
                cond,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            qualify_block(
                then_block,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            for (elif_cond, elif_block) in else_ifs {
                qualify_expr(
                    elif_cond,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
                qualify_block(
                    elif_block,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
            if let Some(block) = else_block {
                qualify_block(
                    block,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        crate::ast::Expr::Loop { body, .. } => {
            qualify_block(
                body,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::While { cond, body, .. } => {
            qualify_expr(
                cond,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            qualify_block(
                body,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::Array(elems, ..) => {
            for elem in elems {
                qualify_expr(
                    elem,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        crate::ast::Expr::Repeat(expr, ..) => {
            qualify_expr(
                expr,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::Index { array, index, .. } => {
            qualify_expr(
                array,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            qualify_expr(
                index,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
        }
        crate::ast::Expr::IfLet {
            pattern,
            scrutinee,
            then_block,
            else_block,
            ..
        } => {
            qualify_pattern(
                pattern,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            qualify_expr(
                scrutinee,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            qualify_block(
                then_block,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            if let Some(block) = else_block {
                qualify_block(
                    block,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        crate::ast::Expr::Match {
            scrutinee, arms, ..
        } => {
            qualify_expr(
                scrutinee,
                local_funcs,
                local_types,
                prefix,
                current_path,
                submodules,
                top_level_modules,
            );
            for arm in arms {
                qualify_pattern(
                    &mut arm.pattern,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
                if let Some(guard_expr) = &mut arm.guard {
                    qualify_expr(
                        guard_expr,
                        local_funcs,
                        local_types,
                        prefix,
                        current_path,
                        submodules,
                        top_level_modules,
                    );
                }
                qualify_block(
                    &mut arm.body,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
        }
        _ => {}
    }
}

fn qualify_block(
    block: &mut crate::ast::Block,
    local_funcs: &HashSet<String>,
    local_types: &HashSet<String>,
    prefix: &str,
    current_path: &[String],
    submodules: &HashSet<String>,
    top_level_modules: &HashSet<String>,
) {
    for stmt in &mut block.stmts {
        match stmt {
            crate::ast::Stmt::Let { type_ann, init, .. } => {
                if let Some(ty) = type_ann {
                    qualify_type(ty, local_types, prefix);
                }
                qualify_expr(
                    init,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
            crate::ast::Stmt::Const { type_ann, init, .. } => {
                if let Some(ty) = type_ann {
                    qualify_type(ty, local_types, prefix);
                }
                qualify_expr(
                    init,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
            crate::ast::Stmt::Expr(expr) => {
                qualify_expr(
                    expr,
                    local_funcs,
                    local_types,
                    prefix,
                    current_path,
                    submodules,
                    top_level_modules,
                );
            }
            crate::ast::Stmt::Return { value, .. } => {
                if let Some(expr) = value {
                    qualify_expr(
                        expr,
                        local_funcs,
                        local_types,
                        prefix,
                        current_path,
                        submodules,
                        top_level_modules,
                    );
                }
            }
            crate::ast::Stmt::Continue { .. } => {}
            crate::ast::Stmt::Break { value, .. } => {
                if let Some(expr) = value {
                    qualify_expr(
                        expr,
                        local_funcs,
                        local_types,
                        prefix,
                        current_path,
                        submodules,
                        top_level_modules,
                    );
                }
            }
        }
    }
    if let Some(tail) = &mut block.tail_expr {
        qualify_expr(
            tail,
            local_funcs,
            local_types,
            prefix,
            current_path,
            submodules,
            top_level_modules,
        );
    }
}

fn flatten_module(
    program: Program,
    current_path: Vec<String>,
    root_program: &mut Program,
    top_level_modules: &HashSet<String>,
) {
    let mut submodules: HashSet<String> = program.modules.iter().map(|m| m.name.clone()).collect();

    let mut imported_funcs = HashSet::new();
    let mut imported_types = HashSet::new();
    for use_decl in &program.uses {
        if use_decl.path.len() >= 2 {
            let last = use_decl.path.last().unwrap();
            if last.chars().next().unwrap().is_uppercase() {
                imported_types.insert(last.clone());
            } else {
                imported_funcs.insert(last.clone());
            }
        }
    }

    submodules.extend(imported_funcs.clone());
    submodules.extend(imported_types.clone());

    // 1. Recursively flatten all nested sub-modules first
    for m in program.modules {
        if let Some(body) = m.body {
            let mut path = current_path.clone();
            path.push(m.name.clone());
            flatten_module(body, path, root_program, top_level_modules);
        }
    }

    // 2. Gather local types, functions, and submodules
    let mut local_types: HashSet<String> = program
        .structs
        .iter()
        .map(|s| s.name.clone())
        .chain(program.enums.iter().map(|e| e.name.clone()))
        .chain(program.traits.iter().map(|t| t.name.clone()))
        .collect();
    local_types.extend(imported_types);

    let mut local_funcs: HashSet<String> = program.funcs.iter().map(|f| f.name.clone()).collect();
    local_funcs.extend(imported_funcs);

    let prefix = current_path.join("::");

    // Qualify local items if not at the root level
    let mut funcs = program.funcs;
    let mut structs = program.structs;
    let mut enums = program.enums;
    let mut traits = program.traits;
    let mut impls = program.impls;
    let mut type_aliases = program.type_aliases;

    for func in &mut funcs {
        for param in &mut func.params {
            qualify_type(&mut param.ty, &local_types, &prefix);
        }
        if let Some(ref mut ret) = func.return_type {
            qualify_type(ret, &local_types, &prefix);
        }
        qualify_block(
            &mut func.body,
            &local_funcs,
            &local_types,
            &prefix,
            &current_path,
            &submodules,
            top_level_modules,
        );
        if !prefix.is_empty() && func.name != "main" {
            func.name = format!("{}::{}", prefix, func.name);
        }
    }

    for s in &mut structs {
        for field in &mut s.fields {
            qualify_type(&mut field.ty, &local_types, &prefix);
        }
        if !prefix.is_empty() {
            s.name = format!("{}::{}", prefix, s.name);
        }
    }

    for e in &mut enums {
        for variant in &mut e.variants {
            if let Some(ref mut ty) = variant.ty {
                qualify_type(ty, &local_types, &prefix);
            }
        }
        if !prefix.is_empty() {
            e.name = format!("{}::{}", prefix, e.name);
        }
    }

    for t in &mut traits {
        for method in &mut t.methods {
            for param in &mut method.params {
                qualify_type(&mut param.ty, &local_types, &prefix);
            }
            if let Some(ref mut ret) = method.return_type {
                qualify_type(ret, &local_types, &prefix);
            }
        }
        if !prefix.is_empty() {
            t.name = format!("{}::{}", prefix, t.name);
        }
    }

    for i in &mut impls {
        qualify_type(&mut i.impl_type, &local_types, &prefix);
        for method in &mut i.methods {
            for param in &mut method.params {
                qualify_type(&mut param.ty, &local_types, &prefix);
            }
            if let Some(ref mut ret) = method.return_type {
                qualify_type(ret, &local_types, &prefix);
            }
            qualify_block(
                &mut method.body,
                &local_funcs,
                &local_types,
                &prefix,
                &current_path,
                &submodules,
                top_level_modules,
            );
        }
    }

    for alias in &mut type_aliases {
        qualify_type(&mut alias.aliased_type, &local_types, &prefix);
        if !prefix.is_empty() {
            alias.name = format!("{}::{}", prefix, alias.name);
        }
    }

    // Merge into the root program
    root_program.funcs.extend(funcs);
    root_program.structs.extend(structs);
    root_program.enums.extend(enums);
    root_program.traits.extend(traits);
    root_program.impls.extend(impls);
    root_program.type_aliases.extend(type_aliases);

    for mut use_decl in program.uses {
        use_decl.module_path = current_path.clone();
        root_program.uses.push(use_decl);
    }
}

fn resolve_and_flatten_modules(mut program: Program, source_path: &str) -> Program {
    // 1. Recursive load all mod files
    resolve_modules(&mut program, source_path);

    // 2. Build top-level module names set
    let top_level_modules: HashSet<String> =
        program.modules.iter().map(|m| m.name.clone()).collect();

    // 3. Flatten recursively
    let mut root_program = Program {
        uses: Vec::new(),
        modules: Vec::new(), // modules are flattened
        funcs: Vec::new(),
        structs: Vec::new(),
        enums: Vec::new(),
        traits: Vec::new(),
        impls: Vec::new(),
        type_aliases: Vec::new(),
    };

    flatten_module(program, vec![], &mut root_program, &top_level_modules);

    root_program
}

#[cfg(test)]
mod project_tests {
    use super::*;

    #[test]
    fn test_parse_ulang_toml() {
        let toml_content = r#"
            [package]
            name = "my_project"

            [profile.dev]
            opt-level = "none"

            [profile.release]
            opt-level = 3

            [build]
            cc = "clang"
        "#;
        let config: UlangToml = toml::from_str(toml_content).unwrap();
        assert_eq!(config.package.name, "my_project");
        assert!(matches!(config.profile.dev.opt_level, OptLevel::None));
        assert!(matches!(
            config.profile.release.opt_level,
            OptLevel::Aggressive
        ));
        assert_eq!(config.build.cc, Some(Cc::Clang));
    }

    #[test]
    fn test_parse_ulang_toml_defaults() {
        let toml_content = r#"
            [package]
            name = "test_defaults"
        "#;
        let config: UlangToml = toml::from_str(toml_content).unwrap();
        assert_eq!(config.package.name, "test_defaults");
        assert_eq!(config.build.cc, None);
        assert!(matches!(config.profile.dev.opt_level, OptLevel::None));
        assert!(matches!(
            config.profile.release.opt_level,
            OptLevel::Aggressive
        ));
    }
}
