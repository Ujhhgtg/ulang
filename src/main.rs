mod ast;
mod codegen;
mod error;
mod lexer;
mod parser;
mod token;

use annotate_snippets::renderer::{AnsiColor, Effects};
use clap::{Parser, Subcommand, builder::Styles};
use std::fs;
use std::path::Path;
use std::process;

use crate::token::{Span, Token};

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
}

#[derive(Subcommand)]
enum Command {
    /// Compile and run the source file (JIT)
    Run {
        /// Path to .u source file
        file: String,
    },
    /// Compile to a native executable
    Build {
        /// Path to .u source file
        file: String,
        /// Output executable path [default: a.out]
        #[arg(short = 'o')]
        output: Option<String>,
    },
}

enum Mode {
    Run,
    Build { output: Option<String> },
}

fn main() {
    let cli = Cli::parse();

    let (mode, path) = match cli.command {
        Command::Run { file } => (Mode::Run, file),
        Command::Build { file, output } => (Mode::Build { output }, file),
    };

    let source = fs::read_to_string(&path).unwrap_or_else(|e| {
        eprintln!("error reading '{}': {}", path, e);
        process::exit(1);
    });

    // --- Lex ---
    let mut lexer = lexer::Lexer::new(&source);
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
                error::emit_error(&source, &path, Span::new(pos, pos), "lex error", &msg);
                process::exit(1);
            }
        }
    }

    // --- Parse ---
    let mut parser = parser::Parser::new(&tokens);
    let program = match parser.parse_program() {
        Ok(prog) => prog,
        Err(e) => {
            error::emit_error(&source, &path, e.span, "parse error", &e.msg);
            process::exit(1);
        }
    };

    let context = inkwell::context::Context::create();

    match mode {
        Mode::Run => {
            let mut codegen = match codegen::CodeGen::new_jit(&context) {
                Ok(cg) => cg,
                Err(msg) => {
                    eprintln!("codegen error: {}", msg);
                    process::exit(1);
                }
            };
            if let Err(msg) = codegen.jit_run(&program) {
                eprintln!("runtime error: {}", msg);
                process::exit(1);
            }
        }
        Mode::Build { output } => {
            let mut codegen = codegen::CodeGen::new_native(&context);

            if let Err(msg) = codegen.compile_module(&program) {
                eprintln!("codegen error: {}", msg);
                process::exit(1);
            }

            let exe_path = output.unwrap_or_else(|| "a.out".to_string());
            let obj_path = format!("{}.o", exe_path);

            if let Err(msg) = codegen.compile_to_object(Path::new(&obj_path)) {
                eprintln!("codegen error: {}", msg);
                process::exit(1);
            }

            if let Err(msg) =
                codegen::CodeGen::link_executable(Path::new(&obj_path), Path::new(&exe_path))
            {
                eprintln!("link error: {}", msg);
                let _ = fs::remove_file(&obj_path);
                process::exit(1);
            }

            let _ = fs::remove_file(&obj_path);
        }
    }
}
