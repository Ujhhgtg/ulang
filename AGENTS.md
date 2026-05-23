# Repository Guidelines

## Project Overview

**ulang** — a tiny compiled language targeting `.u` source files. It compiles via LLVM (through the `inkwell` crate) to either JIT execution or native executables. The language is minimal: functions, `let` bindings, integer arithmetic, and a `print()` builtin. All values are `i32`.

## Architecture & Data Flow

```none
source (.u) → Lexer → [Token] → Parser → AST (Program) → CodeGen → LLVM IR → JIT / native binary
```

Pipeline is sequential, single-pass, no IR middle-end.

- **Lexer** (`lexer.rs`): hand-written, character-at-a-time with single-character lookahead. Skips whitespace and `//` comments. Returns `(Token, Span)` pairs. Spans track byte offsets for error reporting.
- **Parser** (`parser.rs`): recursive descent. Operator precedence: `+ -` (additive) < `* /` (multiplicative), grouped by `parse_expr` → `parse_term` → `parse_primary`. Produces `Program { funcs: Vec<Function> }`.
- **CodeGen** (`codegen.rs`): LLVM IR generation via `inkwell`. Two modes — `new_jit` (with `ExecutionEngine` for in-memory JIT) and `new_native` (for `.o` → `cc` link). `print()` compiles to `printf("%d\n", value)`.
- **Error reporting** (`error.rs`): uses `annotate-snippets` for pretty-printed source-level diagnostics to stderr.

## Key Directories

| Path | Purpose |
|---|---|
| `src/` | All source code (single crate, no workspace) |
| `examples/` | Example `.u` programs |
| `target/` | Build artifacts (gitignored) |

## Development Commands

All commands use Cargo (Rust edition 2024).

```shell
# Build
cargo build

# Run a .u file via JIT
cargo run -- run examples/calc.u

# Compile to native executable
cargo run -- build examples/calc.u -o myprog

# Check (fast)
cargo check

# Format
cargo fmt

# Lint
cargo clippy
```

The default output for `build` is `a.out`; override with `-o <path>`.

## Code Conventions & Common Patterns

- **Error handling**: Functions propagate errors as `Result<(), String>`. The `main` function calls `process::exit(1)` on any error after printing diagnostics. No `thiserror` or `anyhow` — plain `String` errors.
- **AST mutability**: AST types use `#[allow(dead_code)]` on field spans that exist for error reporting but aren't read by downstream passes.
- **Lexer/Parser borrow**: Both `Lexer<'a>` and `Parser<'a>` borrow the source/token slice. `Parser` stores `&'a [(Token, Span)]` — no arena, no cloning.
- **CodeGen lifetime**: `CodeGen<'ctx>` is tied to the `inkwell::Context`. Two constructors: `new_jit` (creates `ExecutionEngine`) and `new_native` (no engine). Symbols map: `HashMap<String, PointerValue>` for `let`-bound variables.
- **Naming**: snake_case for functions/variables, CamelCase for types/enums. Module declarations at top of `main.rs`.
- **Comments**: only `//` line comments supported in the lexer.
- **No async, no trait objects, no DI** — plain synchronous code throughout.
- **Metadata structs**: `Span { lo: usize, hi: usize }` everywhere for error reporting. `emit_error(source, path, span, title, label)` generates annotated diagnostics.

## Important Files

| File | Role |
|---|---|
| `src/main.rs` | Entry point, CLI parsing (`clap`), pipeline orchestration |
| `src/token.rs` | `Token` enum and `Span` struct |
| `src/lexer.rs` | `Lexer` — tokenizer |
| `src/ast.rs` | `Program`, `Function`, `Block`, `Stmt`, `Expr`, `BinOp` |
| `src/parser.rs` | `Parser` — recursive descent parser |
| `src/codegen.rs` | `CodeGen` — LLVM IR emission, JIT + native compilation |
| `src/error.rs` | `emit_error` — pretty source-level diagnostics |
| `examples/calc.u` | Example program |
| `Cargo.toml` | Single crate; depends on `inkwell 0.9` (LLVM 22), `clap 4`, `annotate-snippets 0.12` |

## Runtime/Tooling Preferences

- **Required**: Rust toolchain (edition 2024), LLVM 22 (via `inkwell`), `cc` linker (system `cc` for native builds).
- **Package manager**: Cargo.
- **Formatter**: `cargo fmt` (default).
- **Linter**: `cargo clippy` (default).

## Testing & QA

```shell
# Run all unit and integration tests
cargo test

# Check lints and format (run before pushing)
cargo clippy --release
cargo fmt
```

### Test layout

Tests live as `#[cfg(test)] mod tests { .. }` blocks inline in each source module, plus integration tests under `tests/`.

| File | Tests |
|---|---|
| `src/token.rs` | Span construction, Token equality (3 tests) |
| `src/lexer.rs` | Numbers, identifiers, keywords, operators, whitespace, comments, error chars, span continuity (12 tests) |
| `src/parser.rs` | Valid/invalid programs, operator precedence, parenthesized expressions, call syntax, error messages (15 tests) |
| `src/codegen.rs` | JIT execution (empty, let, arithmetic, print, multiple stmts, two funcs), error paths (undefined var, unknown func), native compilation (9 tests) |
| `tests/integration_test.rs` | End-to-end `run` of `calc.u`, empty main, parse error exit code (3 tests) |

### Running specific test suites

```shell
# Just lexer tests
cargo test -- lexer

# Just parser tests
cargo test -- parser

# Just codegen tests (requires LLVM JIT)
cargo test -- jit

# Just integration tests
cargo test --test integration_test
```

### Notes

- Codegen tests require LLVM 22 (via `inkwell` JIT) and are excluded from `cargo check`.
- Integration tests invoke the built binary from `target/debug/ulang` — always `cargo build` or `cargo test` first.
- Adding new tests: add `#[cfg(test)] mod tests { .. }` to the relevant `src/*.rs` file, or create a new file under `tests/` for integration.
