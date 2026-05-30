# Repository Guidelines

## Project Overview

**ulang** — a tiny compiled language targeting `.u` source files. It compiles via LLVM (through the `inkwell` crate) to either JIT execution or native executables. The language is a Rust subset: functions, structs, enums, traits, generics, pattern matching, modules, and a standard library covering IO, strings, vectors, process spawning, Option/Result, iterators, and operator overloading.

**Target Platform**: ulang officially supports targeting Linux only. Support for Windows and other operating systems is only guaranteed when using `cosmocc` as the linker. The standard library is written specifically for Linux, making direct use of Linux syscalls, POSIX calls, and `libc`.

**vscode-ext**: The VS Code extension located in `./vscode-ext/` provides syntax highlighting and LSP integration for `ulang`. The LSP extension connects to the compiler's own built-in Language Server (`ulang lsp`). Deno must be installed system-wide and is preferred over npm for development tasks inside this folder.

**Design goal**: ulang's grammar & syntax should mostly be a subset of Rust, with minimal modifications. When adding new features, prefer Rust-compatible syntax over novel inventions.

## Architecture & Data Flow

```none
source (.u) → Lexer → [Token] → Parser → AST (Program) → CodeGen → LLVM IR → JIT / native binary
```

Pipeline is sequential, single-pass, no IR middle-end.

- **Lexer** (`lexer.rs`): hand-written, character-at-a-time with single-character lookahead. Skips whitespace and `//` comments. Returns `(Token, Span)` pairs. Spans track byte offsets for error reporting.
- **Parser** (`parser.rs`): recursive descent with operator precedence. Produces `Program` containing functions, structs, enums, traits, impls, type aliases, use declarations, and inline/file modules.
- **CodeGen** (`codegen.rs`): LLVM IR generation via `inkwell`. Two modes — `new_jit` (with `ExecutionEngine` for in-memory JIT) and `new_native` (for `.o` → `cc` link). Supports generics, trait dispatch, method calls, operator overloading, and standard library integration.
- **Error reporting** (`error.rs`): uses `annotate-snippets` for pretty-printed source-level diagnostics to stderr.
- **LSP** (`lsp.rs`): built-in Language Server Protocol implementation providing hover, go-to-definition, and diagnostics for editor integration.
- **Module resolution** (`main.rs`): `use` directives resolve against the standard library (`root/stdlib/`). Inline `mod` blocks and file-based modules are supported with `pub` visibility control.

## Supported Types

Scalar types: `i8`, `i16`, `i32`, `i64`, `u8`, `u16`, `u32`, `u64`, `usize`, `isize`, `f32`, `f64`, `bool`, `str` (string slice), `!` (never type).

Compound types: tuples `(i32, bool)`, arrays `[i32; 3]`, generic arrays `[T; N]`, slices `[T]`, references `&T` / `&mut T`, pointers `*const T` / `*mut T`.

User-defined types: structs (`struct Point { x: i32, y: i32 }`), enums (`enum Option<T> { Some(T), None }`), type aliases (`type MyInt = i32`), `Self` type in impl blocks.

## Expressions

| Category | Forms |
|---|---|
| Literals | `42`, `-1`, `3.14`, `"hello"`, `true`, `false` |
| Suffix literals | `42i32`, `255u8`, `1000i64`, `3.14f64`, `1.5f32` |
| Identifiers | `x`, `my_var` |
| Arithmetic | `+`, `-`, `*`, `/` with precedence |
| Comparison | `==`, `!=`, `<`, `>`, `<=`, `>=` |
| Logical | `&&`, `\|\|`, `!` |
| Assignment | `x = value`, `x[0] = value`, `*ptr = value` |
| References | `&x`, `&mut x` |
| Dereference | `*ptr` |
| Cast | `expr as Type` (e.g. `x as i64`, `3.14 as i32`) |
| Blocks | `{ let x = 1; x + 1 }` (last expression is return value) |
| Control flow | `if cond { … } else { … }`, `if let pat = expr { … }` |
| Loops | `loop { … }`, `while cond { … }`, `for pat in container { … }` |
| Match | `match expr { pat1 => expr1, pat2 => expr2, … }` |
| Calls | `foo(a, b)`, `module::func(a)` |
| Method calls | `obj.method(args)` — dispatches through traits or inherent impls |
| Field access | `obj.field`, `tuple.0` |
| Struct literals | `Point { x: 1, y: 2 }` (field shorthand when var name matches: `Point { x, y }`) |
| Enum literals | `Option::Some(42)`, `Result::Ok(0)` |
| Array/index | `[1, 2, 3]`, `[0; 5]` (repeat), `arr[0]`, `matrix[0][1]` |
| Unary | `-x`, `!x` |

## Statements

- **Let bindings**: `let x = expr;`, `let mut x = expr;`, `let x: Type = expr;`, `let (a, b) = pair;` (destructuring), `let _ = expr;` (wildcard)
- **Const**: `const NAME: Type = expr;` (compile-time evaluated)
- **Return**: `return expr;`, `return;`
- **Continue/Break**: `continue;`, `break;`, `break value;` (loop control)
- **Expression statements**: any expression followed by `;`
- **Tail expression**: last expression in a block without `;` (implicit return)

## Declarations

```rust
// Use declarations
use std::io::println;
use std::cmp::*;

// Functions
fn add(x: i32, y: i32) -> i32 { x + y }
fn greet(name: &str) { println(name); }
fn main() -> i32 { 0 }

// Generic functions
fn identity<T>(x: T) -> T { x }

// Extern functions
extern fn puts(s: &str) -> i32;

// Public items
pub fn helper() { }
pub struct Visible { pub field: i32, internal: bool }

// Structs
struct Point { x: i32, y: i32 }
#[derive(Default, Clone, Eq, Ord)]
struct Flags { active: bool, visible: bool }

// Enums
enum Option<T> { Some(T), None }
enum Result<T, E> { Ok(T), Err(E) }

// Type aliases
type MyArray = [i32; 10];

// Traits
trait Drawable {
    fn draw(&self);
}

// Trait implementations
impl Drawable for Point {
    fn draw(&self) { … }
}

// Inherent implementations
impl Point {
    fn new(x: i32, y: i32) -> Point { Point { x, y } }
    fn area(&self) -> i32 { self.x * self.y }
}

// Inline modules
mod foo {
    fn bar() { }
}
```

## Patterns

Used in `let`, `fn` parameters, `match` arms, `if let`, `for`:

- Variable binding: `x`, `mut x`
- Wildcard: `_`
- Tuple destructuring: `(a, b, _)`
- Struct destructuring: `Point { x, y }`
- Enum destructuring: `Option::Some(val)`, `Result::Ok(v)`
- Literal matching: `42`, `true` (in match arms)
- Or patterns: `1 | 2` (in match arms)

## Standard Library

Located in `root/stdlib/`:

| Module | Contents |
|---|---|
| `core/cmp` | `Eq`, `Ord` traits and implementations |
| `core/clone` | `Clone` trait |
| `core/default` | `Default` trait |
| `core/iter` | `Iterator`, `IntoIterator` traits |
| `core/mem` | `drop`, `size_of`, `swap` |
| `core/ops` | `Add`, `Sub`, `Mul`, `Div`, `Neg`, `Index`, `IndexMut` etc. |
| `core/option` | `Option<T>` enum with `expect`, `unwrap`, `is_some`, etc. |
| `core/result` | `Result<T, E>` enum with `expect`, `unwrap`, `is_ok`, etc. |
| `core/slice` | Slice operations |
| `std/io` | `print`, `println` |
| `std/panic` | `panic` |
| `std/process` | `Command`, `exit` |
| `std/string` | `String` type with `new`, `push_str`, `len`, `clear`, etc. |
| `std/vec` | `Vec<T>` type with `new`, `push`, `pop`, `len`, indexing, iteration |
| `std/alloc` | Allocation support |
| `std/libc` | Raw libc bindings for extern functions |

## Syntax Differences from Rust

### Semicolons after Control Flow Statements

In Rust, block-like expressions (such as `if`, `while`, `loop`, `for`, `match`, and `{}`) used as statements do not require a trailing semicolon.
However, in `ulang`, to clarify statements and expressions, **every statement containing a block-like control flow expression MUST be terminated with a semicolon** if it is followed by subsequent statements. Lacking a semicolon causes the control flow block to be parsed as the tail expression (implicit return of the block), resulting in a syntax error if subsequent statements are found.

For example:

```rust
// In ulang:

// Correct statement usage:
if 1 == 2 {
    println("whoops, mathematics broke");
} else {
    println("everything's fine!");
}; // <- Trailing semicolon is required!

// Correct expression usage:
let value = if true {
    1
} else {
    2
}; // <- Trailing semicolon is required!

// Correct tail expression (implicit return) usage:
fn main() -> i32 {
    if true { 0 } else { 1 }
}
```

## Key Directories

|Path|Purpose|
|---|---|
|`src/`|All source code (single crate, no workspace)|
|`examples/`|Example `.u` programs|
|`root/stdlib/core/`|Core library: clean modules containing no `extern "C"` declarations or allocation dependencies|
|`root/stdlib/std/`|Standard library: modules requiring `extern "C"` or allocation, and re-exporting `core`|
|`target/`|Build artifacts (gitignored)|
|`vscode-ext/`|VS Code extension source, containing TypeScript extension logic, configurations, and TextMate syntax highlighting|

## Development Commands

All commands use Cargo (Rust edition 2024).

```shell
# Build
cargo build

# Run a .u file via JIT
cargo run -- run examples/calc.u

# Compile to native executable
cargo run -- build examples/calc.u -o myprog

# Emit LLVM IR
cargo run -- emit-ir examples/calc.u

# Check (fast)
cargo check

# Format
cargo fmt

# Lint
cargo clippy

# Run LSP server (stdio)
cargo run -- lsp
```

The default output for `build` is `a.out`; override with `-o <path>`.

### VS Code Extension Development

Use Deno (which must be installed system-wide) rather than npm for task execution.

```shell
# Compile VS Code extension using Deno
cd vscode-ext
deno run compile
```

## Code Conventions & Common Patterns

- **Error handling**: Functions propagate errors as `Result<(), String>`. The `main` function calls `process::exit(1)` on any error after printing diagnostics. No `thiserror` or `anyhow` — plain `String` errors. Codegen uses a custom `CodegenError` type with optional source spans.
- **AST mutability**: AST types use `#[allow(dead_code)]` on field spans that exist for error reporting but aren't read by downstream passes.
- **Lexer/Parser borrow**: Both `Lexer<'a>` and `Parser<'a>` borrow the source/token slice. `Parser` stores `&'a [(Token, Span)]` — no arena, no cloning. The parser also tracks `struct_defs`, `type_aliases`, `struct_names`, `enum_names` for name resolution during parsing.
- **CodeGen lifetime**: `CodeGen<'ctx>` is tied to the `inkwell::Context`. Two constructors: `new_jit` (creates `ExecutionEngine`) and `new_native` (no engine). Symbols map: `HashMap<String, PointerValue>` for variables. Supports generics, traits, operator overloading, `continue`/`break` via `LoopContext`, and method dispatch through trait vtables and inherent impls.
- **Naming**: snake_case for functions/variables, CamelCase for types/enums. Module declarations at top of `main.rs`.
- **Comments**: only `//` line comments supported in the lexer.
- **Attributes**: `#[derive(…)]` syntax on structs for auto-implementing `Default`, `Clone`, `Eq`, `Ord`.
- **No async, no trait objects, no DI** — plain synchronous code throughout.
- **Metadata structs**: `Span { lo: usize, hi: usize }` everywhere for error reporting. `emit_error(source, path, span, title, label)` generates annotated diagnostics.
- **Overload map**: The compiler maintains `OverloadMap = HashMap<String, Vec<(String, Vec<Type>)>>` mapping function names to their mangled forms and argument types, enabling function overloading in the standard library via name mangling (`fn$N` suffixed forms).

## Important Files

|File|Role|
|---|---|
|`src/main.rs`|Entry point, CLI parsing (`clap`), pipeline orchestration, module resolution, name qualification|
|`src/token.rs`|`Token` enum and `Span` struct|
|`src/lexer.rs`|`Lexer` — tokenizer|
|`src/ast.rs`|All AST node types: `Program`, `Function`, `Block`, `Stmt`, `Expr`, `BinOp`, `StructDecl`, `EnumDecl`, `TraitDecl`, `ImplDecl`, `TypeAliasDecl`, `ModuleDecl`, `Use`, `Pattern`, `MatchArm`, `Type`, `GenericParam`, `TraitBound`, `Attribute`|
|`src/parser.rs`|`Parser` — recursive descent parser with name tracking and struct literal suppression|
|`src/codegen.rs`|`CodeGen` — LLVM IR emission, JIT + native compilation, generics, trait dispatch, operator overloading|
|`src/error.rs`|`emit_error` — pretty source-level diagnostics|
|`src/lsp.rs`|Built-in Language Server — hover, go-to-definition, diagnostics, stdlib-aware|
|`examples/`|Example `.u` programs covering all language features|
|`Cargo.toml`|Single crate; depends on `inkwell 0.9` (LLVM 22), `clap 4`, `annotate-snippets 0.12`, `lsp-server 0.7`, `lsp-types 0.97`, `url 2`, `serde 1`, `serde_json 1`, `toml 1`|
|`vscode-ext/syntaxes/ulang.tmLanguage.json`|TextMate grammar rules for syntax highlighting|
|`vscode-ext/src/extension.ts`|Entry point of the VS Code extension, launches the ulang LSP client|
|`root/stdlib/`|Core and standard library source files (`.u` modules)|

## Runtime/Tooling Preferences

- **Required**: Linux operating system (targeting Linux is required; Windows and other OS support is only guaranteed when using `cosmocc` as the linker), Rust toolchain (edition 2024), LLVM 22 (via `inkwell`), linker. Also requires Deno installed system-wide for VS Code extension tasks.
- **Package manager**: Cargo. For the VS Code extension, prefer Deno over npm.
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

|File|Tests|
|---|---|
|`src/token.rs`|Span construction, Token equality across all keywords and operators (1 test)|
|`src/lexer.rs`|Numbers, identifiers, keywords, operators, whitespace, comments, error chars, span continuity (12 tests)|
|`src/parser.rs`|Valid/invalid programs, operator precedence, control flow, structs, enums, generics, modules, patterns (many tests)|
|`src/codegen.rs`|JIT execution, let bindings, arithmetic, control flow, struct/trait dispatch, arrays, generics, error paths, native compilation (many tests)|
|`src/lsp.rs`|LSP server tests|
|`tests/integration_test.rs`|End-to-end `run`/`build` tests covering expressions, types, control flow, structs, enums, traits, generics, modules, stdlib (stdlib, arrays, shadowing, for loops, extern functions, pattern matching, move semantics, derive, command, Result/Option, operator overloading, integration) — 80+ tests|

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

# Single integration test by name
cargo test --test integration_test test_struct_create

# Just LSP tests
cargo test -- lsp
```

### Clippy

- When `cargo clippy` reports auto-fixable warnings, run `cargo clippy --fix --allow-dirty` to apply them automatically and immediately, and ignore other non-autofixable ones. DO NOT hand-fix clippy suggestions.

### Notes

- Codegen tests require LLVM 22 (via `inkwell` JIT) and are excluded from `cargo check`.
- Integration tests invoke the built binary from `target/debug/ulang` — always `cargo build` or `cargo test` first.
- Adding new tests: add `#[cfg(test)] mod tests { .. }` to the relevant `src/*.rs` file, or create a new file under `tests/` for integration.
