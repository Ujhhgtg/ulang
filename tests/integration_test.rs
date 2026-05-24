use std::path::{Path, PathBuf};
use std::process::Command;

fn ulang_binary() -> String {
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("debug")
        .join("ulang");
    path.to_string_lossy().to_string()
}

fn test_dir(name: &str) -> PathBuf {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("tmp_integration")
        .join(name);
    let _ = std::fs::create_dir_all(&dir);
    dir
}

fn write_test(src: &str, name: &str) -> PathBuf {
    let dir = test_dir(name);
    let path = dir.join(format!("{}.u", name));
    std::fs::write(&path, src).expect("write test file");
    path
}

fn run_test(name: &str, src: &str) -> bool {
    let path = write_test(src, name);
    let output = Command::new(ulang_binary())
        .args(["run", &path.to_string_lossy()])
        .output()
        .expect("failed to execute ulang");
    output.status.success()
}

fn run_test_expect_error(name: &str, src: &str) -> bool {
    let path = write_test(src, name);
    let output = Command::new(ulang_binary())
        .args(["run", &path.to_string_lossy()])
        .output()
        .expect("failed to execute ulang");
    !output.status.success()
}

#[test]
fn test_run_calc() {
    let calc_u = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("examples")
        .join("calc.u");

    let output = Command::new(ulang_binary())
        .args(["run", &calc_u.to_string_lossy()])
        .output()
        .expect("failed to execute ulang");

    assert!(
        output.status.success(),
        "ulang run exited with code {:?}",
        output.status.code()
    );
}

#[test]
fn test_run_empty_source() {
    assert!(run_test("empty", "fn main() {}\n"));
}

#[test]
fn test_mutable_var_reassign() {
    assert!(run_test(
        "mut_reassign",
        "fn main() { let mut x = 10; x = x + 20; }\n"
    ));
}

#[test]
fn test_const_declaration() {
    assert!(run_test(
        "const_test",
        "fn main() { const X = 42; let y = X; }\n"
    ));
}

#[test]
fn test_assign_to_immutable_errors() {
    assert!(run_test_expect_error(
        "immut_assign",
        "fn main() { let x = 10; x = 20; }\n"
    ));
}

#[test]
fn test_typed_let_declaration() {
    assert!(run_test("typed_let", "fn main() { let x: i32 = 42; }"));
}

#[test]
fn test_float_literal() {
    assert!(run_test("float", "fn main() { let x: f64 = 3.14; }"));
}

#[test]
fn test_as_cast_chain() {
    assert!(run_test(
        "cast_chain",
        "fn main() { let x: u8 = 1000 as i64 as u8; }"
    ));
}

#[test]
fn test_float_to_int_cast() {
    assert!(run_test(
        "float_to_int",
        "fn main() { let x: i32 = 3.99 as i32; }"
    ));
}

#[test]
fn test_int_to_float_cast() {
    assert!(run_test(
        "int_to_float",
        "fn main() { let x: f64 = 42 as f64; }"
    ));
}

#[test]
fn test_print_cast() {
    assert!(run_test(
        "print_cast",
        "use std::io::print; fn main() { print(42); }"
    ));
}

#[test]
fn test_as_cast_precedence() {
    assert!(run_test(
        "cast_prec",
        "fn main() { let x: i64 = 1 + 2 as i64; }"
    ));
}

#[test]
fn test_suffix_int_i32() {
    assert!(run_test("suffix_i32", "fn main() { let x = 42i32; }"));
}

#[test]
fn test_suffix_u8() {
    assert!(run_test("suffix_u8", "fn main() { let x = 255u8; }"));
}

#[test]
fn test_suffix_i64() {
    assert!(run_test("suffix_i64", "fn main() { let x = 1000i64; }"));
}

#[test]
fn test_suffix_float_f64() {
    assert!(run_test("suffix_f64", "fn main() { let x = 3.14f64; }"));
}

#[test]
fn test_suffix_float_f32() {
    assert!(run_test("suffix_f32", "fn main() { let x = 1.5f32; }"));
}

#[test]
fn test_run_str_len_literal() {
    assert!(run_test("str_len_lit", "fn main() { \"hello\".len(); }\n"));
}

#[test]
fn test_run_str_len_variable() {
    assert!(run_test(
        "str_len_var",
        "fn main() { let s = \"hello\"; s.len(); }\n"
    ));
}

#[test]
fn test_run_parse_error() {
    assert!(run_test_expect_error(
        "bad",
        "fn main() { missing_semicolon }\n"
    ));
}

#[test]
fn test_struct_create() {
    assert!(run_test(
        "struct_create",
        "struct Point { x: i32, y: i32, }\nfn main() { let p = Point { x: 10, y: 20 }; }\n"
    ));
}

#[test]
fn test_struct_method_call() {
    assert!(run_test(
        "struct_method",
        "struct Point { x: i32, y: i32, }\nimpl Point {\n    fn new(x: i32, y: i32) -> Point { Point { x: x, y: y }; }\n    fn area(&self) -> i32 { self.x * self.y; }\n}\nfn main() {\n    let p = Point { x: 3, y: 4 };\n    p.area();\n}\n"
    ));
}

#[test]
fn test_builtin_trait_methods() {
    assert!(run_test(
        "builtin_traits",
        "fn main() {\n    let a: i32 = 42;\n    let b: i32 = 42;\n    a.eq(&b);\n    a.ne(&b);\n    a.cmp(&b);\n    a.clone();\n    a.default();\n}\n"
    ));
}

#[test]
fn test_impl_for_trait() {
    assert!(run_test(
        "impl_trait",
        "struct Point { x: i32, y: i32, }\ntrait Drawable { fn draw(&self); }\nimpl Drawable for Point { fn draw(&self) { } }\nfn main() {\n    let p = Point { x: 1, y: 2 };\n    p.draw();\n}\n"
    ));
}

#[test]
fn test_if_expression() {
    assert!(run_test("if_expr", "fn main() { if 1 { }; }\n"));
}

#[test]
fn test_if_else_expression() {
    assert!(run_test("if_else", "fn main() { if 0 { } else { }; }\n"));
}

#[test]
fn test_while_loop() {
    assert!(run_test("while_loop", "fn main() { while 0 { }; }\n"));
}

#[test]
fn test_return_stmt() {
    assert!(run_test("return_stmt", "fn main() -> i32 { return 42; }\n"));
}

#[test]
fn test_implicit_return() {
    assert!(run_test("implicit_return", "fn main() -> i32 { 42 }\n"));
}

#[test]
fn test_comparison_eq() {
    assert!(run_test("cmp_eq", "fn main() { 1 == 1; }\n"));
}

#[test]
fn test_if_return() {
    assert!(run_test(
        "if_return",
        "fn main() -> i32 { if 1 { return 42; }; 0 }\n"
    ));
}

#[test]
fn test_loop_keyword() {
    assert!(run_test(
        "loop_keyword",
        "fn main() { loop { return; }; }\n"
    ));
}

#[test]
fn test_string_new() {
    assert!(run_test(
        "string_new",
        "use std::string::String;\nfn main() { let s = String::new(); }\n"
    ));
}

#[test]
fn test_string_with_capacity() {
    assert!(run_test(
        "string_with_cap",
        "use std::string::String;\nfn main() { let s = String::with_capacity(10); }\n"
    ));
}

#[test]
fn test_string_push_str() {
    assert!(run_test(
        "string_push",
        "use std::string::String;\nfn main() { let mut s = String::new(); s.push_str(\"hello\"); }\n"
    ));
}

#[test]
fn test_string_len_capacity() {
    assert!(run_test(
        "string_len_cap",
        "use std::string::String;\nfn main() { let s = String::new(); s.len(); s.capacity(); }\n"
    ));
}

#[test]
fn test_string_clear() {
    assert!(run_test(
        "string_clear",
        "use std::string::String;\nfn main() { let mut s = String::with_capacity(10); s.clear(); }\n"
    ));
}

#[test]
fn test_string_is_empty() {
    assert!(run_test(
        "string_empty",
        "use std::string::String;\nfn main() { let s = String::new(); s.is_empty(); }\n"
    ));
}

#[test]
fn test_print_str() {
    assert!(run_test(
        "print_str",
        "use std::io::println;\nfn main() { println(\"hello world\"); }\n"
    ));
}

#[test]
fn test_print_string() {
    assert!(run_test(
        "print_string",
        "use std::io::println;\nuse std::string::String;\nfn main() { let mut s = String::new(); s.push_str(\"hello from String\"); println(s); }\n"
    ));
}

#[test]
fn test_print_string_utf8() {
    assert!(run_test(
        "print_string_utf8",
        "use std::io::println;\nuse std::string::String;\nfn main() { let mut s = String::new(); s.push_str(\"héllo wörld 🌍\"); println(s); }\n"
    ));
}

#[test]
fn test_print_str_utf8() {
    assert!(run_test(
        "print_str_utf8",
        "use std::io::println;\nfn main() { println(\"héllo wörld 🌍\"); }\n"
    ));
}

#[test]
fn test_panic_str() {
    assert!(run_test_expect_error(
        "panic_str",
        "use std::panic::panic;\nfn main() { panic(\"something went wrong\"); }\n"
    ));
}

#[test]
fn test_panic_string() {
    assert!(run_test_expect_error(
        "panic_string",
        "use std::string::String;\nuse std::panic::panic;\nfn main() { let mut s = String::new(); s.push_str(\"error occurred\"); panic(s); }\n"
    ));
}

#[test]
fn test_type_alias() {
    assert!(run_test(
        "type_alias",
        "type Meters = i32; fn main() { let x: Meters = 42; }\n"
    ));
}

#[test]
fn test_array_literal() {
    assert!(run_test(
        "array_lit",
        "use std::io::println;\nfn main() { let a = [1, 2, 3]; println(a[0]); println(a[1]); println(a[2]); }\n"
    ));
}

#[test]
fn test_array_repeat() {
    assert!(run_test(
        "array_rep",
        "use std::io::println;\nfn main() { let a = [0; 4]; println(a[0]); println(a[3]); }\n"
    ));
}

#[test]
fn test_array_mutation() {
    assert!(run_test(
        "array_mut",
        "use std::io::println;\nfn main() { let mut a = [10, 20, 30]; a[1] = 99; println(a[0]); println(a[1]); println(a[2]); }\n"
    ));
}

#[test]
fn test_array_typed() {
    assert!(run_test(
        "array_typed",
        "use std::io::println;\nfn main() { let a: [i32; 2] = [5, 7]; println(a[0]); println(a[1]); }\n"
    ));
}

#[test]
fn test_nested_array() {
    assert!(run_test(
        "array_nested",
        "use std::io::println;\nfn main() { let a = [[1, 2], [3, 4]]; println(a[0][0]); println(a[0][1]); println(a[1][0]); println(a[1][1]); }\n"
    ));
}

#[test]
fn test_single_element_array() {
    assert!(run_test(
        "array_single",
        "use std::io::println;\nfn main() { let a = [42]; println(a[0]); }\n"
    ));
}
