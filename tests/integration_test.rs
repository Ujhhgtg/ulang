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
        "fn main() {\n    let a: i32 = 42;\n    let b: i32 = 42;\n    a.eq(&b);\n    a.ne(&b);\n    a.cmp(&b);\n    a.clone();\n    i32::default();\n}\n"
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
    assert!(run_test("if_expr", "fn main() { if true { }; }\n"));
}

#[test]
fn test_if_else_expression() {
    assert!(run_test(
        "if_else",
        "fn main() { if false { } else { }; }\n"
    ));
}

#[test]
fn test_while_loop() {
    assert!(run_test("while_loop", "fn main() { while false { }; }\n"));
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
        "fn main() -> i32 { if true { return 42; }; 0 }\n"
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

#[test]
fn test_option_stdlib() {
    assert!(run_test(
        "option_test",
        r#"use std::option::Option;
        fn main() -> i32 {
            let opt = Option::Some(42);
            if opt.is_some() {
                if opt.unwrap() == 42 {
                    if opt.unwrap_or(100) == 42 {
                        return 0;
                    }
                }
            };
            return 1;
        }"#
    ));
}

#[test]
fn test_option_expect_some() {
    assert!(run_test(
        "option_expect_some",
        r#"use std::option::Option;
        fn main() -> i32 {
            let opt = Option::Some(42);
            if opt.expect("should be some") == 42 {
                return 0;
            };
            return 1;
        }"#
    ));
}

#[test]
fn test_option_expect_none() {
    assert!(run_test_expect_error(
        "option_expect_none",
        r#"use std::option::Option;
        fn main() {
            let opt: Option<i32> = Option::None;
            opt.expect("custom panic message");
        }"#
    ));
}

#[test]
fn test_result_stdlib() {
    assert!(run_test(
        "result_test",
        r#"use std::result::Result;
        fn main() -> i32 {
            let r: Result<i32, i32> = Result::Ok(42);
            let e: Result<i32, i32> = Result::Err(100);
            if r.is_ok() {
                if e.is_err() {
                    if r.unwrap() == 42 {
                        if e.unwrap_err() == 100 {
                            return 0;
                        }
                    }
                }
            };
            return 1;
        }"#
    ));
}

#[test]
fn test_logical_and_or() {
    assert!(run_test(
        "logical_and_or",
        r#"fn main() -> i32 {
            let x = 10;
            let y = 20;
            let a = x < 15 && y > 15;
            let b = x > 15 || y > 15;
            if a && b {
                return 0;
            };
            return 1;
        }"#
    ));
}

#[test]
fn test_command_stdlib() {
    assert!(run_test(
        "command_test",
        r#"use std::process::Command;
        fn main() -> i32 {
            let mut cmd = Command::new("true");
            let status = cmd.status();
            if status == 0 {
                let mut cmd_fail = Command::new("false");
                let status_fail = cmd_fail.status();
                if status_fail == 1 {
                    let mut cmd_args = Command::new("ls");
                    cmd_args.args(["-l", "-a"]);
                    let status_args = cmd_args.status();
                    if status_args == 0 {
                        let mut cmd_spawn = Command::new("true");
                        cmd_spawn.spawn();
                        return 0;
                    }
                }
            };
            return 1;
        }"#
    ));
}

#[test]
fn test_command_args_chaining() {
    assert!(run_test(
        "command_chaining",
        r#"use std::process::Command;
        fn main() -> i32 {
            let mut cmd = Command::new("ls");
            cmd.arg("-l").arg("-a");
            let status = cmd.status();
            if status == 0 {
                return 0;
            };
            return 1;
        }"#
    ));
}

#[test]
fn test_command_invalid_path() {
    assert!(run_test(
        "command_invalid",
        r#"use std::process::Command;
        fn main() -> i32 {
            let mut cmd = Command::new("this_binary_does_not_exist_12345");
            let status = cmd.status();
            if status == 127 {
                return 0;
            };
            return 1;
        }"#
    ));
}

#[test]
fn test_operator_overloading_custom() {
    assert!(run_test(
        "operator_overloading",
        r#"
        struct Point {
            x: i32,
            y: i32,
        }

        trait Add {
            fn add(&self, other: &Self) -> Self;
        }

        impl Add for Point {
            fn add(&self, other: &Point) -> Point {
                Point {
                    x: self.x + other.x,
                    y: self.y + other.y,
                }
            }
        }

        fn main() -> i32 {
            let p1 = Point { x: 10, y: 20 };
            let p2 = Point { x: 1, y: 2 };
            let p3 = p1 + p2;
            if p3.x == 11 && p3.y == 22 {
                return 0;
            };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_operator_overloading_ord() {
    assert!(run_test(
        "operator_overloading_ord",
        r#"
        struct Point {
            x: i32,
            y: i32,
        }

        trait Ord {
            fn cmp(&self, other: &Self) -> i32;
        }

        impl Ord for Point {
            fn cmp(&self, other: &Point) -> i32 {
                self.x.cmp(&other.x)
            }
        }

        fn main() -> i32 {
            let p1 = Point { x: 10, y: 20 };
            let p2 = Point { x: 1, y: 2 };
            let p3 = Point { x: 10, y: 5 };
            if p2 < p1 {
                if p1 > p2 {
                    if p1 >= p3 {
                        if p3 <= p1 {
                            return 0;
                        }
                    }
                }
            };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_empty_struct() {
    assert!(run_test(
        "empty_struct",
        r#"
struct Empty;

impl Empty {
    fn can_have_fn() {}
}

fn main() {
    let e = Empty;
    e.can_have_fn();
}
"#,
    ));
}

#[test]
fn test_same_scope_shadowing() {
    assert!(run_test(
        "same_scope_shadow",
        "fn main() -> i32 { let x = 1; let x = x + 1; x }\n"
    ));
}

#[test]
fn test_if_block_shadowing() {
    assert!(run_test(
        "if_shadow",
        "fn main() -> i32 { let x = 1; if true { let x = 99; }; x }\n"
    ));
}

#[test]
fn test_while_block_no_exec_shadowing() {
    assert!(run_test(
        "while_shadow",
        "fn main() -> i32 { let x = 1; while false { let x = 99; }; x }\n"
    ));
}

#[test]
fn test_else_block_shadowing() {
    assert!(run_test(
        "else_shadow",
        "fn main() -> i32 { let x = 1; if false { } else { let x = 99; }; x }\n"
    ));
}

#[test]
fn test_else_if_block_shadowing() {
    assert!(run_test(
        "elif_shadow",
        "fn main() -> i32 { let x = 1; if false { } else if true { let x = 99; }; x }\n"
    ));
}

#[test]
fn test_shadowing_changes_type() {
    assert!(run_test(
        "type_shadow",
        "fn main() -> i32 { let x = 1; let x = true; x as i32 }\n"
    ));
}

#[test]
fn test_shadowing_with_loop_body() {
    assert!(run_test(
        "loop_shadow",
        "fn main() -> i32 { loop { let x = 2; return x; }; 0 }\n"
    ));
}

#[test]
fn test_nested_shadowing() {
    // Shadowing inside a block that itself shadows
    assert!(run_test(
        "nested_shadow",
        "fn main() -> i32 { let x = 1; if true { let x = 2; if true { let x = 3; }; x }; 1 }\n"
    ));
}

#[test]
fn test_inline_modules() {
    assert!(run_test(
        "inline_mod",
        r#"
        mod logging {
            pub fn get_val() -> i32 { 42 }
        }
        fn main() -> i32 {
            if logging::get_val() == 42 {
                if crate::logging::get_val() == 42 {
                    return 0;
                }
            };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_file_modules() {
    // Write the helper file directly in the temporary directory
    let dir = test_dir("file_mod");
    let mod_path = dir.join("helper.u");
    std::fs::write(&mod_path, "pub fn get_val() -> i32 { 100 }\n").expect("write helper");

    assert!(run_test(
        "file_mod",
        r#"
        mod helper;
        fn main() -> i32 {
            if helper::get_val() == 100 {
                return 0;
            };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_directory_module_mod_u() {
    let dir = test_dir("dir_mod");
    let subdir = dir.join("helper");
    std::fs::create_dir_all(&subdir).expect("create helper dir");
    let mod_path = subdir.join("mod.u");
    std::fs::write(&mod_path, "pub fn get_val() -> i32 { 200 }\n").expect("write helper/mod.u");

    assert!(run_test(
        "dir_mod",
        r#"
        mod helper;
        fn main() -> i32 {
            if helper::get_val() == 200 {
                return 0;
            };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_visibility_failures() {
    // 1. Private function call failure
    assert!(run_test_expect_error(
        "priv_call_fail",
        r#"
        mod mymod {
            fn f() {}
        }
        fn main() {
            mymod::f();
        }
        "#
    ));
}

#[test]
fn test_pub_struct_literal_construction() {
    // Successful construction and field access when fields are pub
    assert!(run_test(
        "pub_struct_success",
        r#"
        mod mymod {
            pub struct Foo {
                pub x: i32,
            }
        }
        fn main() -> i32 {
            let f = mymod::Foo { x: 42 };
            if f.x == 42 {
                return 0;
            };
            return 1;
        }
        "#
    ));

    // Failed construction when field is private
    assert!(run_test_expect_error(
        "priv_struct_const_fail",
        r#"
        mod mymod {
            pub struct Foo {
                x: i32,
            }
        }
        fn main() {
            let f = mymod::Foo { x: 42 };
        }
        "#
    ));

    // Failed member access when field is private
    assert!(run_test_expect_error(
        "priv_struct_access_fail",
        r#"
        mod mymod {
            pub struct Foo {
                x: i32,
            }
            impl Foo {
                pub fn new() -> Foo {
                    Foo { x: 42 }
                }
            }
        }
        fn main() {
            let f = mymod::Foo::new();
            let val = f.x;
        }
        "#
    ));
}

#[test]
fn test_single_extern_functions() {
    assert!(run_test(
        "single_extern",
        r#"
        extern "C" fn fork() -> i32;
        fn main() {
            let x = 1;
        }
        "#
    ));
}

#[test]
fn test_pub_extern_functions() {
    assert!(run_test(
        "pub_extern",
        r#"
        mod mymod {
            pub extern "C" fn fork() -> i32;
        }
        use mymod::fork;
        fn main() {
            let x = 1;
        }
        "#
    ));
}
