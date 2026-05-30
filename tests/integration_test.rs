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

// ---------------------------------------------------------------------------
// LSP Completion Integration Tests
// ---------------------------------------------------------------------------

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, Stdio};

/// Send a JSON-RPC message to an LSP server via stdin.
fn lsp_send(stdin: &mut dyn Write, msg: &serde_json::Value) {
    let body = serde_json::to_string(msg).unwrap();
    let header = format!("Content-Length: {}\r\n\r\n", body.len());
    stdin.write_all(header.as_bytes()).unwrap();
    stdin.write_all(body.as_bytes()).unwrap();
    stdin.flush().unwrap();
}

/// Read a JSON-RPC message from an LSP server via stdout.
fn lsp_read(stdout: &mut BufReader<Box<dyn Read>>) -> serde_json::Value {
    let mut header = String::new();
    loop {
        let mut line = String::new();
        stdout.read_line(&mut line).unwrap();
        if line == "\r\n" {
            break;
        }
        header.push_str(&line);
    }
    let content_len: usize = header
        .lines()
        .find(|l| l.to_lowercase().starts_with("content-length"))
        .and_then(|l| l.split(':').nth(1))
        .and_then(|s| s.trim().parse().ok())
        .expect("Content-Length header");
    let mut body = vec![0u8; content_len];
    stdout.read_exact(&mut body).unwrap();
    serde_json::from_slice(&body).unwrap()
}

/// Start ulang LSP server, initialize it, return (child, stdin, stdout_reader).
fn lsp_start() -> (Child, Box<dyn Write>, BufReader<Box<dyn Read>>) {
    let mut child = Command::new(ulang_binary())
        .arg("lsp")
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::inherit())
        .spawn()
        .expect("Failed to spawn ulang lsp");

    let mut stdin: Box<dyn Write> = Box::new(child.stdin.take().unwrap());
    let stdout: Box<dyn Read> = Box::new(child.stdout.take().unwrap());
    let mut stdout_reader = BufReader::new(stdout);

    // Send initialize request
    lsp_send(
        &mut *stdin,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "processId": null,
                "capabilities": {},
                "rootUri": "file:///tmp/ulang-test-project",
                "workspaceFolders": [{
                    "uri": "file:///tmp/ulang-test-project",
                    "name": "test"
                }]
            }
        }),
    );

    // Read the initialize response
    let _init_resp = lsp_read(&mut stdout_reader);

    // Send initialized notification
    lsp_send(
        &mut *stdin,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "method": "initialized",
            "params": {}
        }),
    );

    (child, stdin, stdout_reader)
}

/// Open a document in the LSP server.
fn lsp_open(stdin: &mut dyn Write, url: &str, text: &str) {
    lsp_send(
        stdin,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "method": "textDocument/didOpen",
            "params": {
                "textDocument": {
                    "uri": url,
                    "languageId": "ulang",
                    "version": 1,
                    "text": text
                }
            }
        }),
    );
}

/// Request completion at a given position and return the response.
fn lsp_completion(
    stdin: &mut dyn Write,
    stdout: &mut BufReader<Box<dyn Read>>,
    url: &str,
    line: u32,
    character: u32,
) -> serde_json::Value {
    lsp_send(
        stdin,
        &serde_json::json!({
            "jsonrpc": "2.0",
            "id": 100,
            "method": "textDocument/completion",
            "params": {
                "textDocument": {
                    "uri": url
                },
                "position": {
                    "line": line,
                    "character": character
                }
            }
        }),
    );
    // Read responses, skipping notifications until we get our result
    loop {
        let msg = lsp_read(stdout);
        if msg.get("id").and_then(|id| id.as_i64()) == Some(100) {
            return msg;
        }
    }
}

#[test]
fn test_lsp_completion_stdlib_symbols() {
    let (mut child, mut stdin, mut stdout) = lsp_start();

    let url = "file:///tmp/ulang-test-project/main.u";
    lsp_open(&mut stdin, url, "fn main() {}");

    // Request completion on empty document (line 0, after fn main() {})
    let resp = lsp_completion(&mut stdin, &mut stdout, url, 0, 0);

    // The response should contain an array of completion items
    let items = resp["result"].as_array().expect("Should have result array");
    assert!(!items.is_empty(), "Should have completion items");

    // Check that Option and Result are present
    let labels: Vec<&str> = items.iter().filter_map(|i| i["label"].as_str()).collect();
    assert!(
        labels.contains(&"Option"),
        "Option should be in completions"
    );
    assert!(
        labels.contains(&"Result"),
        "Result should be in completions"
    );

    // Unimported symbols should have additionalTextEdits
    let option_item = items
        .iter()
        .find(|i| i["label"].as_str() == Some("Option"))
        .unwrap();
    assert!(
        option_item.get("additionalTextEdits").is_some(),
        "Option should have additionalTextEdits for use insertion"
    );

    child.kill().ok();
}

#[test]
fn test_lsp_completion_imported_no_use_insertion() {
    let (mut child, mut stdin, mut stdout) = lsp_start();

    let url = "file:///tmp/ulang-test-project/main.u";
    lsp_open(
        &mut stdin,
        url,
        "use std::option::Option;\nfn main() { let x = Option::None; }",
    );

    // Request completion after "Option" is already imported
    let resp = lsp_completion(&mut stdin, &mut stdout, url, 1, 0);

    let items = resp["result"].as_array().expect("Should have result array");
    let option_item = items.iter().find(|i| i["label"].as_str() == Some("Option"));
    assert!(option_item.is_some(), "Option should still appear");
    assert!(
        option_item.unwrap().get("additionalTextEdits").is_none(),
        "Option should NOT have additionalTextEdits (already imported)"
    );

    child.kill().ok();
}

#[test]
fn test_lsp_completion_path_prefix_vec() {
    let (mut child, mut stdin, mut stdout) = lsp_start();

    let url = "file:///tmp/ulang-test-project/main.u";
    // Cursor is right after std::vec::
    lsp_open(&mut stdin, url, "std::vec::");

    let resp = lsp_completion(&mut stdin, &mut stdout, url, 0, "std::vec::".len() as u32);

    let items = resp["result"].as_array().expect("Should have result array");
    assert!(!items.is_empty(), "Should have vec module completions");

    let labels: Vec<&str> = items.iter().filter_map(|i| i["label"].as_str()).collect();
    assert!(
        labels.contains(&"Vec"),
        "Vec should be in std::vec:: completions"
    );

    child.kill().ok();
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
fn test_trait_default_method() {
    assert!(run_test(
        "trait_default",
        "use std::panic::panic;\nstruct Point { x: i32, y: i32, }\ntrait Greeter { fn greet(&self) -> i32 { 42 } }\nimpl Greeter for Point { }\nfn main() {\n    let p = Point { x: 1, y: 2 };\n    let val = p.greet();\n    if val != 42 { panic(\"expected 42\"); }\n}\n"
    ));
}

#[test]
fn test_trait_default_method_override() {
    assert!(run_test(
        "trait_default_override",
        "use std::panic::panic;\nstruct Point { x: i32, y: i32, }\ntrait Greeter { fn greet(&self) -> i32 { 42 } }\nimpl Greeter for Point { fn greet(&self) -> i32 { 99 } }\nfn main() {\n    let p = Point { x: 1, y: 2 };\n    let val = p.greet();\n    if val != 99 { panic(\"expected 99\"); }\n}\n"
    ));
}

#[test]
fn test_trait_default_method_mixed() {
    assert!(run_test(
        "trait_default_mixed",
        "use std::panic::panic;\ntrait Animal { fn noise(&self) -> i32 { 42 } fn legs(&self) -> i32; }\nstruct Dog {}\nimpl Animal for Dog { fn legs(&self) -> i32 { 4 } }\nfn main() {\n    let d = Dog {};\n    let n = d.noise();\n    let l = d.legs();\n    if n != 42 { panic(\"expected 42\"); };\n    if l != 4 { panic(\"expected 4\"); }\n}\n"
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
        "use std::io::println;\nuse std::string::String;\nfn main() { let mut s = String::new(); s.push_str(\"hello from String\"); println(&s); }\n"
    ));
}

#[test]
fn test_print_string_utf8() {
    assert!(run_test(
        "print_string_utf8",
        "use std::io::println;\nuse std::string::String;\nfn main() { let mut s = String::new(); s.push_str(\"héllo wörld 🌍\"); println(&s); }\n"
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
fn test_str_to_string() {
    assert!(run_test(
        "str_to_string",
        "use std::string::String;\nfn main() -> i32 { let s = \"hello, world\".to_string(); let len = s.len(); if len != 12 { return 1; }; 0 }\n"
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
fn test_option_unwrap_unchecked() {
    assert!(run_test(
        "option_unwrap_unchecked",
        r#"use std::option::Option;
        fn main() -> i32 {
            let opt = Option::Some(42);
            if opt.unwrap_unchecked() == 42 {
                return 0;
            };
            return 1;
        }"#
    ));
}

#[test]
fn test_result_unwrap_unchecked() {
    assert!(run_test(
        "result_unwrap_unchecked",
        r#"use std::result::Result;
        fn main() -> i32 {
            let r: Result<i32, i32> = Result::Ok(42);
            let e: Result<i32, i32> = Result::Err(100);
            if r.unwrap_unchecked() == 42 {
                if e.unwrap_err_unchecked() == 100 {
                    return 0;
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

#[test]
fn test_array_iter() {
    assert!(run_test(
        "array_iter",
        r#"
        use std::iter::IntoIterator;
        use std::iter::Iterator;
        use std::option::Option;
        use std::io::print;

        fn main() {
            let a = [10, 20, 30];
            let mut it = a.iter();
            
            match it.next() {
                Option::Some(x) => { print(*x); }
                Option::None => {}
            };
            match it.next() {
                Option::Some(x) => { print(*x); }
                Option::None => {}
            };
            match it.next() {
                Option::Some(x) => { print(*x); }
                Option::None => {}
            };
            match it.next() {
                Option::Some(x) => { print(*x); }
                Option::None => { print(999); }
            };
        }
        "#
    ));
}

#[test]
fn test_array_iter_mut() {
    assert!(run_test(
        "array_iter_mut",
        r#"
        use std::iter::IntoIteratorMut;
        use std::iter::Iterator;
        use std::option::Option;
        use std::io::print;

        fn main() {
            let mut a = [10, 20, 30];
            
            // Increment each element
            let mut it = a.iter_mut();
            match it.next() {
                Option::Some(x) => { *x = *x + 1; }
                Option::None => {}
            };
            match it.next() {
                Option::Some(x) => { *x = *x + 1; }
                Option::None => {}
            };
            match it.next() {
                Option::Some(x) => { *x = *x + 1; }
                Option::None => {}
            };

            // Print them to check they were mutated in-place
            print(a[0]);
            print(a[1]);
            print(a[2]);
        }
        "#
    ));
}

#[test]
fn test_vec_iter() {
    assert!(run_test(
        "vec_iter",
        r#"
        use std::vec::Vec;
        use std::option::Option;
        use std::io::print;

        fn main() {
            let mut v: Vec<i32> = Vec::new();
            v.push(100);
            v.push(200);
            
            let mut it = v.iter();
            match it.next() {
                Option::Some(x) => { print(*x); }
                Option::None => {}
            };
            match it.next() {
                Option::Some(x) => { print(*x); }
                Option::None => {}
            };
            match it.next() {
                Option::Some(x) => {}
                Option::None => { print(999); }
            };
        }
        "#
    ));
}

#[test]
fn test_vec_iter_mut() {
    assert!(run_test(
        "vec_iter_mut",
        r#"
        use std::vec::Vec;
        use std::option::Option;
        use std::io::print;

        fn main() {
            let mut v: Vec<i32> = Vec::new();
            v.push(100);
            v.push(200);
            
            let mut it = v.iter_mut();
            match it.next() {
                Option::Some(x) => { *x = *x + 5; }
                Option::None => {}
            };
            match it.next() {
                Option::Some(x) => { *x = *x + 5; }
                Option::None => {}
            };

            // Pop them to verify they mutated
            match v.pop() {
                Option::Some(val) => { print(val); }
                Option::None => {}
            };
            match v.pop() {
                Option::Some(val) => { print(val); }
                Option::None => {}
            };
        }
        "#
    ));
}

#[test]
fn test_for_loop_array() {
    assert!(run_test(
        "for_loop_array",
        r#"
        use std::iter::IntoIterator;
        use std::iter::Iterator;
        use std::io::print;

        fn main() {
            let a = [10, 20, 30];
            for x in a {
                print(*x);
            }
        }
        "#
    ));
}

#[test]
fn test_for_loop_vec() {
    assert!(run_test(
        "for_loop_vec",
        r#"
        use std::vec::Vec;
        use std::io::print;

        fn main() {
            let mut v: Vec<i32> = Vec::new();
            v.push(100);
            v.push(200);
            for x in v {
                print(*x);
            };
        }
        "#
    ));
}

#[test]
fn test_for_loop_iterator() {
    assert!(run_test(
        "for_loop_iterator",
        r#"
        use std::iter::IntoIterator;
        use std::iter::Iterator;
        use std::io::print;

        fn main() {
            let a = [5, 10];
            let mut it = a.iter();
            for x in it {
                print(*x);
            }
        }
        "#
    ));
}

#[test]
fn test_for_loop_nested() {
    assert!(run_test(
        "for_loop_nested",
        r#"
        use std::iter::IntoIterator;
        use std::iter::Iterator;
        use std::io::print;

        fn main() {
            let a = [1, 2];
            let b = [10, 20];
            for x in a {
                for y in b {
                    print(*x + *y);
                }
            }
        }
        "#
    ));
}

#[test]
fn test_static_method_calls() {
    assert!(run_test(
        "static_method_calls",
        r#"use std::option::Option;
        
        struct Point {
            x: i32,
            y: i32,
        }
        
        impl Point {
            fn area(&self) -> i32 {
                self.x * self.y
            }
        }
        
        fn main() -> i32 {
            let p = Point { x: 3, y: 4 };
            let opt = Option::Some(42);
            
            let area = Point::area(&p);
            let is_some = Option::is_some(&opt);
            
            if area == 12 && is_some {
                return 0;
            };
            return 1;
        }"#
    ));
}

#[test]
fn test_wildcard_let_integration() {
    // 1. Basic wildcard let-binding
    assert!(run_test(
        "wildcard_basic",
        "fn main() { let _ = 42; let _ = true; }"
    ));

    // 2. Typed wildcard let-binding
    assert!(run_test(
        "wildcard_typed",
        "fn main() { let _ : i32 = 42; }"
    ));

    // 3. side effects inside wildcard let-binding
    assert!(run_test(
        "wildcard_side_effects",
        r#"
        fn mutate(x: *mut i32) -> i32 {
            *x = 100;
            0
        }
        fn main() -> i32 {
            let mut val = 0;
            let _ = mutate(&mut val);
            if val == 100 {
                return 0;
            };
            return 1;
        }
        "#
    ));

    // 4. Underscore is not accessible (must produce a compile/parse error)
    assert!(run_test_expect_error(
        "wildcard_inaccessible",
        "fn main() { let _ = 42; let y = _; }"
    ));
}

#[test]
fn test_emit_ir_subcommand() {
    let path = write_test("fn main() { let x = 42; }", "emit_ir_test");
    let output = Command::new(ulang_binary())
        .args(["emit-ir", &path.to_string_lossy()])
        .output()
        .expect("failed to execute ulang emit-ir");
    assert!(output.status.success());
    let ir = String::from_utf8_lossy(&output.stdout);
    assert!(ir.contains("define i32 @main()"));
    assert!(ir.contains("store i32 42, ptr %x"));
}

#[test]
fn test_non_exhaustive_let_pattern() {
    assert!(run_test_expect_error(
        "non_exhaustive_pattern",
        "fn main() { let opt = Option::Some(0); let Some(val) = opt; }"
    ));
}

#[test]
fn test_irrefutable_let_pattern_binding() {
    assert!(run_test("irrefutable_binding", "fn main() { let x = 42; }"));
}

#[test]
fn test_irrefutable_let_pattern_wildcard() {
    assert!(run_test(
        "irrefutable_wildcard",
        "fn main() { let _ = 42; }"
    ));
}

#[test]
fn test_generic_bounds_functions_and_structs() {
    assert!(run_test(
        "generic_bounds",
        r#"
        trait Driver {
            fn drive(&self) -> i32;
        }

        trait Pilot {
            fn fly(&self) -> i32;
        }

        struct Vehicle {
            speed: i32,
        }

        impl Driver for Vehicle {
            fn drive(&self) -> i32 { self.speed }
        }

        impl Pilot for Vehicle {
            fn fly(&self) -> i32 { self.speed * 2 }
        }

        // Test generic function with constraints
        fn operate_vehicle<T: Driver + Pilot>(vehicle: T) -> i32 {
            vehicle.drive() + vehicle.fly()
        }

        // Test impl Trait parameter with constraints
        fn operate_vehicle_impl(vehicle: impl Driver + Pilot) -> i32 {
            vehicle.drive() + vehicle.fly()
        }

        struct Wrapper<T: Driver + Pilot> {
            val: T,
        }

        fn main() -> i32 {
            let v1 = Vehicle { speed: 10 };
            let r1 = operate_vehicle(v1);
            let v2 = Vehicle { speed: 10 };
            let r2 = operate_vehicle_impl(v2);
            let v3 = Vehicle { speed: 10 };
            let w: Wrapper<Vehicle> = Wrapper { val: v3 };
            let r3 = w.val.drive() + w.val.fly();
            if r1 == 30 && r2 == 30 && r3 == 30 {
                return 0;
            };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_impl_into_iterator_args() {
    assert!(run_test(
        "impl_into_iterator",
        r#"
        use std::vec::Vec;
        use std::iter::IntoIterator;

        struct ArgCollector {
            count: i32,
        }

        impl ArgCollector {
            fn arg(&mut self, s: &str) {
                self.count = self.count + 1;
            }

            fn args(&mut self, args: impl IntoIterator<&str>) -> &mut ArgCollector {
                for arg in args {
                    self.arg(arg);
                };
                self
            }
        }

        fn main() -> i32 {
            let mut collector = ArgCollector { count: 0 };
            
            // Test with a Vec
            let mut v: Vec<&str> = Vec::new();
            v.push("hello");
            v.push("world");
            collector.args(v);

            // Test with an array
            collector.args(["a", "b"]);

            if collector.count == 4 {
                return 0;
            };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_move_non_copy_struct() {
    assert!(run_test_expect_error(
        "move_non_copy",
        r#"
        struct NonCopy {
            x: i32,
        }

        fn main() -> i32 {
            let a = NonCopy { x: 42 };
            let b = a;
            let c = a;
            return 0;
        }
        "#
    ));
}

#[test]
fn test_copy_primitive() {
    assert!(run_test(
        "copy_prim",
        r#"
        fn main() -> i32 {
            let a = 42;
            let b = a;
            let c = a;
            return a + b + c;
        }
        "#
    ));
}

#[test]
fn test_bool_copy() {
    assert!(run_test(
        "bool_copy",
        r#"
        fn main() -> i32 {
            let a = true;
            let b = a;
            let c = a;
            if b && c { return 0; };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_move_into_fn_by_value() {
    assert!(run_test_expect_error(
        "move_into_fn",
        r#"
        struct NonCopy { x: i32 }

        fn consume(_val: NonCopy) -> i32 {
            return 0;
        }

        fn main() -> i32 {
            let a = NonCopy { x: 42 };
            let r1 = consume(a);
            let r2 = consume(a);
            return 0;
        }
        "#
    ));
}

#[test]
fn test_copy_to_fn_by_value() {
    assert!(run_test(
        "copy_to_fn",
        r#"
        fn double(x: i32) -> i32 {
            return x * 2;
        }

        fn main() -> i32 {
            let a = 21;
            let r1 = double(a);
            let r2 = double(a);
            return r1 + r2;
        }
        "#
    ));
}

#[test]
fn test_derive_copy_struct() {
    assert!(run_test(
        "derive_copy",
        r#"
        #[derive(Clone, Copy)]
        struct Point {
            x: i32,
            y: i32,
        }

        fn main() -> i32 {
            let p = Point { x: 10, y: 20 };
            let q = p;
            let r = p;
            return p.x + q.y + r.x;
        }
        "#
    ));
}

#[test]
fn test_std_mem_drop() {
    assert!(run_test(
        "std_mem_drop",
        r#"
        use std::mem::drop;

        struct NonCopy { x: i32 }

        fn main() -> i32 {
            let a = NonCopy { x: 42 };
            drop(a);
            return 0;
        }
        "#
    ));
}

#[test]
fn test_use_after_move_expr() {
    assert!(run_test_expect_error(
        "use_after_move_expr",
        r#"
        struct NonCopy { x: i32 }

        fn main() -> i32 {
            let a = NonCopy { x: 42 };
            let b = a;
            let c = a.x;
            return 0;
        }
        "#
    ));
}

#[test]
fn test_no_direct_drop_call() {
    assert!(run_test_expect_error(
        "no_direct_drop",
        r#"
        use std::mem::drop;
        use std::drop::Drop;

        struct MyType { x: i32 }

        impl Drop for MyType {
            fn drop(&mut self) {
                // nothing
            }
        }

        fn main() -> i32 {
            let a = MyType { x: 42 };
            a.drop();
            return 0;
        }
        "#
    ));
}

#[test]
fn test_derive_copy_without_clone_error() {
    assert!(run_test_expect_error(
        "derive_copy_no_clone",
        r#"
        #[derive(Copy)]
        struct Bad { x: i32 }

        fn main() -> i32 {
            return 0;
        }
        "#
    ));
}

#[test]
fn test_continue_break_loop() {
    assert!(run_test(
        "continue_break_loop",
        r#"
        fn main() -> i32 {
            let mut i = 0;
            let mut sum = 0;
            loop {
                i = i + 1;
                if i == 10 { break; };
                if i == 5 { continue; };
                sum = sum + i;
            };
            sum
        }
        "#
    ));
}

#[test]
fn test_never_type_blocks_diverge() {
    assert!(run_test(
        "never_type_blocks_diverge",
        r#"
        fn main() -> i32 {
            let val = if true { return 0; } else { return 1; };
            val
        }
        "#
    ));
}

#[test]
fn test_never_type_coercion() {
    assert!(run_test(
        "never_type_coercion",
        r#"
        use std::process::exit;
        fn main() -> i32 {
            let val = if true { 42 } else { exit(0) };
            val
        }
        "#
    ));
}

#[test]
fn test_match_single_expression() {
    assert!(run_test(
        "match_single_expression",
        r#"
        use std::option::Option;

        fn main() -> i32 {
            let opt = Option::Some(42);
            let result = match opt {
                Option::Some(x) => x,
                Option::None => 0
            };
            if result == 42 {
                return 0;
            };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_match_single_expression_custom_enum() {
    assert!(run_test(
        "match_single_expression_custom_enum",
        r#"
        enum Status {
            Ok(i32),
            Error,
        }
        fn main() -> i32 {
            let s = Status::Ok(100);
            let val = match s {
                Status::Ok(x) => x,
                Status::Error => 0
            };
            if val == 100 {
                return 0;
            };
            return 1;
        }
        "#
    ));
}

#[test]
fn test_use_direct_import_blocks_qualified_call() {
    // use std::io::println should NOT make io::println callable
    assert!(run_test_expect_error(
        "use_direct_import_blocks_qualified",
        "use std::io::println; fn main() { io::println(42); }"
    ));
}

#[test]
fn test_use_overloaded_direct_import_blocks_qualified_call() {
    // use std::io::print should NOT make io::print callable (overloaded case)
    assert!(run_test_expect_error(
        "use_overloaded_direct_import_blocks_qualified",
        "use std::io::print; fn main() { io::print(42); }"
    ));
}

#[test]
fn test_use_namespace_import_allows_qualified_call() {
    // use std::io should make io::println callable
    assert!(run_test(
        "use_namespace_import_allows_qualified",
        "use std::io; fn main() { io::println(42); }"
    ));
}

#[test]
fn test_use_overloaded_namespace_import_allows_qualified_call() {
    // use std::io should make io::print (overloaded) callable
    assert!(run_test(
        "use_overloaded_namespace_import_allows_qualified",
        r#"use std::io;
fn main() {
    io::print(42);
    io::print("hello");
}
        "#
    ));
}

#[test]
fn test_vec_as_slice() {
    assert!(run_test(
        "vec_as_slice",
        r#"
        use std::vec::Vec;

        fn main() -> i32 {
            let mut v: Vec<i32> = Vec::new();
            v.push(10);
            v.push(20);
            v.push(30);

            let s = v.as_slice();
            if s[0] != 10 { return 1; };
            if s[1] != 20 { return 1; };
            if s[2] != 30 { return 1; };

            0
        }
        "#
    ));
}

#[test]
fn test_vec_as_mut_slice() {
    assert!(run_test(
        "vec_as_mut_slice",
        r#"
        use std::vec::Vec;

        fn main() -> i32 {
            let mut v: Vec<i32> = Vec::new();
            v.push(10);
            v.push(20);
            v.push(30);

            // Read through shared slice first
            let s = v.as_slice();
            if s[0] != 10 { return 1; };
            if s[1] != 20 { return 1; };
            if s[2] != 30 { return 1; };

            // Mutate through mutable slice
            let ms = v.as_mut_slice();
            ms[0] = 99;
            ms[1] = 88;

            // Verify mutation via shared slice
            if s[0] != 99 { return 1; };
            if s[1] != 88 { return 1; };
            if s[2] != 30 { return 1; };

            0
        }
        "#
    ));
}

#[test]
fn test_turbofish_generic_functions() {
    assert!(run_test(
        "turbofish_generic_functions",
        r#"
        // Test turbofish with explicit type arguments
        fn identity<T>(x: T) -> T { x }

        fn pair<A, B>(a: A, b: B) -> i32 { 42 }

        struct Container { value: i32, }
        impl Container {
            fn identity<T>(&self, x: T) -> T { x }
        }

        fn main() -> i32 {
            // Basic turbofish on function call
            let x = identity::<i32>(42);
            if x != 42 { return 1; };

            // Inference still works when turbofish is not used
            let y = identity(100);
            if y != 100 { return 1; };

            // Multiple type params with turbofish
            let z = pair::<i32, f64>(1, 2.0);
            if z != 42 { return 1; };

            // Qualified call with turbofish (Container::identity)
            // Uses turbofish on the method call
            let c = Container { value: 10 };
            let r = c.identity::<i32>(7);
            if r != 7 { return 1; };

            0
        }
        "#
    ));
}

#[test]
fn test_turbofish_wildcard_infer() {
    assert!(run_test(
        "turbofish_wildcard",
        r#"
        // Test _ wildcard in turbofish: lets inference fill in some params
        // while others are explicitly provided
        fn convert<T, D>(x: T) -> D {
            let result: D = 42 as D;
            result
        }

        fn main() -> i32 {
            // D is not inferrable from args (only T is), use _ for T
            let r = convert::<_, i32>(100);
            if r != 42 { return 1; };

            // Full explicit also works
            let r2 = convert::<i32, i32>(200);
            if r2 != 42 { return 1; };

            0
        }
        "#
    ));
}

#[test]
fn test_inline_attributes() {
    let src = r#"
    #[inline]
    fn f_default() -> i32 { 1 }

    #[inline(always)]
    fn f_always() -> i32 { 2 }

    #[inline(never)]
    fn f_never() -> i32 { 3 }

    struct MyStruct;
    impl MyStruct {
        #[inline(always)]
        fn method_always(&self) -> i32 { 4 }

        #[inline(never)]
        fn method_never(&self) -> i32 { 5 }
    }

    trait Foo {
        fn trait_method(&self) -> i32;
    }
    impl Foo for MyStruct {
        #[inline(always)]
        fn trait_method(&self) -> i32 { 6 }
    }

    fn main() -> i32 {
        let s = MyStruct;
        if f_default() != 1 { return 1; };
        if f_always() != 2 { return 2; };
        if f_never() != 3 { return 3; };
        if s.method_always() != 4 { return 4; };
        if s.method_never() != 5 { return 5; };
        if s.trait_method() != 6 { return 6; };
        0
    }
    "#;

    // 1. Verify JIT execution works
    assert!(run_test("inline_attr_jit", src));

    // 2. Verify emitted LLVM IR contains alwaysinline and noinline attributes
    let path = write_test(src, "inline_attr_ir");
    let output = Command::new(ulang_binary())
        .args(["emit-ir", &path.to_string_lossy()])
        .output()
        .expect("failed to execute ulang emit-ir");
    assert!(output.status.success());
    let ir = String::from_utf8_lossy(&output.stdout);
    assert!(ir.contains("alwaysinline"));
    assert!(ir.contains("noinline"));
}

#[test]
fn test_invalid_inline_attributes() {
    // 1. Invalid argument spelling
    assert!(run_test_expect_error(
        "invalid_inline_arg",
        r#"
        #[inline(foo)]
        fn main() {}
        "#
    ));

    // 2. Too many arguments
    assert!(run_test_expect_error(
        "too_many_inline_args",
        r#"
        #[inline(always, never)]
        fn main() {}
        "#
    ));

    // 3. Conflicting attributes
    assert!(run_test_expect_error(
        "conflicting_inline_attrs",
        r#"
        #[inline(always)]
        #[inline(never)]
        fn main() {}
        "#
    ));
}

#[test]
fn test_inline_on_struct_enum_errors() {
    // 1. inline on struct
    assert!(run_test_expect_error(
        "inline_on_struct",
        r#"
        #[inline]
        struct S {
            x: i32,
        }
        fn main() {}
        "#
    ));

    // 2. inline on enum
    assert!(run_test_expect_error(
        "inline_on_enum",
        r#"
        #[inline]
        enum E {
            A,
        }
        fn main() {}
        "#
    ));
}
