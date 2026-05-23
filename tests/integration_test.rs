use std::process::Command;
use std::path::Path;

fn ulang_binary() -> String {
    // During `cargo test`, the binary is built as `target/debug/ulang`
    let path = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("target")
        .join("debug")
        .join("ulang");
    path.to_string_lossy().to_string()
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
    // Create a temporary empty .u file
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("tmp_integration");
    let _ = std::fs::create_dir_all(&dir);
    let empty_u = dir.join("empty.u");
    std::fs::write(&empty_u, "fn main() {}\n").expect("write empty test file");

    let output = Command::new(ulang_binary())
        .args(["run", &empty_u.to_string_lossy()])
        .output()
        .expect("failed to execute ulang");

    assert!(
        output.status.success(),
        "ulang run (empty main) exited with code {:?}",
        output.status.code()
    );

    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn test_run_parse_error() {
    let dir = Path::new(env!("CARGO_MANIFEST_DIR")).join("target").join("tmp_integration");
    let _ = std::fs::create_dir_all(&dir);
    let bad_u = dir.join("bad.u");
    std::fs::write(&bad_u, "fn main() { missing_semicolon }\n").expect("write bad test file");

    let output = Command::new(ulang_binary())
        .args(["run", &bad_u.to_string_lossy()])
        .output()
        .expect("failed to execute ulang");

    assert!(
        !output.status.success(),
        "ulang should exit with error on parse error"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
