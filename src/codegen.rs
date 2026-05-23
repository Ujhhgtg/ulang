use std::collections::HashMap;
use std::path::Path;

use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine,
};
use inkwell::types::IntType;
use inkwell::values::{BasicValueEnum, IntValue, PointerValue};

use crate::ast::{BinOp, Expr, Function, Program, Stmt};

type MainFunc = unsafe extern "C" fn() -> i32;

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: Option<ExecutionEngine<'ctx>>,
    i32_type: IntType<'ctx>,
    symbols: HashMap<String, PointerValue<'ctx>>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new_jit(context: &'ctx Context) -> Result<Self, String> {
        let module = context.create_module("ulang");
        let execution_engine = module
            .create_jit_execution_engine(OptimizationLevel::None)
            .map_err(|e| format!("failed to create JIT engine: {}", e))?;
        let builder = context.create_builder();
        let i32_type = context.i32_type();

        Ok(Self {
            context,
            module,
            builder,
            execution_engine: Some(execution_engine),
            i32_type,
            symbols: HashMap::new(),
        })
    }

    pub fn new_native(context: &'ctx Context) -> Self {
        let module = context.create_module("ulang");
        let builder = context.create_builder();
        let i32_type = context.i32_type();

        Self {
            context,
            module,
            builder,
            execution_engine: None,
            i32_type,
            symbols: HashMap::new(),
        }
    }

    pub fn compile_module(&mut self, program: &Program) -> Result<(), String> {
        let printf_type = self.i32_type.fn_type(
            &[self.context.ptr_type(AddressSpace::default()).into()],
            true,
        );
        if self.module.get_function("printf").is_none() {
            self.module.add_function("printf", printf_type, None);
        }

        for func in &program.funcs {
            self.compile_function(func)?;
        }
        Ok(())
    }

    pub fn jit_run(&mut self, program: &Program) -> Result<i32, String> {
        self.compile_module(program)?;

        let ee = self
            .execution_engine
            .as_ref()
            .ok_or("JIT execution engine not available")?;

        let main: JitFunction<MainFunc> = unsafe {
            ee.get_function("main")
                .map_err(|e| format!("failed to JIT lookup 'main': {}", e))?
        };

        let result = unsafe { main.call() };
        Ok(result)
    }

    pub fn compile_to_object(&self, path: &Path) -> Result<(), String> {
        Target::initialize_native(&InitializationConfig::default())
            .map_err(|e| format!("failed to initialize native target: {}", e))?;

        let triple = TargetMachine::get_default_triple();
        let target = Target::from_triple(&triple)
            .map_err(|e| format!("failed to get target for native triple: {}", e))?;
        let machine = target
            .create_target_machine(
                &triple,
                "",
                "",
                OptimizationLevel::Default,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or("failed to create target machine")?;

        machine
            .write_to_file(&self.module, FileType::Object, path)
            .map_err(|e| format!("failed to write object file: {}", e))?;
        Ok(())
    }

    pub fn link_executable(obj_path: &Path, exe_path: &Path) -> Result<(), String> {
        let output = std::process::Command::new("cc")
            .arg(obj_path)
            .arg("-o")
            .arg(exe_path)
            .output()
            .map_err(|e| format!("failed to run cc: {}", e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("linker failed: {}", stderr));
        }
        Ok(())
    }

    fn compile_function(&mut self, func: &Function) -> Result<(), String> {
        let fn_type = self.i32_type.fn_type(&[], false);
        let function = self.module.add_function(&func.name, fn_type, None);

        let entry = self.context.append_basic_block(function, "entry");
        self.builder.position_at_end(entry);

        for stmt in &func.body.stmts {
            self.compile_stmt(stmt)?;
        }

        self.builder
            .build_return(Some(&self.i32_type.const_zero()))
            .map_err(|e| format!("failed to build return: {}", e))?;

        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let { name, init, .. } => {
                let alloca = self
                    .builder
                    .build_alloca(self.i32_type, name)
                    .map_err(|e| format!("failed to build alloca: {}", e))?;
                let value = self.compile_expr(init)?;
                self.builder
                    .build_store(alloca, value)
                    .map_err(|e| format!("failed to build store: {}", e))?;
                self.symbols.insert(name.clone(), alloca);
                Ok(())
            }
            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
                Ok(())
            }
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<BasicValueEnum<'ctx>, String> {
        match expr {
            Expr::IntLit(val) => Ok(self.i32_type.const_int(*val as u64, true).into()),
            Expr::Ident(name) => {
                if let Some(ptr) = self.symbols.get(name) {
                    let val = self
                        .builder
                        .build_load(self.i32_type, *ptr, name)
                        .map_err(|e| format!("failed to build load: {}", e))?;
                    Ok(val)
                } else {
                    Err(format!("undefined variable '{}'", name))
                }
            }
            Expr::Call { callee, arg } => {
                if callee == "print" {
                    let arg_val = self.compile_expr(arg)?;
                    let int_val = arg_val.into_int_value();
                    self.emit_printf(int_val)?;
                    Ok(self.i32_type.const_zero().into())
                } else {
                    Err(format!("unknown function '{}'", callee))
                }
            }
            Expr::Binary { op, lhs, rhs } => {
                let lhs_val = self.compile_expr(lhs)?;
                let rhs_val = self.compile_expr(rhs)?;

                let lhs_int = lhs_val.into_int_value();
                let rhs_int = rhs_val.into_int_value();

                let result = match op {
                    BinOp::Add => self
                        .builder
                        .build_int_add(lhs_int, rhs_int, "tmp")
                        .map_err(|e| format!("failed to build add: {}", e))?
                        .into(),
                    BinOp::Sub => self
                        .builder
                        .build_int_sub(lhs_int, rhs_int, "tmp")
                        .map_err(|e| format!("failed to build sub: {}", e))?
                        .into(),
                    BinOp::Mul => self
                        .builder
                        .build_int_mul(lhs_int, rhs_int, "tmp")
                        .map_err(|e| format!("failed to build mul: {}", e))?
                        .into(),
                    BinOp::Div => self
                        .builder
                        .build_int_signed_div(lhs_int, rhs_int, "tmp")
                        .map_err(|e| format!("failed to build div: {}", e))?
                        .into(),
                };
                Ok(result)
            }
        }
    }

    fn emit_printf(&self, value: IntValue<'ctx>) -> Result<(), String> {
        let printf_fn = self
            .module
            .get_function("printf")
            .ok_or("printf not declared")?;

        let fmt = self
            .builder
            .build_global_string_ptr("%d\n", "fmt")
            .map_err(|e| format!("failed to build format string: {}", e))?;

        let fmt_ptr = fmt.as_pointer_value();

        self.builder
            .build_call(printf_fn, &[fmt_ptr.into(), value.into()], "printf_call")
            .map_err(|e| format!("failed to build printf call: {}", e))?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lexer::Lexer;
    use crate::parser::Parser;
    use crate::token::Token;

    fn jit(src: &str) -> Result<i32, String> {
        let mut lexer = Lexer::new(src);
        let mut tokens = Vec::new();
        loop {
            let (token, span) = lexer.next_token().expect("lexer error");
            tokens.push((token, span));
            if matches!(tokens.last().unwrap().0, Token::Eof) {
                break;
            }
        }
        let mut parser = Parser::new(&tokens);
        let program = parser.parse_program().map_err(|e| e.msg)?;

        let context = Context::create();
        let mut cg = CodeGen::new_jit(&context)?;
        cg.jit_run(&program)
    }

    #[test]
    fn test_jit_empty_function() {
        assert_eq!(jit("fn main() {}").unwrap(), 0);
    }

    #[test]
    fn test_jit_with_let() {
        assert_eq!(jit("fn main() { let x = 42; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_arithmetic() {
        assert_eq!(jit("fn main() { 1+2*3; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_undefined_var() {
        let result = jit("fn main() { x; }");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("undefined variable"),
            "error should mention undefined variable"
        );
    }

    #[test]
    fn test_jit_unknown_function() {
        let result = jit("fn main() { foo(1); }");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("unknown function"),
            "error should mention unknown function"
        );
    }

    #[test]
    fn test_jit_two_functions() {
        assert_eq!(jit("fn a(){} fn main(){}").unwrap(), 0);
    }

    #[test]
    fn test_compile_module_valid() {
        let src = "fn f() { let x = 1+2; }";
        let mut lexer = Lexer::new(src);
        let mut tokens = Vec::new();
        loop {
            let (token, span) = lexer.next_token().expect("lexer error");
            tokens.push((token, span));
            if matches!(tokens.last().unwrap().0, Token::Eof) {
                break;
            }
        }
        let mut parser = Parser::new(&tokens);
        let program = parser.parse_program().expect("parse error");

        let context = Context::create();
        let mut cg = CodeGen::new_native(&context);
        assert!(cg.compile_module(&program).is_ok());
    }

    #[test]
    fn test_jit_multiple_statements() {
        assert_eq!(
            jit("fn main() { let x = 10; let y = 20; x + y; }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_call_print() {
        assert_eq!(jit("fn main() { print(42); }").unwrap(), 0);
    }
}
