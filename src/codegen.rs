use std::collections::{HashMap, HashSet};
use std::path::Path;

use inkwell::AddressSpace;
use inkwell::OptimizationLevel;
use inkwell::builder::Builder;
use inkwell::context::Context;
use inkwell::execution_engine::{ExecutionEngine, JitFunction};
use inkwell::module::Module;
use inkwell::targets::{
    CodeModel, FileType, InitializationConfig, RelocMode, Target, TargetMachine, TargetTriple,
};
use inkwell::types::{BasicMetadataTypeEnum, BasicType, BasicTypeEnum, FloatType, IntType};
use inkwell::values::{AnyValue, BasicMetadataValueEnum, BasicValueEnum, PointerValue};

use inkwell::attributes::Attribute;

use crate::ast::{
    BinOp, Block, EnumDecl, Expr, Function, ImplDecl, MatchArm, Pattern, Program, Stmt, StructDecl,
    StructField, Type,
};

use crate::token::Span;

type OverloadMap = HashMap<String, Vec<(String, Vec<Type>)>>;

type MainFunc = unsafe extern "C" fn() -> i32;

pub struct CodeGen<'ctx> {
    context: &'ctx Context,
    module: Module<'ctx>,
    builder: Builder<'ctx>,
    execution_engine: Option<ExecutionEngine<'ctx>>,
    i32_type: IntType<'ctx>,
    i8_type: IntType<'ctx>,
    i16_type: IntType<'ctx>,
    i64_type: IntType<'ctx>,
    bool_type: IntType<'ctx>,
    f32_type: FloatType<'ctx>,
    f64_type: FloatType<'ctx>,
    ptr_int_type: IntType<'ctx>,
    symbols: HashMap<String, (PointerValue<'ctx>, bool, Type)>,
    consts: HashMap<String, i64>,
    pub overloads: OverloadMap,
    opt_level: OptimizationLevel,
    struct_fields: HashMap<String, Vec<StructField>>,
    // Maps type_name -> LLVM struct type
    struct_types: HashMap<String, inkwell::types::StructType<'ctx>>,
    // Maps type_name -> Vec<(method_name, mangled_fn_name)>
    impl_methods: HashMap<String, Vec<(String, String)>>,
    // Maps type_name -> set of trait names implemented
    trait_impls: HashMap<String, HashSet<String>>,
    // Maps function name -> declared return type
    fn_return_types: HashMap<String, Type>,
    // Maps function name -> declared parameter types
    fn_param_types: HashMap<String, Vec<Type>>,
    // Generic struct definitions: base name -> StructDecl
    generic_struct_defs: HashMap<String, StructDecl>,
    // Generic enum definitions: base name -> EnumDecl
    generic_enum_defs: HashMap<String, EnumDecl>,
    // Enum definitions: name -> EnumDecl
    enum_defs: HashMap<String, EnumDecl>,
    // Generic impl blocks: base name -> Vec<(type_params, ImplDecl)>
    generic_impls: HashMap<String, Vec<(Vec<String>, ImplDecl)>>,
    // Set of already-monomorphized concrete instance names
    monomorphized: HashSet<String>,
    // Current monomorphization context: (base_name, mangled_name)
    current_monomorphization: Option<(String, String)>,
    // Maps base_name -> mangled_name for all monomorphized instances
    monomorphized_names: HashMap<String, String>,
    // Slice impl blocks: impl<T> [T] { ... } (stored for on-demand monomorphization)
    slice_impls: Vec<ImplDecl>,
    pub visibility_map: HashMap<String, bool>,
    pub field_visibility_map: HashMap<String, HashMap<String, bool>>,
    pub current_module_path: Vec<String>,
}

impl<'ctx> CodeGen<'ctx> {
    pub fn new_jit(context: &'ctx Context, opt: OptimizationLevel) -> Result<Self, String> {
        let module = context.create_module("ulang");
        let execution_engine = module
            .create_jit_execution_engine(opt)
            .map_err(|e| format!("failed to create JIT engine: {}", e))?;
        let builder = context.create_builder();

        Ok(Self {
            context,
            module,
            builder,
            execution_engine: Some(execution_engine),
            i32_type: context.i32_type(),
            bool_type: context.bool_type(),
            i8_type: context.i8_type(),
            i16_type: context.i16_type(),
            i64_type: context.i64_type(),
            f32_type: context.f32_type(),
            f64_type: context.f64_type(),
            ptr_int_type: context.i64_type(),
            symbols: HashMap::new(),
            consts: HashMap::new(),
            overloads: HashMap::new(),
            struct_fields: HashMap::new(),
            struct_types: HashMap::new(),
            impl_methods: HashMap::new(),
            trait_impls: HashMap::new(),
            fn_return_types: HashMap::new(),
            fn_param_types: HashMap::new(),
            enum_defs: HashMap::new(),
            generic_enum_defs: HashMap::new(),
            generic_struct_defs: HashMap::new(),
            generic_impls: HashMap::new(),
            monomorphized: HashSet::new(),
            current_monomorphization: None,
            monomorphized_names: HashMap::new(),
            slice_impls: Vec::new(),
            visibility_map: HashMap::new(),
            field_visibility_map: HashMap::new(),
            current_module_path: Vec::new(),
            opt_level: opt,
        })
    }

    pub fn new_native(context: &'ctx Context, opt: OptimizationLevel) -> Self {
        let module = context.create_module("ulang");
        let builder = context.create_builder();

        Self {
            context,
            module,
            builder,
            execution_engine: None,
            i32_type: context.i32_type(),
            bool_type: context.bool_type(),
            i8_type: context.i8_type(),
            i16_type: context.i16_type(),
            i64_type: context.i64_type(),
            f32_type: context.f32_type(),
            f64_type: context.f64_type(),
            ptr_int_type: context.i64_type(),
            symbols: HashMap::new(),
            consts: HashMap::new(),
            overloads: HashMap::new(),
            struct_fields: HashMap::new(),
            struct_types: HashMap::new(),
            impl_methods: HashMap::new(),
            trait_impls: HashMap::new(),
            fn_return_types: HashMap::new(),
            fn_param_types: HashMap::new(),
            enum_defs: HashMap::new(),
            generic_enum_defs: HashMap::new(),
            generic_struct_defs: HashMap::new(),
            generic_impls: HashMap::new(),
            monomorphized: HashSet::new(),
            current_monomorphization: None,
            monomorphized_names: HashMap::new(),
            slice_impls: Vec::new(),
            visibility_map: HashMap::new(),
            field_visibility_map: HashMap::new(),
            current_module_path: Vec::new(),
            opt_level: opt,
        }
    }

    fn get_module_path_for_item_name(&self, name: &str) -> Vec<String> {
        let mut clean_name = name.to_string();
        if clean_name.starts_with("__trait_")
            && let Some(idx) = clean_name.strip_prefix("__trait_")
        {
            let parts: Vec<&str> = idx.split('_').collect();
            if parts.len() >= 2 {
                let mut underscore_count = 0;
                let mut start_idx = 0;
                for (i, c) in name.char_indices() {
                    if c == '_' {
                        underscore_count += 1;
                        if underscore_count == 4 {
                            start_idx = i + 1;
                            break;
                        }
                    }
                }
                if start_idx > 0 {
                    clean_name = name[start_idx..].to_string();
                }
            }
        }

        let mut segs: Vec<String> = clean_name.split("::").map(|s| s.to_string()).collect();
        if segs.is_empty() {
            return Vec::new();
        }

        if let Some(last) = segs.last_mut() {
            if let Some(idx) = last.find('/') {
                *last = last[..idx].to_string();
            }
            if let Some(idx) = last.find('$') {
                *last = last[..idx].to_string();
            }
        }

        for (i, seg) in segs.iter().enumerate() {
            if let Some(c) = seg.chars().next()
                && c.is_uppercase()
            {
                return segs[0..i].to_vec();
            }
        }

        if segs.len() > 1 {
            segs[0..segs.len() - 1].to_vec()
        } else {
            Vec::new()
        }
    }

    fn check_visibility_of_path(
        &self,
        caller_path: &[String],
        target_name: &str,
    ) -> Result<(), String> {
        let target_def_path = self.get_module_path_for_item_name(target_name);
        let is_descendant = caller_path.len() >= target_def_path.len()
            && caller_path[0..target_def_path.len()] == target_def_path;
        if is_descendant {
            return Ok(());
        }
        let base_target_name = if let Some(idx) = target_name.find('$') {
            &target_name[..idx]
        } else {
            target_name
        };
        let is_pub = self
            .visibility_map
            .get(base_target_name)
            .copied()
            .or_else(|| self.visibility_map.get(target_name).copied())
            .unwrap_or(true);
        if !is_pub {
            return Err(format!(
                "error: '{}' is private and cannot be accessed from module '{}'",
                target_name,
                caller_path.join("::")
            ));
        }
        Ok(())
    }

    fn check_visibility_of_type(&self, caller_path: &[String], ty: &Type) -> Result<(), String> {
        match ty {
            Type::Struct(name) => {
                self.check_visibility_of_path(caller_path, name)?;
            }
            Type::GenericInstance(name, args) => {
                self.check_visibility_of_path(caller_path, name)?;
                for arg in args {
                    self.check_visibility_of_type(caller_path, arg)?;
                }
            }
            Type::Tuple(elems) => {
                for elem in elems {
                    self.check_visibility_of_type(caller_path, elem)?;
                }
            }
            Type::Ref { inner, .. } | Type::Ptr { inner, .. } => {
                self.check_visibility_of_type(caller_path, inner)?;
            }
            Type::Array { inner, .. } => {
                self.check_visibility_of_type(caller_path, inner)?;
            }
            Type::Alias(name, args) => {
                self.check_visibility_of_path(caller_path, name)?;
                for arg in args {
                    self.check_visibility_of_type(caller_path, arg)?;
                }
            }
            _ => {}
        }
        Ok(())
    }

    fn check_field_visibility(
        &self,
        caller_path: &[String],
        struct_name: &str,
        field_name: &str,
    ) -> Result<(), String> {
        let target_def_path = self.get_module_path_for_item_name(struct_name);
        let is_descendant = caller_path.len() >= target_def_path.len()
            && caller_path[0..target_def_path.len()] == target_def_path;
        if is_descendant {
            return Ok(());
        }
        let is_pub = if let Some(fields) = self.field_visibility_map.get(struct_name) {
            fields.get(field_name).copied().unwrap_or(true)
        } else {
            true
        };
        if !is_pub {
            return Err(format!(
                "error: field '{}' of struct '{}' is private and cannot be accessed from module '{}'",
                field_name,
                struct_name,
                caller_path.join("::")
            ));
        }
        Ok(())
    }

    fn check_struct_literal_construction(
        &self,
        caller_path: &[String],
        struct_name: &str,
    ) -> Result<(), String> {
        let target_def_path = self.get_module_path_for_item_name(struct_name);
        let is_descendant = caller_path.len() >= target_def_path.len()
            && caller_path[0..target_def_path.len()] == target_def_path;
        if is_descendant {
            return Ok(());
        }
        if let Some(fields) = self.field_visibility_map.get(struct_name) {
            for &is_pub in fields.values() {
                if !is_pub {
                    return Err(format!(
                        "error: cannot construct struct '{}' with private fields from module '{}'",
                        struct_name,
                        caller_path.join("::")
                    ));
                }
            }
        }
        Ok(())
    }

    fn resolve_mangled_name(&self, base_name: &str) -> String {
        if let Some((ref base, ref mangled)) = self.current_monomorphization
            && base == base_name
        {
            mangled.clone()
        } else {
            self.monomorphized_names
                .get(base_name)
                .cloned()
                .unwrap_or_else(|| base_name.to_string())
        }
    }

    fn with_expected_type<F, R>(&mut self, ty: &Type, f: F) -> Result<R, String>
    where
        F: FnOnce(&mut Self) -> Result<R, String>,
    {
        let mut saved = None;
        let mut current_ty = ty;
        while let Type::Ref { inner, .. } = current_ty {
            current_ty = inner;
        }
        if let Type::GenericInstance(name, args) = current_ty {
            let mangled = Self::mangle_generic_instance(name, args);
            let prev = self.monomorphized_names.insert(name.clone(), mangled);
            saved = Some((name.clone(), prev));
        }
        let res = f(self);
        if let Some((name, prev)) = saved {
            if let Some(p) = prev {
                self.monomorphized_names.insert(name, p);
            } else {
                self.monomorphized_names.remove(&name);
            }
        }
        res
    }

    /// Resolve
    fn resolve_self_type(func: &mut Function, actual_type: &Type) {
        for param in &mut func.params {
            Self::resolve_type_self(&mut param.ty, actual_type);
        }
        if let Some(ref mut ret_ty) = func.return_type {
            Self::resolve_type_self(ret_ty, actual_type);
        }
    }

    fn resolve_type_self(ty: &mut Type, actual_type: &Type) {
        match ty {
            Type::SelfType => {
                *ty = actual_type.clone();
            }
            Type::Ref { inner, .. }
            | Type::Ptr { inner, .. }
            | Type::Array { inner, .. }
            | Type::GenericArray { inner, .. }
            | Type::Slice { inner, .. } => {
                Self::resolve_type_self(inner, actual_type);
            }
            _ => {}
        }
    }

    fn primitive_type_name(ty: &Type) -> Option<&'static str> {
        match ty {
            Type::Bool => Some("bool"),
            Type::I8 => Some("i8"),
            Type::I16 => Some("i16"),
            Type::I32 => Some("i32"),
            Type::I64 => Some("i64"),
            Type::U8 => Some("u8"),
            Type::U16 => Some("u16"),
            Type::U32 => Some("u32"),
            Type::U64 => Some("u64"),
            Type::Usize => Some("usize"),
            Type::Isize => Some("isize"),
            Type::F32 => Some("f32"),
            Type::F64 => Some("f64"),
            _ => None,
        }
    }

    fn generate_primitive_trait_impls(&mut self) -> Result<(), String> {
        let primitives = [
            Type::Bool,
            Type::I8,
            Type::I16,
            Type::I32,
            Type::I64,
            Type::U8,
            Type::U16,
            Type::U32,
            Type::U64,
            Type::Usize,
            Type::Isize,
            Type::F32,
            Type::F64,
        ];

        for ty in &primitives {
            let ty_name = Self::primitive_type_name(ty).unwrap();
            let llvm_ty = self.type_to_llvm(ty);
            let is_float = Self::is_float(ty);

            // Default::default() — returns zero (false for bool)
            {
                let fn_name = format!("__builtin_Default_default_{}", ty_name);
                let fn_type = llvm_ty.fn_type(&[], false);
                if self.module.get_function(&fn_name).is_none() {
                    let fn_val = self.module.add_function(&fn_name, fn_type, None);
                    let entry = self.context.append_basic_block(fn_val, "entry");
                    self.builder.position_at_end(entry);
                    let default_val: BasicValueEnum<'ctx> = if is_float {
                        llvm_ty.into_float_type().const_float(0.0).into()
                    } else {
                        llvm_ty.into_int_type().const_zero().into()
                    };
                    self.builder
                        .build_return(Some(&default_val))
                        .map_err(|e| format!("failed to build default return: {}", e))?;
                }
                self.impl_methods
                    .entry(ty_name.to_string())
                    .or_default()
                    .push(("default".to_string(), fn_name));
            }

            // Clone::clone(&self) — returns *self
            {
                let fn_name = format!("__builtin_Clone_clone_{}", ty_name);
                let param_types = [self.context.ptr_type(AddressSpace::default()).into()];
                let fn_type = llvm_ty.fn_type(&param_types, false);
                if self.module.get_function(&fn_name).is_none() {
                    let fn_val = self.module.add_function(&fn_name, fn_type, None);
                    let entry = self.context.append_basic_block(fn_val, "entry");
                    self.builder.position_at_end(entry);
                    let ptr_param = fn_val.get_first_param().unwrap();
                    let loaded = self
                        .builder
                        .build_load(llvm_ty, ptr_param.into_pointer_value(), "cloned")
                        .map_err(|e| format!("failed to load for clone: {}", e))?;
                    self.builder
                        .build_return(Some(&loaded))
                        .map_err(|e| format!("failed to build clone return: {}", e))?;
                }
                self.impl_methods
                    .entry(ty_name.to_string())
                    .or_default()
                    .push(("clone".to_string(), fn_name));
            }

            // Eq::eq — returns bool (i1)
            {
                let fn_name = format!("__builtin_Eq_eq_{}", ty_name);
                let param_types = [
                    self.context.ptr_type(AddressSpace::default()).into(),
                    self.context.ptr_type(AddressSpace::default()).into(),
                ];
                let ret_ty: BasicTypeEnum = self.bool_type.into();
                let fn_type = ret_ty.fn_type(&param_types, false);
                if self.module.get_function(&fn_name).is_none() {
                    let fn_val = self.module.add_function(&fn_name, fn_type, None);
                    let entry = self.context.append_basic_block(fn_val, "entry");
                    self.builder.position_at_end(entry);
                    let params = fn_val.get_params();
                    let self_loaded = self
                        .builder
                        .build_load(llvm_ty, params[0].into_pointer_value(), "self")
                        .map_err(|e| format!("failed to load self: {}", e))?;
                    let other_loaded = self
                        .builder
                        .build_load(llvm_ty, params[1].into_pointer_value(), "other")
                        .map_err(|e| format!("failed to load other: {}", e))?;

                    let cmp = if is_float {
                        self.builder
                            .build_float_compare(
                                inkwell::FloatPredicate::OEQ,
                                self_loaded.into_float_value(),
                                other_loaded.into_float_value(),
                                "eq",
                            )
                            .map_err(|e| format!("failed to build float eq: {}", e))?
                    } else {
                        self.builder
                            .build_int_compare(
                                inkwell::IntPredicate::EQ,
                                self_loaded.into_int_value(),
                                other_loaded.into_int_value(),
                                "eq",
                            )
                            .map_err(|e| format!("failed to build int eq: {}", e))?
                    };
                    // cmp is i1, return directly (no zext to i32)
                    self.builder
                        .build_return(Some(&BasicValueEnum::from(cmp)))
                        .map_err(|e| format!("failed to build eq return: {}", e))?;
                }
                self.impl_methods
                    .entry(ty_name.to_string())
                    .or_default()
                    .push(("eq".to_string(), fn_name));
            }

            // Eq::ne — returns !eq (bool, i1)
            {
                let fn_name = format!("__builtin_Eq_ne_{}", ty_name);
                let param_types = [
                    self.context.ptr_type(AddressSpace::default()).into(),
                    self.context.ptr_type(AddressSpace::default()).into(),
                ];
                let ret_ty: BasicTypeEnum = self.bool_type.into();
                let fn_type = ret_ty.fn_type(&param_types, false);
                if self.module.get_function(&fn_name).is_none() {
                    let fn_val = self.module.add_function(&fn_name, fn_type, None);
                    let entry = self.context.append_basic_block(fn_val, "entry");
                    self.builder.position_at_end(entry);
                    let params = fn_val.get_params();
                    let self_loaded = self
                        .builder
                        .build_load(llvm_ty, params[0].into_pointer_value(), "self")
                        .map_err(|e| format!("failed to load self: {}", e))?;
                    let other_loaded = self
                        .builder
                        .build_load(llvm_ty, params[1].into_pointer_value(), "other")
                        .map_err(|e| format!("failed to load other: {}", e))?;

                    // Compute eq, then NOT for ne
                    let eq = if is_float {
                        self.builder
                            .build_float_compare(
                                inkwell::FloatPredicate::OEQ,
                                self_loaded.into_float_value(),
                                other_loaded.into_float_value(),
                                "eq",
                            )
                            .map_err(|e| format!("failed to build float eq for ne: {}", e))?
                    } else {
                        self.builder
                            .build_int_compare(
                                inkwell::IntPredicate::EQ,
                                self_loaded.into_int_value(),
                                other_loaded.into_int_value(),
                                "eq",
                            )
                            .map_err(|e| format!("failed to build int eq for ne: {}", e))?
                    };
                    let ne = self
                        .builder
                        .build_not(eq, "ne")
                        .map_err(|e| format!("failed to build not for ne: {}", e))?;
                    self.builder
                        .build_return(Some(&BasicValueEnum::from(ne)))
                        .map_err(|e| format!("failed to build ne return: {}", e))?;
                }
                self.impl_methods
                    .entry(ty_name.to_string())
                    .or_default()
                    .push(("ne".to_string(), fn_name));
            }

            // Ord::cmp — sign of difference
            {
                let fn_name = format!("__builtin_Ord_cmp_{}", ty_name);
                let param_types = [
                    self.context.ptr_type(AddressSpace::default()).into(),
                    self.context.ptr_type(AddressSpace::default()).into(),
                ];
                let ret_ty: BasicTypeEnum = self.i32_type.into();
                let fn_type = ret_ty.fn_type(&param_types, false);
                if self.module.get_function(&fn_name).is_none() {
                    let fn_val = self.module.add_function(&fn_name, fn_type, None);
                    let entry = self.context.append_basic_block(fn_val, "entry");
                    self.builder.position_at_end(entry);
                    let params = fn_val.get_params();
                    let self_loaded = self
                        .builder
                        .build_load(llvm_ty, params[0].into_pointer_value(), "self")
                        .map_err(|e| format!("failed to load self: {}", e))?;
                    let other_loaded = self
                        .builder
                        .build_load(llvm_ty, params[1].into_pointer_value(), "other")
                        .map_err(|e| format!("failed to load other: {}", e))?;

                    let result = if is_float {
                        let self_f = self_loaded.into_float_value();
                        let other_f = other_loaded.into_float_value();
                        // cmp = self > other ? 1 : self < other ? -1 : 0
                        let gt = self
                            .builder
                            .build_float_compare(
                                inkwell::FloatPredicate::OGT,
                                self_f,
                                other_f,
                                "gt",
                            )
                            .map_err(|e| format!("failed to build gt: {}", e))?;
                        let lt = self
                            .builder
                            .build_float_compare(
                                inkwell::FloatPredicate::OLT,
                                self_f,
                                other_f,
                                "lt",
                            )
                            .map_err(|e| format!("failed to build lt: {}", e))?;
                        let gt_ext = self
                            .builder
                            .build_int_z_extend(gt, self.i32_type, "gt_i32")
                            .map_err(|e| format!("failed to extend gt: {}", e))?;
                        let lt_ext = self
                            .builder
                            .build_int_z_extend(lt, self.i32_type, "lt_i32")
                            .map_err(|e| format!("failed to extend lt: {}", e))?;
                        self.builder
                            .build_int_sub(gt_ext, lt_ext, "cmp")
                            .map_err(|e| format!("failed to build sub for cmp: {}", e))?
                    } else {
                        let self_i = self_loaded.into_int_value();
                        let other_i = other_loaded.into_int_value();
                        // For integers: sign of (self - other)
                        let gt = self
                            .builder
                            .build_int_compare(inkwell::IntPredicate::SGT, self_i, other_i, "gt")
                            .map_err(|e| format!("failed to build gt: {}", e))?;
                        let lt = self
                            .builder
                            .build_int_compare(inkwell::IntPredicate::SLT, self_i, other_i, "lt")
                            .map_err(|e| format!("failed to build lt: {}", e))?;
                        let gt_ext = self
                            .builder
                            .build_int_z_extend(gt, self.i32_type, "gt_i32")
                            .map_err(|e| format!("failed to extend gt: {}", e))?;
                        let lt_ext = self
                            .builder
                            .build_int_z_extend(lt, self.i32_type, "lt_i32")
                            .map_err(|e| format!("failed to extend lt: {}", e))?;
                        self.builder
                            .build_int_sub(gt_ext, lt_ext, "cmp")
                            .map_err(|e| format!("failed to build sub for cmp: {}", e))?
                    };

                    self.builder
                        .build_return(Some(&BasicValueEnum::from(result)))
                        .map_err(|e| format!("failed to build cmp return: {}", e))?;
                }
                self.impl_methods
                    .entry(ty_name.to_string())
                    .or_default()
                    .push(("cmp".to_string(), fn_name));
            }

            // Add, Sub, Mul, Div for numeric primitives
            if ty != &Type::Bool {
                for (trait_name, method_name) in [
                    ("Add", "add"),
                    ("Sub", "sub"),
                    ("Mul", "mul"),
                    ("Div", "div"),
                ] {
                    let fn_name = format!("__builtin_{}_{}_{}", trait_name, method_name, ty_name);
                    let param_types = [
                        self.context.ptr_type(AddressSpace::default()).into(),
                        self.context.ptr_type(AddressSpace::default()).into(),
                    ];
                    let ret_ty: BasicTypeEnum = llvm_ty;
                    let fn_type = ret_ty.fn_type(&param_types, false);
                    if self.module.get_function(&fn_name).is_none() {
                        let fn_val = self.module.add_function(&fn_name, fn_type, None);
                        let entry = self.context.append_basic_block(fn_val, "entry");
                        self.builder.position_at_end(entry);
                        let params = fn_val.get_params();
                        let self_loaded = self
                            .builder
                            .build_load(llvm_ty, params[0].into_pointer_value(), "self")
                            .map_err(|e| format!("failed to load self: {}", e))?;
                        let other_loaded = self
                            .builder
                            .build_load(llvm_ty, params[1].into_pointer_value(), "other")
                            .map_err(|e| format!("failed to load other: {}", e))?;

                        let res: BasicValueEnum<'ctx> = match trait_name {
                            "Add" => {
                                if is_float {
                                    self.builder
                                        .build_float_add(
                                            self_loaded.into_float_value(),
                                            other_loaded.into_float_value(),
                                            "add",
                                        )
                                        .map_err(|e| format!("failed to build float add: {}", e))?
                                        .into()
                                } else {
                                    self.builder
                                        .build_int_add(
                                            self_loaded.into_int_value(),
                                            other_loaded.into_int_value(),
                                            "add",
                                        )
                                        .map_err(|e| format!("failed to build int add: {}", e))?
                                        .into()
                                }
                            }
                            "Sub" => {
                                if is_float {
                                    self.builder
                                        .build_float_sub(
                                            self_loaded.into_float_value(),
                                            other_loaded.into_float_value(),
                                            "sub",
                                        )
                                        .map_err(|e| format!("failed to build float sub: {}", e))?
                                        .into()
                                } else {
                                    self.builder
                                        .build_int_sub(
                                            self_loaded.into_int_value(),
                                            other_loaded.into_int_value(),
                                            "sub",
                                        )
                                        .map_err(|e| format!("failed to build int sub: {}", e))?
                                        .into()
                                }
                            }
                            "Mul" => {
                                if is_float {
                                    self.builder
                                        .build_float_mul(
                                            self_loaded.into_float_value(),
                                            other_loaded.into_float_value(),
                                            "mul",
                                        )
                                        .map_err(|e| format!("failed to build float mul: {}", e))?
                                        .into()
                                } else {
                                    self.builder
                                        .build_int_mul(
                                            self_loaded.into_int_value(),
                                            other_loaded.into_int_value(),
                                            "mul",
                                        )
                                        .map_err(|e| format!("failed to build int mul: {}", e))?
                                        .into()
                                }
                            }
                            "Div" => {
                                if is_float {
                                    self.builder
                                        .build_float_div(
                                            self_loaded.into_float_value(),
                                            other_loaded.into_float_value(),
                                            "div",
                                        )
                                        .map_err(|e| format!("failed to build float div: {}", e))?
                                        .into()
                                } else if Self::is_signed(ty) {
                                    self.builder
                                        .build_int_signed_div(
                                            self_loaded.into_int_value(),
                                            other_loaded.into_int_value(),
                                            "div",
                                        )
                                        .map_err(|e| format!("failed to build signed div: {}", e))?
                                        .into()
                                } else {
                                    self.builder
                                        .build_int_unsigned_div(
                                            self_loaded.into_int_value(),
                                            other_loaded.into_int_value(),
                                            "div",
                                        )
                                        .map_err(|e| {
                                            format!("failed to build unsigned div: {}", e)
                                        })?
                                        .into()
                                }
                            }
                            _ => unreachable!(),
                        };

                        self.builder.build_return(Some(&res)).map_err(|e| {
                            format!("failed to build {} return: {}", method_name, e)
                        })?;
                    }
                    self.impl_methods
                        .entry(ty_name.to_string())
                        .or_default()
                        .push((method_name.to_string(), fn_name));
                }
            }

            // Register in trait_impls
            let mut traits: HashSet<String> = ["Default", "Clone", "Eq", "Ord"]
                .iter()
                .map(|s| s.to_string())
                .collect();
            if ty != &Type::Bool {
                traits.insert("Add".to_string());
                traits.insert("Sub".to_string());
                traits.insert("Mul".to_string());
                traits.insert("Div".to_string());
            }
            self.trait_impls.insert(ty_name.to_string(), traits);
        }
        Ok(())
    }

    /// Convert a type to its string name for trait method naming.
    fn type_to_mangled_name(ty: &Type) -> String {
        if let Some(name) = Self::primitive_type_name(ty) {
            return name.to_string();
        }
        match ty {
            Type::Struct(name) => name.clone(),
            Type::GenericInstance(name, _) => name.clone(),
            Type::Ref { inner, .. } => Self::type_to_mangled_name(inner),
            Type::Array { inner, len } => {
                format!("array_{}_{}", Self::type_to_mangled_name(inner), len)
            }
            _ => panic!("unsupported type for trait method: {:?}", ty),
        }
    }

    /// Get the mangled function name for a trait method on a given type.
    fn trait_method_name(ty: &Type, trait_name: &str, method_name: &str) -> String {
        let type_name = Self::type_to_mangled_name(ty);
        format!("__builtin_{}_{}_{}", trait_name, method_name, type_name)
    }

    /// Check if a type implements a given trait.
    fn check_type_implements_trait(&self, ty: &Type, trait_name: &str) -> bool {
        match ty {
            Type::Bool
            | Type::I8
            | Type::I16
            | Type::I32
            | Type::I64
            | Type::U8
            | Type::U16
            | Type::U32
            | Type::U64
            | Type::Usize
            | Type::Isize
            | Type::F32
            | Type::F64 => true,
            Type::Struct(name) => self
                .trait_impls
                .get(name)
                .map(|traits| traits.contains(trait_name))
                .unwrap_or(false),
            Type::Ref { inner, .. } => self.check_type_implements_trait(inner, trait_name),
            Type::Array { .. } => {
                // Arrays implement Index/IndexMut (builtin), and also
                // any trait their element type implements (for derive checks)
                trait_name == "Index" || trait_name == "IndexMut"
            }
            // Tuple/unit types are not derivable (no plan for these)
            _ => false,
        }
    }

    /// Process all #[derive(...)] attributes on a struct.
    fn process_struct_derives(
        &mut self,
        decl: &StructDecl,
        _program: &Program,
    ) -> Result<(), String> {
        for attr in &decl.attribs {
            if attr.name != "derive" {
                continue;
            }
            for trait_name in &attr.args {
                self.process_derive_trait(decl, trait_name)?;
            }
        }
        Ok(())
    }

    /// Process a single derived trait for a struct.
    fn process_derive_trait(&mut self, decl: &StructDecl, trait_name: &str) -> Result<(), String> {
        // Validate all fields implement the trait
        for field in &decl.fields {
            if !self.check_type_implements_trait(&field.ty, trait_name) {
                return Err(format!(
                    "cannot derive trait '{}' for struct '{}': field '{}' of type {:?} does not implement {}",
                    trait_name, decl.name, field.name, field.ty, trait_name
                ));
            }
        }

        // Generate implementations based on trait
        match trait_name {
            "Default" => self.generate_derive_default(decl)?,
            "Clone" => self.generate_derive_clone(decl)?,
            "Eq" => self.generate_derive_eq(decl)?,
            "Ord" => self.generate_derive_ord(decl, "Ord", "cmp")?,
            _ => {
                return Err(format!(
                    "cannot derive unknown trait '{}' for struct '{}'",
                    trait_name, decl.name
                ));
            }
        }

        // Register trait in trait_impls
        self.trait_impls
            .entry(decl.name.clone())
            .or_default()
            .insert(trait_name.to_string());

        Ok(())
    }

    /// Generate `Default::default()` for a struct.
    fn generate_derive_default(&mut self, decl: &StructDecl) -> Result<(), String> {
        let fn_name =
            Self::trait_method_name(&Type::Struct(decl.name.clone()), "Default", "default");
        let struct_ty = self
            .struct_types
            .get(&decl.name)
            .copied()
            .ok_or_else(|| format!("struct type '{}' not declared", decl.name))?;
        let fn_type = struct_ty.fn_type(&[], false);

        // Skip if already declared
        if self.module.get_function(&fn_name).is_some() {
            return Ok(());
        }

        let fn_val = self.module.add_function(&fn_name, fn_type, None);
        let entry = self.context.append_basic_block(fn_val, "entry");
        self.builder.position_at_end(entry);

        // Build struct with default values for each field
        let mut result: inkwell::values::AggregateValueEnum<'ctx> = struct_ty.get_undef().into();

        for (i, field) in decl.fields.iter().enumerate() {
            let field_trait_fn = Self::trait_method_name(&field.ty, "Default", "default");
            let field_fn = self.module.get_function(&field_trait_fn).ok_or_else(|| {
                format!(
                    "internal: default function '{}' not found for field '{}'",
                    field_trait_fn, field.name
                )
            })?;

            let field_val = self
                .builder
                .build_call(field_fn, &[], "field_default")
                .map_err(|e| format!("failed to call default for field '{}': {}", field.name, e))?;

            let field_any = self.try_extract_result(field_val);
            result = self
                .builder
                .build_insert_value(result, field_any, i as u32, &field.name)
                .map_err(|e| format!("failed to insert default field '{}': {}", field.name, e))?;
        }

        match result {
            inkwell::values::AggregateValueEnum::StructValue(sv) => {
                self.builder
                    .build_return(Some(&BasicValueEnum::from(sv)))
                    .map_err(|e| format!("failed to build default return: {}", e))?;
            }
            _ => {
                return Err("expected struct value for default".to_string());
            }
        }

        // Register method
        self.impl_methods
            .entry(decl.name.clone())
            .or_default()
            .push(("default".to_string(), fn_name));

        Ok(())
    }

    /// Generate `Clone::clone(&self)` for a struct.
    fn generate_derive_clone(&mut self, decl: &StructDecl) -> Result<(), String> {
        let fn_name = Self::trait_method_name(&Type::Struct(decl.name.clone()), "Clone", "clone");
        let struct_ty = self
            .struct_types
            .get(&decl.name)
            .copied()
            .ok_or_else(|| format!("struct type '{}' not declared", decl.name))?;
        let param_types = [self.context.ptr_type(AddressSpace::default()).into()];
        let fn_type = struct_ty.fn_type(&param_types, false);

        if self.module.get_function(&fn_name).is_some() {
            return Ok(());
        }

        let fn_val = self.module.add_function(&fn_name, fn_type, None);
        let entry = self.context.append_basic_block(fn_val, "entry");
        self.builder.position_at_end(entry);

        let self_ptr = fn_val.get_first_param().unwrap().into_pointer_value();
        let self_struct = self
            .builder
            .build_load(struct_ty, self_ptr, "self")
            .map_err(|e| format!("failed to load self for clone: {}", e))?;

        let mut result: inkwell::values::AggregateValueEnum<'ctx> = struct_ty.get_undef().into();

        for (i, field) in decl.fields.iter().enumerate() {
            let field_val = self
                .builder
                .build_extract_value(self_struct.into_struct_value(), i as u32, &field.name)
                .map_err(|e| {
                    format!("failed to extract field '{}' for clone: {}", field.name, e)
                })?;

            let field_trait_fn = Self::trait_method_name(&field.ty, "Clone", "clone");
            let field_fn = self.module.get_function(&field_trait_fn).ok_or_else(|| {
                format!(
                    "internal: clone function '{}' not found for field '{}'",
                    field_trait_fn, field.name
                )
            })?;

            // Store field value to alloca and pass pointer to clone function
            let field_llvm_ty = self.type_to_llvm(&field.ty);
            let field_alloca = self
                .builder
                .build_alloca(field_llvm_ty, &format!("{}_clone", field.name))
                .map_err(|e| format!("failed to build alloca for clone: {}", e))?;
            self.builder
                .build_store(field_alloca, field_val)
                .map_err(|e| format!("failed to store field for clone: {}", e))?;

            let cloned_val = self
                .builder
                .build_call(field_fn, &[field_alloca.into()], "field_clone")
                .map_err(|e| format!("failed to call clone for field '{}': {}", field.name, e))?;

            let cloned_any = self.try_extract_result(cloned_val);
            result = self
                .builder
                .build_insert_value(result, cloned_any, i as u32, &field.name)
                .map_err(|e| format!("failed to insert cloned field '{}': {}", field.name, e))?;
        }

        match result {
            inkwell::values::AggregateValueEnum::StructValue(sv) => {
                self.builder
                    .build_return(Some(&BasicValueEnum::from(sv)))
                    .map_err(|e| format!("failed to build clone return: {}", e))?;
            }
            _ => {
                return Err("expected struct value for clone".to_string());
            }
        }

        self.impl_methods
            .entry(decl.name.clone())
            .or_default()
            .push(("clone".to_string(), fn_name));

        Ok(())
    }

    /// Generate `Eq::eq(&self, other: &Self) -> bool` and `ne` for a struct.
    fn generate_derive_eq(&mut self, decl: &StructDecl) -> Result<(), String> {
        let struct_ty = self
            .struct_types
            .get(&decl.name)
            .copied()
            .ok_or_else(|| format!("struct type '{}' not declared", decl.name))?;

        // Generate eq
        let eq_fn_name = Self::trait_method_name(&Type::Struct(decl.name.clone()), "Eq", "eq");
        let param_types = [
            self.context.ptr_type(AddressSpace::default()).into(),
            self.context.ptr_type(AddressSpace::default()).into(),
        ];
        let ret_ty: BasicTypeEnum = self.bool_type.into();
        let eq_fn_type = ret_ty.fn_type(&param_types, false);

        let eq_fn = if self.module.get_function(&eq_fn_name).is_none() {
            let fn_val = self.module.add_function(&eq_fn_name, eq_fn_type, None);
            let entry = self.context.append_basic_block(fn_val, "entry");
            self.builder.position_at_end(entry);

            let params = fn_val.get_params();
            let self_struct = self
                .builder
                .build_load(struct_ty, params[0].into_pointer_value(), "self")
                .map_err(|e| format!("failed to load self for eq: {}", e))?;
            let other_struct = self
                .builder
                .build_load(struct_ty, params[1].into_pointer_value(), "other")
                .map_err(|e| format!("failed to load other for eq: {}", e))?;

            let mut result_val = self.bool_type.const_int(1, false);

            for (i, field) in decl.fields.iter().enumerate() {
                let self_field = self
                    .builder
                    .build_extract_value(self_struct.into_struct_value(), i as u32, &field.name)
                    .map_err(|e| format!("failed to extract self field '{}': {}", field.name, e))?;
                let other_field = self
                    .builder
                    .build_extract_value(other_struct.into_struct_value(), i as u32, &field.name)
                    .map_err(|e| {
                        format!("failed to extract other field '{}': {}", field.name, e)
                    })?;

                let eq_trait_fn = Self::trait_method_name(&field.ty, "Eq", "eq");
                let field_eq_fn = self.module.get_function(&eq_trait_fn).ok_or_else(|| {
                    format!(
                        "internal: eq function '{}' not found for field '{}'",
                        eq_trait_fn, field.name
                    )
                })?;

                // Store both field values to allocas and pass pointers to eq
                let field_llvm_ty = self.type_to_llvm(&field.ty);
                let self_alloca = self
                    .builder
                    .build_alloca(field_llvm_ty, &format!("{}_self", field.name))
                    .map_err(|e| format!("failed to build alloca for eq: {}", e))?;
                self.builder
                    .build_store(self_alloca, self_field)
                    .map_err(|e| format!("failed to store self field for eq: {}", e))?;
                let other_alloca = self
                    .builder
                    .build_alloca(field_llvm_ty, &format!("{}_other", field.name))
                    .map_err(|e| format!("failed to build alloca for eq: {}", e))?;
                self.builder
                    .build_store(other_alloca, other_field)
                    .map_err(|e| format!("failed to store other field for eq: {}", e))?;

                let field_result = self
                    .builder
                    .build_call(
                        field_eq_fn,
                        &[self_alloca.into(), other_alloca.into()],
                        "field_eq",
                    )
                    .map_err(|e| format!("failed to call eq for field '{}': {}", field.name, e))?;
                let field_result_bool = self.try_extract_result(field_result).into_int_value();

                result_val = self
                    .builder
                    .build_and(result_val, field_result_bool, "and")
                    .map_err(|e| format!("failed to build and for eq: {}", e))?;
            }

            self.builder
                .build_return(Some(&BasicValueEnum::from(result_val)))
                .map_err(|e| format!("failed to build eq return: {}", e))?;

            fn_val
        } else {
            self.module.get_function(&eq_fn_name).unwrap()
        };
        let _ = eq_fn; // used for ne

        self.impl_methods
            .entry(decl.name.clone())
            .or_default()
            .push(("eq".to_string(), eq_fn_name.clone()));

        // Generate ne: return !self.eq(other)
        let ne_fn_name = Self::trait_method_name(&Type::Struct(decl.name.clone()), "Eq", "ne");
        let ne_fn_type = ret_ty.fn_type(&param_types, false);

        if self.module.get_function(&ne_fn_name).is_none() {
            let fn_val = self.module.add_function(&ne_fn_name, ne_fn_type, None);
            let entry = self.context.append_basic_block(fn_val, "entry");
            self.builder.position_at_end(entry);

            let params = fn_val.get_params();
            let self_struct = self
                .builder
                .build_load(struct_ty, params[0].into_pointer_value(), "self")
                .map_err(|e| format!("failed to load self for ne: {}", e))?;
            let other_struct = self
                .builder
                .build_load(struct_ty, params[1].into_pointer_value(), "other")
                .map_err(|e| format!("failed to load other for ne: {}", e))?;

            let field_llvm_ty = self.type_to_llvm(&Type::Struct(decl.name.clone()));
            let self_alloca = self
                .builder
                .build_alloca(field_llvm_ty, "ne_self")
                .map_err(|e| format!("failed to build alloca for ne self: {}", e))?;
            self.builder
                .build_store(self_alloca, self_struct)
                .map_err(|e| format!("failed to store self for ne: {}", e))?;
            let other_alloca = self
                .builder
                .build_alloca(field_llvm_ty, "ne_other")
                .map_err(|e| format!("failed to build alloca for ne other: {}", e))?;
            self.builder
                .build_store(other_alloca, other_struct)
                .map_err(|e| format!("failed to store other for ne: {}", e))?;

            let eq_result = self
                .builder
                .build_call(
                    self.module.get_function(&eq_fn_name).unwrap(),
                    &[self_alloca.into(), other_alloca.into()],
                    "eq_call",
                )
                .map_err(|e| format!("failed to call eq for ne: {}", e))?;
            let eq_val = self.try_extract_result(eq_result).into_int_value();

            // ne = !eq (i1)
            let ne_val = self
                .builder
                .build_not(eq_val, "ne")
                .map_err(|e| format!("failed to build not for ne: {}", e))?;

            self.builder
                .build_return(Some(&BasicValueEnum::from(ne_val)))
                .map_err(|e| format!("failed to build ne return: {}", e))?;
        }

        self.impl_methods
            .entry(decl.name.clone())
            .or_default()
            .push(("ne".to_string(), ne_fn_name));

        Ok(())
    }

    /// Generate `cmp` with lexicographic ordering for a struct.
    fn generate_derive_ord(
        &mut self,
        decl: &StructDecl,
        trait_name: &str,
        cmp_name: &str,
    ) -> Result<(), String> {
        let struct_ty = self
            .struct_types
            .get(&decl.name)
            .copied()
            .ok_or_else(|| format!("struct type '{}' not declared", decl.name))?;

        let fn_name =
            Self::trait_method_name(&Type::Struct(decl.name.clone()), trait_name, cmp_name);
        let param_types = [
            self.context.ptr_type(AddressSpace::default()).into(),
            self.context.ptr_type(AddressSpace::default()).into(),
        ];
        let ret_ty: BasicTypeEnum = self.i32_type.into();
        let fn_type = ret_ty.fn_type(&param_types, false);

        if self.module.get_function(&fn_name).is_some() {
            return Ok(());
        }

        let fn_val = self.module.add_function(&fn_name, fn_type, None);
        let entry = self.context.append_basic_block(fn_val, "entry");
        self.builder.position_at_end(entry);

        let params = fn_val.get_params();
        let self_struct = self
            .builder
            .build_load(struct_ty, params[0].into_pointer_value(), "self")
            .map_err(|e| format!("failed to load self for {}: {}", cmp_name, e))?;
        let other_struct = self
            .builder
            .build_load(struct_ty, params[1].into_pointer_value(), "other")
            .map_err(|e| format!("failed to load other for {}: {}", cmp_name, e))?;

        // Pre-store structs into allocas and extract fields for each comparison
        let struct_llvm_ty = self.type_to_llvm(&Type::Struct(decl.name.clone()));
        let self_alloca = self
            .builder
            .build_alloca(struct_llvm_ty, "cmp_self")
            .map_err(|e| format!("failed to build alloca for cmp self: {}", e))?;
        self.builder
            .build_store(self_alloca, self_struct)
            .map_err(|e| format!("failed to store self for cmp: {}", e))?;
        let other_alloca = self
            .builder
            .build_alloca(struct_llvm_ty, "cmp_other")
            .map_err(|e| format!("failed to build alloca for cmp other: {}", e))?;
        self.builder
            .build_store(other_alloca, other_struct)
            .map_err(|e| format!("failed to store other for cmp: {}", e))?;

        let num_fields = decl.fields.len();
        // We'll use the entry block for the first field comparison
        // and create blocks for subsequent fields

        // If no fields, return 0
        if num_fields == 0 {
            self.builder
                .build_return(Some(&BasicValueEnum::from(self.i32_type.const_zero())))
                .map_err(|e| format!("failed to build empty struct cmp return: {}", e))?;

            self.impl_methods
                .entry(decl.name.clone())
                .or_default()
                .push((cmp_name.to_string(), fn_name));
            return Ok(());
        }

        // Process the first field in the entry block
        let mut current_block = self.builder.get_insert_block().unwrap();
        let mut next_block: Option<inkwell::basic_block::BasicBlock<'ctx>> = None;

        for (i, field) in decl.fields.iter().enumerate() {
            let is_last = i == num_fields - 1;

            // Create the next block (for when cmp == 0)
            if !is_last {
                let check_next = self
                    .context
                    .append_basic_block(fn_val, &format!("field_{}", i));
                next_block = Some(check_next);
            }

            // Position builder at current block
            self.builder.position_at_end(current_block);

            // Load self and other structs from allocas
            let self_loaded = self
                .builder
                .build_load(struct_ty, self_alloca, "self_reload")
                .map_err(|e| format!("failed to reload self: {}", e))?;
            let other_loaded = self
                .builder
                .build_load(struct_ty, other_alloca, "other_reload")
                .map_err(|e| format!("failed to reload other: {}", e))?;

            let self_field = self
                .builder
                .build_extract_value(self_loaded.into_struct_value(), i as u32, &field.name)
                .map_err(|e| format!("failed to extract self field '{}': {}", field.name, e))?;
            let other_field = self
                .builder
                .build_extract_value(other_loaded.into_struct_value(), i as u32, &field.name)
                .map_err(|e| format!("failed to extract other field '{}': {}", field.name, e))?;

            let cmp_trait_fn = Self::trait_method_name(&field.ty, trait_name, cmp_name);
            let field_cmp_fn = self.module.get_function(&cmp_trait_fn).ok_or_else(|| {
                format!(
                    "internal: {} function '{}' not found for field '{}'",
                    cmp_name, cmp_trait_fn, field.name
                )
            })?;

            let field_llvm_ty = self.type_to_llvm(&field.ty);
            let self_field_alloca = self
                .builder
                .build_alloca(field_llvm_ty, &format!("{}_self", field.name))
                .map_err(|e| format!("failed to build alloca for cmp: {}", e))?;
            self.builder
                .build_store(self_field_alloca, self_field)
                .map_err(|e| format!("failed to store self field for cmp: {}", e))?;
            let other_field_alloca = self
                .builder
                .build_alloca(field_llvm_ty, &format!("{}_other", field.name))
                .map_err(|e| format!("failed to build alloca for cmp: {}", e))?;
            self.builder
                .build_store(other_field_alloca, other_field)
                .map_err(|e| format!("failed to store other field for cmp: {}", e))?;

            let field_result = self
                .builder
                .build_call(
                    field_cmp_fn,
                    &[self_field_alloca.into(), other_field_alloca.into()],
                    &format!("field_{}_cmp", i),
                )
                .map_err(|e| {
                    format!(
                        "failed to call {} for field '{}': {}",
                        cmp_name, field.name, e
                    )
                })?;
            let cmp_result = self.try_extract_result(field_result).into_int_value();

            if is_last {
                // Last field: return the comparison result directly
                self.builder
                    .build_return(Some(&BasicValueEnum::from(cmp_result)))
                    .map_err(|e| format!("failed to build cmp return: {}", e))?;
            } else {
                // Not last: check if result != 0 and return early
                let is_nonzero = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::NE,
                        cmp_result,
                        self.i32_type.const_zero(),
                        "is_nonzero",
                    )
                    .map_err(|e| format!("failed to compare cmp result: {}", e))?;

                let ret_block = self
                    .context
                    .append_basic_block(fn_val, &format!("ret_field_{}", i));
                self.builder.position_at_end(ret_block);
                self.builder
                    .build_return(Some(&BasicValueEnum::from(cmp_result)))
                    .map_err(|e| format!("failed to build early return: {}", e))?;

                // Branch from current block
                self.builder.position_at_end(current_block);
                self.builder
                    .build_conditional_branch(is_nonzero, ret_block, next_block.unwrap())
                    .map_err(|e| format!("failed to build conditional branch: {}", e))?;

                current_block = next_block.unwrap();
            }
        }

        self.impl_methods
            .entry(decl.name.clone())
            .or_default()
            .push((cmp_name.to_string(), fn_name));

        Ok(())
    }

    /// Check if a type is a type parameter (single uppercase letter name).
    fn is_type_param(ty: &Type) -> bool {
        match ty {
            Type::Struct(name) => {
                name.len() == 1 && name.chars().next().is_some_and(|c| c.is_uppercase())
            }
            _ => false,
        }
    }

    /// Mangle a generic instance name: BaseName__Arg1_Arg2_...
    fn mangle_generic_instance(base_name: &str, args: &[Type]) -> String {
        if args.is_empty() {
            return format!("{}__", base_name);
        }
        let arg_strs: Vec<String> = args
            .iter()
            .map(|a| {
                if let Some(prim) = Self::primitive_type_name(a) {
                    prim.to_string()
                } else {
                    match a {
                        Type::Struct(name) => name.clone(),
                        Type::GenericInstance(name, inner_args) => {
                            Self::mangle_generic_instance(name, inner_args)
                        }
                        Type::Ref { inner, .. } => {
                            format!("ref_{}", Self::mangle_generic_instance_inner(inner))
                        }
                        Type::Ptr { inner, .. } => {
                            format!("ptr_{}", Self::mangle_generic_instance_inner(inner))
                        }
                        Type::Array { inner, len } => {
                            format!(
                                "array_{}_{}",
                                Self::mangle_generic_instance_inner(inner),
                                len
                            )
                        }
                        _ => format!("{:?}", a),
                    }
                }
            })
            .collect();
        format!("{}__{}", base_name, arg_strs.join("_"))
    }

    fn mangle_generic_instance_inner(ty: &Type) -> String {
        if let Some(prim) = Self::primitive_type_name(ty) {
            return prim.to_string();
        }
        match ty {
            Type::Struct(name) => name.clone(),
            Type::GenericInstance(name, args) => Self::mangle_generic_instance(name, args),
            Type::Array { inner, len } => {
                format!(
                    "array_{}_{}",
                    Self::mangle_generic_instance_inner(inner),
                    len
                )
            }
            _ => format!("{:?}", ty),
        }
    }

    /// Recursively substitute type params in an expression's Cast types.
    fn m_substitute_types_in_expr(expr: &mut Box<Expr>, params: &[String], args: &[Type]) {
        match expr.as_mut() {
            Expr::Cast { to_type, .. } => {
                Self::substitute_type_params(to_type, params, args);
            }
            Expr::Call {
                args: call_args, ..
            }
            | Expr::QualifiedCall {
                args: call_args, ..
            } => {
                for arg in call_args.iter_mut() {
                    let mut inner = Box::new(arg.clone());
                    Self::m_substitute_types_in_expr(&mut inner, params, args);
                    *arg = *inner;
                }
            }
            Expr::EnumLit {
                enum_name, payload, ..
            } => {
                if let Some(pos) = params.iter().position(|p| p == enum_name) {
                    if let Some(Type::GenericInstance(base, _)) = args.get(pos) {
                        *enum_name = base.clone();
                    } else if let Some(arg) = args.get(pos)
                        && let Some(name) = Self::primitive_type_name(arg)
                    {
                        *enum_name = name.to_string();
                    }
                }
                if let Some(inner) = payload {
                    let mut boxed = Box::new((**inner).clone());
                    Self::m_substitute_types_in_expr(&mut boxed, params, args);
                    **inner = *boxed;
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                Self::m_substitute_types_in_expr(lhs, params, args);
                Self::m_substitute_types_in_expr(rhs, params, args);
            }
            Expr::UnaryNot(inner) => {
                Self::m_substitute_types_in_expr(inner, params, args);
            }
            Expr::UnaryMinus(inner) => {
                Self::m_substitute_types_in_expr(inner, params, args);
            }
            Expr::If {
                cond,
                then_block,
                else_ifs,
                else_block,
            } => {
                let mut cond_box = Box::new((**cond).clone());
                Self::m_substitute_types_in_expr(&mut cond_box, params, args);
                *cond = cond_box;
                Self::m_substitute_block(then_block, params, args);
                for (elif_cond, elif_block) in else_ifs.iter_mut() {
                    let mut ec = Box::new(elif_cond.clone());
                    Self::m_substitute_types_in_expr(&mut ec, params, args);
                    *elif_cond = *ec;
                    Self::m_substitute_block(elif_block, params, args);
                }
                if let Some(eb) = else_block {
                    Self::m_substitute_block(eb, params, args);
                }
            }
            Expr::Loop { body } => {
                Self::m_substitute_block(body, params, args);
            }
            Expr::While { cond, body } => {
                let mut cond_box = Box::new((**cond).clone());
                Self::m_substitute_types_in_expr(&mut cond_box, params, args);
                *cond = cond_box;
                Self::m_substitute_block(body, params, args);
            }
            Expr::IfLet {
                scrutinee,
                then_block,
                else_block,
                ..
            } => {
                Self::m_substitute_types_in_expr(scrutinee, params, args);
                Self::m_substitute_block(then_block, params, args);
                if let Some(eb) = else_block {
                    Self::m_substitute_block(eb, params, args);
                }
            }
            Expr::Match { scrutinee, arms } => {
                Self::m_substitute_types_in_expr(scrutinee, params, args);
                for arm in arms.iter_mut() {
                    if let Some(ref mut guard) = arm.guard {
                        Self::m_substitute_types_in_expr(guard, params, args);
                    }
                    Self::m_substitute_block(&mut arm.body, params, args);
                }
            }
            Expr::Assign { target, value } => {
                Self::m_substitute_types_in_expr(target, params, args);
                Self::m_substitute_types_in_expr(value, params, args);
            }
            Expr::Ref { expr: inner, .. } => {
                Self::m_substitute_types_in_expr(inner, params, args);
            }
            Expr::Deref(inner) => {
                Self::m_substitute_types_in_expr(inner, params, args);
            }
            Expr::Member { expr: inner, .. } => {
                Self::m_substitute_types_in_expr(inner, params, args);
            }
            Expr::MethodCall {
                expr: receiver,
                args: call_args,
                ..
            } => {
                Self::m_substitute_types_in_expr(receiver, params, args);
                for arg in call_args.iter_mut() {
                    let mut inner = Box::new(arg.clone());
                    Self::m_substitute_types_in_expr(&mut inner, params, args);
                    *arg = *inner;
                }
            }
            Expr::Tuple(elems) => {
                for elem in elems.iter_mut() {
                    let mut inner = Box::new(elem.clone());
                    Self::m_substitute_types_in_expr(&mut inner, params, args);
                    *elem = *inner;
                }
            }
            Expr::StructLit {
                struct_name,
                fields,
                ..
            } => {
                if let Some(pos) = params.iter().position(|p| p == struct_name) {
                    if let Some(Type::GenericInstance(base, _)) = args.get(pos) {
                        *struct_name = base.clone();
                    } else if let Some(arg) = args.get(pos)
                        && let Some(name) = Self::primitive_type_name(arg)
                    {
                        *struct_name = name.to_string();
                    }
                }
                for (_, expr) in fields.iter_mut() {
                    let mut inner = Box::new(expr.clone());
                    Self::m_substitute_types_in_expr(&mut inner, params, args);
                    *expr = *inner;
                }
            }
            Expr::Array(elems) => {
                for elem in elems.iter_mut() {
                    let mut inner = Box::new(elem.clone());
                    Self::m_substitute_types_in_expr(&mut inner, params, args);
                    *elem = *inner;
                }
            }
            Expr::Repeat(expr, _) => {
                Self::m_substitute_types_in_expr(expr, params, args);
            }
            Expr::Index { array, index } => {
                Self::m_substitute_types_in_expr(array, params, args);
                Self::m_substitute_types_in_expr(index, params, args);
            }
            _ => {}
        }
    }

    /// Substitute type params in a block (let type annotations, expression casts).
    fn m_substitute_block(block: &mut Block, params: &[String], args: &[Type]) {
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Let { type_ann, init, .. } => {
                    if let Some(ty) = type_ann {
                        Self::substitute_type_params(ty, params, args);
                    }
                    let mut expr_box = Box::new(init.clone());
                    Self::m_substitute_types_in_expr(&mut expr_box, params, args);
                    *init = *expr_box;
                }
                Stmt::Const { type_ann, init, .. } => {
                    if let Some(ty) = type_ann {
                        Self::substitute_type_params(ty, params, args);
                    }
                    let mut expr_box = Box::new(init.clone());
                    Self::m_substitute_types_in_expr(&mut expr_box, params, args);
                    *init = *expr_box;
                }
                Stmt::Expr(expr) => {
                    let mut expr_box = Box::new(expr.clone());
                    Self::m_substitute_types_in_expr(&mut expr_box, params, args);
                    *expr = *expr_box;
                }
                Stmt::Return { value, .. } => {
                    if let Some(inner) = value {
                        let mut expr_box = Box::new((**inner).clone());
                        Self::m_substitute_types_in_expr(&mut expr_box, params, args);
                        **inner = *expr_box;
                    }
                }
            }
        }
        if let Some(tail) = &mut block.tail_expr {
            let mut expr_box = Box::new((**tail).clone());
            Self::m_substitute_types_in_expr(&mut expr_box, params, args);
            **tail = *expr_box;
        }
    }

    /// Substitute const generic identifiers in an expression with literal values.
    /// E.g., `Expr::Ident("L")` → `Expr::IntLit(5)` when `const_params = ["L"], values = [5]`.
    fn m_substitute_const_in_expr(expr: &mut Expr, const_params: &[String], values: &[i64]) {
        match expr {
            Expr::Ident(name) => {
                if let Some(pos) = const_params.iter().position(|p| p == name)
                    && let Some(&val) = values.get(pos)
                {
                    *expr = Expr::IntLit(val);
                }
            }
            Expr::Binary { lhs, rhs, .. } => {
                Self::m_substitute_const_in_expr(lhs, const_params, values);
                Self::m_substitute_const_in_expr(rhs, const_params, values);
            }
            Expr::UnaryNot(inner) | Expr::UnaryMinus(inner) => {
                Self::m_substitute_const_in_expr(inner, const_params, values);
            }
            Expr::Cast { expr: inner, .. } => {
                Self::m_substitute_const_in_expr(inner, const_params, values);
            }
            Expr::Call { args, .. } | Expr::QualifiedCall { args, .. } => {
                for arg in args.iter_mut() {
                    Self::m_substitute_const_in_expr(arg, const_params, values);
                }
            }
            Expr::MethodCall {
                expr: receiver,
                args,
                ..
            } => {
                Self::m_substitute_const_in_expr(receiver, const_params, values);
                for arg in args.iter_mut() {
                    Self::m_substitute_const_in_expr(arg, const_params, values);
                }
            }
            Expr::Assign { target, value } => {
                Self::m_substitute_const_in_expr(target, const_params, values);
                Self::m_substitute_const_in_expr(value, const_params, values);
            }
            Expr::Ref { expr: inner, .. } | Expr::Deref(inner) => {
                Self::m_substitute_const_in_expr(inner, const_params, values);
            }
            Expr::Member { expr: inner, .. } => {
                Self::m_substitute_const_in_expr(inner, const_params, values);
            }
            Expr::Index { array, index } => {
                Self::m_substitute_const_in_expr(array, const_params, values);
                Self::m_substitute_const_in_expr(index, const_params, values);
            }
            Expr::Array(elems) => {
                for elem in elems.iter_mut() {
                    Self::m_substitute_const_in_expr(elem, const_params, values);
                }
            }
            Expr::Repeat(inner, _) => {
                Self::m_substitute_const_in_expr(inner, const_params, values);
            }
            Expr::If {
                cond,
                then_block,
                else_ifs,
                else_block,
            } => {
                Self::m_substitute_const_in_expr(cond, const_params, values);
                Self::m_substitute_const_block(then_block, const_params, values);
                for (cond, block) in else_ifs.iter_mut() {
                    Self::m_substitute_const_in_expr(cond, const_params, values);
                    Self::m_substitute_const_block(block, const_params, values);
                }
                if let Some(block) = else_block {
                    Self::m_substitute_const_block(block, const_params, values);
                }
            }
            Expr::Loop { body } | Expr::While { body, .. } => {
                Self::m_substitute_const_block(body, const_params, values);
            }
            Expr::IfLet {
                then_block,
                else_block,
                ..
            } => {
                Self::m_substitute_const_block(then_block, const_params, values);
                if let Some(block) = else_block {
                    Self::m_substitute_const_block(block, const_params, values);
                }
            }
            Expr::Match {
                scrutinee, arms, ..
            } => {
                Self::m_substitute_const_in_expr(scrutinee, const_params, values);
                for arm in arms.iter_mut() {
                    if let Some(ref mut guard) = arm.guard {
                        Self::m_substitute_const_in_expr(guard, const_params, values);
                    }
                    Self::m_substitute_const_block(&mut arm.body, const_params, values);
                }
            }
            // Tuple, StructLit, EnumLit, BoolLit, IntLit, FloatLit, StrLit, Unit — no substitution needed
            _ => {}
        }
    }

    /// Substitute const generic identifiers in a block.
    fn m_substitute_const_block(block: &mut Block, const_params: &[String], values: &[i64]) {
        for stmt in &mut block.stmts {
            match stmt {
                Stmt::Let { init, .. } | Stmt::Const { init, .. } => {
                    Self::m_substitute_const_in_expr(init, const_params, values);
                }
                Stmt::Expr(expr) => {
                    Self::m_substitute_const_in_expr(expr, const_params, values);
                }
                Stmt::Return { value, .. } => {
                    if let Some(inner) = value {
                        Self::m_substitute_const_in_expr(inner, const_params, values);
                    }
                }
            }
        }
        if let Some(tail) = &mut block.tail_expr {
            Self::m_substitute_const_in_expr(tail, const_params, values);
        }
    }

    /// Substitute type params in a Type value.
    fn substitute_type_params(ty: &mut Type, params: &[String], args: &[Type]) {
        match ty {
            Type::Struct(name) => {
                if let Some(pos) = params.iter().position(|p| p == name)
                    && let Some(arg) = args.get(pos)
                {
                    *ty = arg.clone();
                }
            }
            Type::Ref { inner, .. }
            | Type::Ptr { inner, .. }
            | Type::Array { inner, .. }
            | Type::GenericArray { inner, .. } => {
                Self::substitute_type_params(inner, params, args);
            }
            Type::Tuple(elems) => {
                for elem in elems.iter_mut() {
                    Self::substitute_type_params(elem, params, args);
                }
            }
            Type::GenericInstance(_name, gen_args) => {
                // Substitute in the args
                for arg in gen_args.iter_mut() {
                    Self::substitute_type_params(arg, params, args);
                }
            }
            Type::Alias(_, alias_args) => {
                for arg in alias_args.iter_mut() {
                    Self::substitute_type_params(arg, params, args);
                }
            }
            _ => {}
        }
    }

    /// Collect GenericInstance from a method's params, return type, and body.
    fn collect_generic_instances_from_method(func: &Function) -> Vec<(String, Vec<Type>)> {
        let mut instances = Vec::new();
        let mut seen = HashSet::new();

        for param in &func.params {
            Self::collect_from_types(std::slice::from_ref(&param.ty), &mut instances, &mut seen);
        }
        if let Some(ref ret) = func.return_type {
            Self::collect_from_types(std::slice::from_ref(ret), &mut instances, &mut seen);
        }
        Self::collect_from_block(&func.body, &mut instances, &mut seen);
        let mut one_param_enums = HashSet::new();
        one_param_enums.insert("Option".to_string());
        Self::collect_enum_lits_from_block(&func.body, &mut instances, &mut seen, &one_param_enums);

        instances
    }

    /// Collect GenericInstance from a block's expressions.
    fn collect_from_block(
        block: &Block,
        instances: &mut Vec<(String, Vec<Type>)>,
        seen: &mut HashSet<String>,
    ) {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let {
                    type_ann: Some(ty), ..
                }
                | Stmt::Const {
                    type_ann: Some(ty), ..
                } => {
                    Self::collect_from_types(std::slice::from_ref(ty), instances, seen);
                }
                _ => {}
            }
        }
    }

    /// Collect GenericInstance from a slice of types.
    fn collect_from_types(
        types: &[Type],
        instances: &mut Vec<(String, Vec<Type>)>,
        seen: &mut HashSet<String>,
    ) {
        for ty in types {
            match ty {
                Type::GenericInstance(name, args) => {
                    // Skip instances with unresolved type params
                    if args.iter().any(Self::is_type_param) {
                        continue;
                    }
                    let key = Self::mangle_generic_instance(name, args);
                    if seen.insert(key) {
                        instances.push((name.clone(), args.clone()));
                    }
                }
                Type::Ref { inner, .. }
                | Type::Ptr { inner, .. }
                | Type::Array { inner, .. }
                | Type::GenericArray { inner, .. }
                | Type::Slice { inner, .. } => {
                    Self::collect_from_types(&[inner.as_ref().clone()], instances, seen);
                }
                Type::Tuple(elems) => {
                    Self::collect_from_types(elems, instances, seen);
                }
                _ => {}
            }
        }
    }

    /// Collect all Type::GenericInstance references from a program.
    fn collect_generic_instances(program: &Program) -> Vec<(String, Vec<Type>)> {
        let mut instances = Vec::new();
        let mut seen = HashSet::new();

        // Scan functions
        for func in &program.funcs {
            Self::collect_from_types(
                &func.params.iter().map(|p| p.ty.clone()).collect::<Vec<_>>(),
                &mut instances,
                &mut seen,
            );
            if let Some(ref ret) = func.return_type {
                Self::collect_from_types(std::slice::from_ref(ret), &mut instances, &mut seen);
            }
            Self::collect_from_block(&func.body, &mut instances, &mut seen);
        }

        let one_param_enums: HashSet<String> = program
            .enums
            .iter()
            .filter(|e| e.type_params.len() == 1)
            .map(|e| e.name.clone())
            .collect();

        // Scan for generic enum literals (e.g., Option::Some(42) needs Option<i32>)
        for func in &program.funcs {
            Self::collect_enum_lits_from_block(
                &func.body,
                &mut instances,
                &mut seen,
                &one_param_enums,
            );
        }

        // Scan structs
        for decl in &program.structs {
            let field_types: Vec<Type> = decl.fields.iter().map(|f| f.ty.clone()).collect();
            Self::collect_from_types(&field_types, &mut instances, &mut seen);
        }

        // Scan enums
        for decl in &program.enums {
            let variant_types: Vec<Type> =
                decl.variants.iter().filter_map(|v| v.ty.clone()).collect();
            Self::collect_from_types(&variant_types, &mut instances, &mut seen);
        }

        // Scan impls
        for decl in &program.impls {
            Self::collect_from_types(
                std::slice::from_ref(&decl.impl_type),
                &mut instances,
                &mut seen,
            );
            for method in &decl.methods {
                Self::collect_from_types(
                    &method
                        .params
                        .iter()
                        .map(|p| p.ty.clone())
                        .collect::<Vec<_>>(),
                    &mut instances,
                    &mut seen,
                );
                if let Some(ref ret) = method.return_type {
                    Self::collect_from_types(std::slice::from_ref(ret), &mut instances, &mut seen);
                }
                Self::collect_from_block(&method.body, &mut instances, &mut seen);
            }
        }

        instances
    }

    /// Monomorphize a generic method by substituting type params and resolving SelfType.
    fn monomorphize_method(
        func: &mut Function,
        impl_params: &[String],
        args: &[Type],
        self_type: &Type,
    ) {
        for param in &mut func.params {
            Self::substitute_type_params(&mut param.ty, impl_params, args);
        }
        if let Some(ref mut ret_ty) = func.return_type {
            Self::substitute_type_params(ret_ty, impl_params, args);
        }
        Self::resolve_self_type(func, self_type);
        Self::m_substitute_block(&mut func.body, impl_params, args);
    }

    /// Scan a block for generic enum literals that need monomorphization.
    fn collect_enum_lits_from_block(
        block: &Block,
        instances: &mut Vec<(String, Vec<Type>)>,
        seen: &mut HashSet<String>,
        one_param_enums: &HashSet<String>,
    ) {
        for stmt in &block.stmts {
            match stmt {
                Stmt::Let { init, .. } | Stmt::Const { init, .. } => {
                    Self::collect_enum_lits_from_expr(init, instances, seen, one_param_enums);
                }
                Stmt::Expr(expr) => {
                    Self::collect_enum_lits_from_expr(expr, instances, seen, one_param_enums);
                }
                Stmt::Return { value, .. } => {
                    if let Some(expr) = value {
                        Self::collect_enum_lits_from_expr(expr, instances, seen, one_param_enums);
                    }
                }
            }
        }
        if let Some(ref tail) = block.tail_expr {
            Self::collect_enum_lits_from_expr(tail, instances, seen, one_param_enums);
        }
    }

    /// Recursively scan an expression for generic enum literals.
    fn collect_enum_lits_from_expr(
        expr: &Expr,
        instances: &mut Vec<(String, Vec<Type>)>,
        seen: &mut HashSet<String>,
        one_param_enums: &HashSet<String>,
    ) {
        match expr {
            Expr::EnumLit {
                enum_name, payload, ..
            } => {
                if one_param_enums.contains(enum_name)
                    && let Some(payload_expr) = payload
                {
                    let payload_ty = Self::literal_type(payload_expr);
                    // Only for primitive payloads (skip type params like T)
                    if Self::is_concrete_type(&payload_ty) {
                        let key = Self::mangle_generic_instance(
                            enum_name,
                            std::slice::from_ref(&payload_ty),
                        );
                        if seen.insert(key) {
                            instances.push((enum_name.clone(), vec![payload_ty]));
                        }
                    }
                }
            }
            Expr::If {
                cond,
                then_block,
                else_ifs,
                else_block,
            } => {
                Self::collect_enum_lits_from_expr(cond, instances, seen, one_param_enums);
                Self::collect_enum_lits_from_block(then_block, instances, seen, one_param_enums);
                for (econd, eblock) in else_ifs {
                    Self::collect_enum_lits_from_expr(econd, instances, seen, one_param_enums);
                    Self::collect_enum_lits_from_block(eblock, instances, seen, one_param_enums);
                }
                if let Some(eb) = else_block {
                    Self::collect_enum_lits_from_block(eb, instances, seen, one_param_enums);
                }
            }
            Expr::IfLet {
                scrutinee,
                then_block,
                else_block,
                ..
            } => {
                Self::collect_enum_lits_from_expr(scrutinee, instances, seen, one_param_enums);
                Self::collect_enum_lits_from_block(then_block, instances, seen, one_param_enums);
                if let Some(eb) = else_block {
                    Self::collect_enum_lits_from_block(eb, instances, seen, one_param_enums);
                }
            }
            Expr::Match { scrutinee, arms } => {
                Self::collect_enum_lits_from_expr(scrutinee, instances, seen, one_param_enums);
                for arm in arms {
                    if let Some(guard) = &arm.guard {
                        Self::collect_enum_lits_from_expr(guard, instances, seen, one_param_enums);
                    }
                    Self::collect_enum_lits_from_block(&arm.body, instances, seen, one_param_enums);
                }
            }
            _ => {}
        }
    }

    /// Check if a type is concrete (not a type parameter).
    fn is_concrete_type(ty: &Type) -> bool {
        matches!(
            ty,
            Type::Bool
                | Type::I8
                | Type::I16
                | Type::I32
                | Type::I64
                | Type::U8
                | Type::U16
                | Type::U32
                | Type::U64
                | Type::Usize
                | Type::Isize
                | Type::F32
                | Type::F64
                | Type::Str
        )
    }

    /// Ensure a generic struct/enum instance is monomorphized.
    fn ensure_monomorphized(&mut self, base_name: &str, args: &[Type]) -> Result<(), String> {
        let mangled = Self::mangle_generic_instance(base_name, args);
        if self.monomorphized.contains(&mangled) {
            return Ok(());
        }
        self.monomorphized.insert(mangled.clone());
        self.monomorphized_names
            .insert(base_name.to_string(), mangled.clone());

        // 1a. Create concrete LLVM struct type (generic struct)
        if let Some(decl) = self.generic_struct_defs.get(base_name) {
            let param_names: Vec<String> = decl.type_params.clone();
            let substituted_fields: Vec<StructField> = decl
                .fields
                .iter()
                .map(|f| {
                    let mut ty = f.ty.clone();
                    Self::substitute_type_params(&mut ty, &param_names, args);
                    StructField {
                        name: f.name.clone(),
                        ty,
                        is_pub: f.is_pub,
                        span: f.span,
                    }
                })
                .collect();

            let struct_type = self.context.opaque_struct_type(&mangled);
            let field_types: Vec<BasicTypeEnum<'ctx>> = substituted_fields
                .iter()
                .map(|f| self.type_to_llvm(&f.ty))
                .collect();
            struct_type.set_body(&field_types, false);
            self.struct_fields
                .insert(mangled.clone(), substituted_fields);
            self.struct_types.insert(mangled.clone(), struct_type);

            // Inherit visibility for monomorphized struct
            if let Some(base_decl) = self.generic_struct_defs.get(base_name) {
                self.visibility_map
                    .insert(mangled.clone(), base_decl.is_pub);
                let mut field_map = HashMap::new();
                for field in &base_decl.fields {
                    field_map.insert(field.name.clone(), field.is_pub);
                }
                self.field_visibility_map.insert(mangled.clone(), field_map);
            }
        }

        // 1b. Create concrete LLVM struct type (generic enum)
        if let Some(decl) = self.generic_enum_defs.get(base_name) {
            let param_names: Vec<String> = decl.type_params.clone();
            let mut substituted_fields: Vec<StructField> = vec![StructField {
                name: "__tag".to_string(),
                ty: Type::I8,
                is_pub: false,
                span: Span::empty(0),
            }];
            for variant in &decl.variants {
                if let Some(ref payload_ty) = variant.ty {
                    let mut ty = payload_ty.clone();
                    Self::substitute_type_params(&mut ty, &param_names, args);
                    substituted_fields.push(StructField {
                        name: format!("__{}", variant.name),
                        ty,
                        is_pub: false,
                        span: Span::empty(0),
                    });
                }
            }
            let struct_type = self.context.opaque_struct_type(&mangled);
            let field_types: Vec<BasicTypeEnum<'ctx>> = substituted_fields
                .iter()
                .map(|f| self.type_to_llvm(&f.ty))
                .collect();
            struct_type.set_body(&field_types, false);
            self.struct_fields
                .insert(mangled.clone(), substituted_fields);
            self.struct_types.insert(mangled.clone(), struct_type);

            // Inherit visibility for monomorphized enum
            if let Some(base_decl) = self.generic_enum_defs.get(base_name) {
                self.visibility_map
                    .insert(mangled.clone(), base_decl.is_pub);
            }

            // Register concrete enum in enum_defs for variant lookup
            let mut concrete_decl = decl.clone();
            concrete_decl.name = mangled.clone();
            for variant in &mut concrete_decl.variants {
                if let Some(ref mut payload_ty) = variant.ty {
                    Self::substitute_type_params(payload_ty, &param_names, args);
                }
            }
            self.enum_defs.insert(mangled.clone(), concrete_decl);
        }

        // 2. Compile generic impl methods for this concrete instance
        let impl_blocks: Vec<(Vec<String>, ImplDecl)> = self
            .generic_impls
            .get(base_name)
            .cloned()
            .unwrap_or_default();
        let self_type = Type::GenericInstance(base_name.to_string(), args.to_vec());

        self.current_monomorphization = Some((base_name.to_string(), mangled.clone()));

        for (impl_params, impl_decl) in &impl_blocks {
            for method in &impl_decl.methods {
                let mut method_func = method.clone();

                // Substitute type params in the method
                Self::monomorphize_method(&mut method_func, impl_params, args, &self_type);

                // Compute mangled name
                let mangled_method = format!(
                    "{}::{}/{}",
                    mangled,
                    method_func.name,
                    method_func.params.len()
                );
                method_func.name = mangled_method.clone();

                // Recursively monomorphize any GenericInstance types
                let body_instances = Self::collect_generic_instances_from_method(&method_func);
                // Save our monomorphized_names entry — recursive calls may overwrite it
                let saved_name = self.monomorphized_names.get(base_name).cloned();
                for (sub_base, sub_args) in &body_instances {
                    if self.generic_struct_defs.contains_key(sub_base)
                        || self.generic_enum_defs.contains_key(sub_base)
                    {
                        self.ensure_monomorphized(sub_base, sub_args)?;
                    }
                }
                // Restore our own monomorphized_names entry
                if let Some(ref saved) = saved_name {
                    self.monomorphized_names
                        .insert(base_name.to_string(), saved.clone());
                }

                // Declare and compile
                self.declare_function(&method_func)?;
                self.compile_function_body(&method_func)?;

                // Register in impl_methods
                self.impl_methods
                    .entry(mangled.clone())
                    .or_default()
                    .push((method.name.clone(), mangled_method));
            }
        }

        self.current_monomorphization = None;
        Ok(())
    }

    pub fn compile_module(&mut self, program: &Program) -> Result<(), String> {
        // Populate visibility map
        for func in &program.funcs {
            self.visibility_map.insert(func.name.clone(), func.is_pub);
        }
        for decl in &program.structs {
            self.visibility_map.insert(decl.name.clone(), decl.is_pub);
            let mut field_map = HashMap::new();
            for field in &decl.fields {
                field_map.insert(field.name.clone(), field.is_pub);
            }
            self.field_visibility_map
                .insert(decl.name.clone(), field_map);
        }
        for decl in &program.enums {
            self.visibility_map.insert(decl.name.clone(), decl.is_pub);
        }
        for decl in &program.traits {
            self.visibility_map.insert(decl.name.clone(), decl.is_pub);
        }
        for decl in &program.type_aliases {
            self.visibility_map.insert(decl.name.clone(), decl.is_pub);
        }
        for decl in &program.impls {
            let type_name = match &decl.impl_type {
                Type::Struct(name) => name.clone(),
                Type::GenericInstance(name, _) => name.clone(),
                _ => continue,
            };
            for method in &decl.methods {
                let mangled_name = if let Some(ref trait_name) = decl.trait_name {
                    format!("__trait_{}_{}_{}", trait_name, method.name, type_name)
                } else {
                    format!("{}::{}/{}", type_name, method.name, method.params.len())
                };
                self.visibility_map.insert(mangled_name, method.is_pub);
            }
        }

        // Phase 0: Create opaque struct types for ALL structs/enums
        for decl in &program.structs {
            if !decl.type_params.is_empty() {
                self.generic_struct_defs
                    .insert(decl.name.clone(), decl.clone());
                self.impl_methods.entry(decl.name.clone()).or_default();
            }
            let struct_type = self.context.opaque_struct_type(&decl.name);
            self.struct_fields
                .insert(decl.name.clone(), decl.fields.clone());
            self.struct_types.insert(decl.name.clone(), struct_type);
        }
        for decl in &program.enums {
            if !decl.type_params.is_empty() {
                self.generic_enum_defs
                    .insert(decl.name.clone(), decl.clone());
                self.impl_methods.entry(decl.name.clone()).or_default();
            }
            let mut fields: Vec<StructField> = vec![StructField {
                name: "__tag".to_string(),
                ty: Type::I8,
                is_pub: false,
                span: Span::empty(0),
            }];
            for variant in &decl.variants {
                if let Some(ref payload_ty) = variant.ty {
                    fields.push(StructField {
                        name: format!("__{}", variant.name),
                        ty: payload_ty.clone(),
                        is_pub: false,
                        span: Span::empty(0),
                    });
                }
            }
            self.enum_defs.insert(decl.name.clone(), decl.clone());
            let struct_type = self.context.opaque_struct_type(&decl.name);
            self.struct_fields.insert(decl.name.clone(), fields);
            self.struct_types.insert(decl.name.clone(), struct_type);
        }
        // Phase 0.25: Generate builtin trait implementations for primitives
        self.generate_primitive_trait_impls()?;
        // Phase 0.5: Separate generic impls, process non-generic ones
        for decl in &program.impls {
            let type_name = match &decl.impl_type {
                Type::Struct(name) => name.clone(),
                Type::GenericInstance(name, _) => name.clone(),
                Type::SelfType => "Self".to_string(),
                Type::Slice { .. } | Type::GenericArray { .. } => {
                    // Slice/GenericArray impls are always generic; store for on-demand monomorphization
                    self.slice_impls.push(decl.clone());
                    continue;
                }
                _ => {
                    return Err(format!(
                        "impl target must be a struct type or Self, got {:?}",
                        decl.impl_type
                    ));
                }
            };
            if !decl.type_params.is_empty() {
                self.generic_impls
                    .entry(type_name)
                    .or_default()
                    .push((decl.type_params.clone(), decl.clone()));
                continue;
            }
            for method in &decl.methods {
                let mangled_name = if let Some(ref trait_name) = decl.trait_name {
                    format!("__trait_{}_{}_{}", trait_name, method.name, type_name)
                } else {
                    format!("{}::{}/{}", type_name, method.name, method.params.len())
                };
                let mut method_func = method.clone();
                method_func.name = mangled_name.clone();
                let self_type = Type::Struct(type_name.clone());
                Self::resolve_self_type(&mut method_func, &self_type);
                self.declare_function(&method_func)?;
                self.impl_methods
                    .entry(type_name.clone())
                    .or_default()
                    .push((method.name.clone(), mangled_name));
            }
        }
        // Phase 1: Declare all functions
        for func in &program.funcs {
            self.declare_function(func)?;
        }
        // Phase 0.6: Discover and monomorphize all needed generic instances
        let generic_instances = Self::collect_generic_instances(program);
        for (base_name, args) in &generic_instances {
            if self.generic_struct_defs.contains_key(base_name)
                || self.generic_enum_defs.contains_key(base_name)
            {
                self.ensure_monomorphized(base_name, args)?;
            }
        }
        // Phase 0.65: Set struct bodies for non-generic structs/enums
        for decl in &program.structs {
            if !decl.type_params.is_empty() {
                continue;
            }
            if let Some(st) = self.struct_types.get(&decl.name) {
                let ft: Vec<BasicTypeEnum<'ctx>> = decl
                    .fields
                    .iter()
                    .map(|f| self.type_to_llvm(&f.ty))
                    .collect();
                st.set_body(&ft, false);
            }
        }
        for decl in &program.enums {
            if !decl.type_params.is_empty() {
                continue;
            }
            if let Some(st) = self.struct_types.get(&decl.name) {
                let fields = self.struct_fields.get(&decl.name).unwrap();
                let ft: Vec<BasicTypeEnum<'ctx>> =
                    fields.iter().map(|f| self.type_to_llvm(&f.ty)).collect();
                st.set_body(&ft, false);
            }
        }
        // Phase 0.75: Process #[derive(...)] (non-generic only)
        for decl in &program.structs {
            if decl.type_params.is_empty() {
                self.process_struct_derives(decl, program)?;
            }
        }
        for decl in &program.enums {
            if decl.type_params.is_empty() {
                let f = self
                    .struct_fields
                    .get(&decl.name)
                    .cloned()
                    .unwrap_or_default();
                let sd = StructDecl {
                    name: decl.name.clone(),
                    fields: f,
                    type_params: vec![],
                    is_pub: decl.is_pub,
                    attribs: decl.attribs.clone(),
                    span: decl.span,
                };
                self.process_struct_derives(&sd, program)?;
            }
        }
        // Phase 2: Compile bodies for non-extern functions
        for func in &program.funcs {
            if !func.is_extern {
                self.compile_function_body(func)?;
            }
        }
        // Phase 2b: Compile non-generic impl method bodies
        for decl in &program.impls {
            if !decl.type_params.is_empty() {
                continue;
            }
            let type_name = match &decl.impl_type {
                Type::Struct(name) => name.clone(),
                Type::SelfType => "Self".to_string(),
                _ => continue,
            };
            let self_type = Type::Struct(type_name.clone());
            for method in &decl.methods {
                let mangled_name = if let Some(ref trait_name) = decl.trait_name {
                    format!("__trait_{}_{}_{}", trait_name, method.name, type_name)
                } else {
                    format!("{}::{}/{}", type_name, method.name, method.params.len())
                };
                let mut method_func = method.clone();
                method_func.name = mangled_name;
                Self::resolve_self_type(&mut method_func, &self_type);
                self.compile_function_body(&method_func)?;
            }
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
        let triple = TargetMachine::get_default_triple();
        self.compile_to_object_for_triple(&triple, path)
    }

    pub fn compile_to_object_for_triple(
        &self,
        triple: &TargetTriple,
        path: &Path,
    ) -> Result<(), String> {
        Target::initialize_all(&InitializationConfig::default());

        let target = Target::from_triple(triple)
            .map_err(|e| format!("failed to get target for triple: {}", e))?;
        let machine = target
            .create_target_machine(
                triple,
                "",
                "",
                self.opt_level,
                RelocMode::PIC,
                CodeModel::Default,
            )
            .ok_or_else(|| format!("failed to create target machine for '{}'", triple))?;

        let _ = self
            .module
            .print_to_file("/home/keneeeeert/.gemini/antigravity/scratch/debug.ll");

        machine
            .write_to_file(&self.module, FileType::Object, path)
            .map_err(|e| format!("failed to write object file: {}", e))?;
        Ok(())
    }

    pub fn link_executable(args: &[&str], obj_path: &Path, exe_path: &Path) -> Result<(), String> {
        let output = std::process::Command::new(args[0])
            .args(&args[1..])
            .arg(obj_path)
            .arg("-o")
            .arg(exe_path)
            .output()
            .map_err(|e| format!("failed to run '{}': {}", args[0], e))?;

        if !output.status.success() {
            let stderr = String::from_utf8_lossy(&output.stderr);
            return Err(format!("linker failed: {}", stderr));
        }
        Ok(())
    }

    fn type_to_llvm(&self, ty: &Type) -> BasicTypeEnum<'ctx> {
        match ty {
            Type::Bool => self.bool_type.into(),
            Type::I8 => self.i8_type.into(),
            Type::I16 => self.i16_type.into(),
            Type::I32 => self.i32_type.into(),
            Type::I64 => self.i64_type.into(),
            Type::U8 => self.i8_type.into(),
            Type::U16 => self.i16_type.into(),
            Type::U32 => self.i32_type.into(),
            Type::U64 => self.i64_type.into(),
            Type::Isize => self.ptr_int_type.into(),
            Type::Usize => self.ptr_int_type.into(),
            Type::F32 => self.f32_type.into(),
            Type::F64 => self.f64_type.into(),
            Type::Never => {
                panic!("Never type cannot be used as a value type");
            }
            Type::Tuple(elems) => {
                let llvm_elems: Vec<BasicTypeEnum<'ctx>> =
                    elems.iter().map(|t| self.type_to_llvm(t)).collect();
                self.context.struct_type(&llvm_elems, false).into()
            }
            Type::Unit => self.context.struct_type(&[], false).into(),
            Type::Str => {
                let elems: [BasicTypeEnum; 2] = [
                    self.context.ptr_type(AddressSpace::default()).into(),
                    self.i64_type.into(),
                ];
                self.context.struct_type(&elems, false).into()
            }
            Type::Ref { inner, .. } if **inner == Type::Str => {
                let elems: [BasicTypeEnum; 2] = [
                    self.context.ptr_type(AddressSpace::default()).into(),
                    self.i64_type.into(),
                ];
                self.context.struct_type(&elems, false).into()
            }
            Type::Ref { .. } => self.context.ptr_type(AddressSpace::default()).into(),
            Type::Ptr { .. } => self.context.ptr_type(AddressSpace::default()).into(),
            Type::Array { inner, len } => {
                let elem_ty = self.type_to_llvm(inner);
                let arr_ty = match elem_ty {
                    BasicTypeEnum::IntType(it) => it.array_type(*len as u32),
                    BasicTypeEnum::FloatType(ft) => ft.array_type(*len as u32),
                    BasicTypeEnum::StructType(st) => st.array_type(*len as u32),
                    BasicTypeEnum::ArrayType(at) => at.array_type(*len as u32),
                    _ => panic!("unsupported array element type: {:?}", elem_ty),
                };
                arr_ty.into()
            }
            Type::Struct(name) => self
                .struct_types
                .get(name)
                .copied()
                .unwrap_or_else(|| {
                    panic!(
                        "unknown struct type '{}' — known types: {:?}",
                        name,
                        self.struct_types.keys().collect::<Vec<_>>()
                    )
                })
                .into(),
            Type::GenericInstance(name, args) => {
                let mangled = Self::mangle_generic_instance(name, args);
                self.struct_types
                    .get(&mangled)
                    .copied()
                    .unwrap_or_else(|| {
                        self.struct_types.get(name).copied().unwrap_or_else(|| {
                            panic!(
                                "unknown generic struct instance '{}' — known types: {:?}",
                                mangled,
                                self.struct_types.keys().collect::<Vec<_>>()
                            )
                        })
                    })
                    .into()
            }
            Type::Alias(_, _) => {
                panic!("Type::Alias should have been resolved before codegen");
            }
            Type::SelfType => {
                panic!("SelfType used outside of impl context");
            }
            Type::Slice { .. } => {
                panic!("Slice type cannot be used as a value type (unsized)");
            }
            Type::GenericArray { .. } => {
                panic!("GenericArray type must be resolved to Array before codegen");
            }
        }
    }

    fn type_to_metadata_type(&self, ty: &Type) -> BasicMetadataTypeEnum<'ctx> {
        match ty {
            Type::Ref { .. } => self.type_to_llvm(ty).into(),
            Type::Ptr { .. } => self.context.ptr_type(AddressSpace::default()).into(),
            Type::Str => self.type_to_llvm(ty).into(),
            _ => self.type_to_llvm(ty).into(),
        }
    }

    fn is_signed(ty: &Type) -> bool {
        matches!(
            ty,
            Type::I8 | Type::I16 | Type::I32 | Type::I64 | Type::Isize
        )
    }

    #[allow(dead_code)]
    fn is_bool(ty: &Type) -> bool {
        matches!(ty, Type::Bool)
    }

    fn is_float(ty: &Type) -> bool {
        matches!(ty, Type::F32 | Type::F64)
    }

    /// Compute the type of an expression, using the symbol table for variable lookups.
    fn expr_type(&self, expr: &Expr) -> Type {
        match expr {
            Expr::MethodCall {
                expr: receiver,
                method,
                ..
            } => {
                // Try to determine return type from method definition
                let receiver_type = self.expr_type(receiver);
                if let Type::Ref { inner, .. } = &receiver_type
                    && **inner == Type::Str
                    && method == "len"
                {
                    return Type::Usize;
                }
                // For struct/primitive methods, look up return type from impl methods
                let type_name = match &receiver_type {
                    Type::Array { inner, len } => Some(Self::array_type_key(inner, *len)),
                    Type::Struct(name) => Some(name.clone()),
                    Type::GenericInstance(name, args) => {
                        Some(Self::mangle_generic_instance(name, args))
                    }
                    Type::Ref { inner, .. } => match inner.as_ref() {
                        Type::Array {
                            inner: arr_inner,
                            len,
                        } => Some(Self::array_type_key(arr_inner, *len)),
                        Type::Struct(name) => Some(name.clone()),
                        Type::GenericInstance(name, args) => {
                            Some(Self::mangle_generic_instance(name, args))
                        }
                        _ => Self::primitive_type_name(inner).map(|s| s.to_string()),
                    },
                    _ => Self::primitive_type_name(&receiver_type).map(|s| s.to_string()),
                };
                if let Some(type_name) = type_name
                    && let Some(methods) = self.impl_methods.get(&type_name)
                {
                    if let Some((_, mangled)) = methods.iter().find(|(name, _)| name == method)
                        && let Some(ret_ty) = self.fn_return_types.get(mangled)
                    {
                        return ret_ty.clone();
                    }
                    // clone/default return Self, eq/ne return bool, cmp returns i32
                    if method == "clone" || method == "default" {
                        return receiver_type;
                    }
                    if method == "eq" || method == "ne" {
                        return Type::Bool;
                    }
                    if methods.iter().any(|(name, _)| name == method) {
                        return Type::I32;
                    }
                }
                Self::literal_type(expr)
            }
            Expr::Ident(name) => {
                if self.consts.contains_key(name) {
                    return Type::I32;
                }
                if let Some((_, _, ty)) = self.symbols.get(name) {
                    return ty.clone();
                }
                Type::I32
            }
            Expr::Member {
                expr: inner,
                index,
                field,
            } => {
                let parent_ty = self.expr_type(inner);
                // Resolve struct types to their inner type for field access
                let resolved_ty = match &parent_ty {
                    Type::Ref { inner, .. } => inner.as_ref(),
                    Type::Struct(_) => &parent_ty,
                    Type::GenericInstance(_, _) => &parent_ty,
                    _ => &parent_ty,
                };
                match resolved_ty {
                    Type::Tuple(elems) => {
                        if *index < elems.len() {
                            return elems[*index].clone();
                        }
                        Type::I32
                    }
                    Type::Ref {
                        inner: ref_inner, ..
                    } if **ref_inner == Type::Str => {
                        // &str is a fat pointer {ptr, len}
                        match index {
                            0 => Type::Ptr {
                                inner: Box::new(Type::I8),
                                is_mut: false,
                            },
                            1 => Type::Usize,
                            _ => Type::I32,
                        }
                    }
                    Type::Str => {
                        // When resolved_ty is already Str (outer Ref unwound)
                        match index {
                            0 => Type::Ptr {
                                inner: Box::new(Type::I8),
                                is_mut: false,
                            },
                            1 => Type::Usize,
                            _ => Type::I32,
                        }
                    }
                    Type::Struct(name) => {
                        if let Some(field_name) = field
                            && let Some(fields) = self.struct_fields.get(name)
                            && let Some(f) = fields.iter().find(|f| f.name == *field_name)
                        {
                            f.ty.clone()
                        } else if *index < usize::MAX {
                            // Tuple-like access on struct (not typical)
                            if let Some(fields) = self.struct_fields.get(name)
                                && *index < fields.len()
                            {
                                return fields[*index].ty.clone();
                            }
                            Type::I32
                        } else {
                            Type::I32
                        }
                    }
                    Type::GenericInstance(name, args) => {
                        let mangled = Self::mangle_generic_instance(name, args);
                        if let Some(field_name) = field
                            && let Some(fields) = self.struct_fields.get(&mangled)
                            && let Some(f) = fields.iter().find(|f| f.name == *field_name)
                        {
                            f.ty.clone()
                        } else if *index < usize::MAX {
                            if let Some(fields) = self.struct_fields.get(&mangled)
                                && *index < fields.len()
                            {
                                return fields[*index].ty.clone();
                            }
                            Type::I32
                        } else {
                            Type::I32
                        }
                    }
                    _ => Type::I32,
                }
            }
            Expr::If {
                then_block,
                else_block,
                else_ifs,
                ..
            } => {
                // Determine result type from then/else blocks
                then_block
                    .tail_expr
                    .as_ref()
                    .map(|e| self.expr_type(e))
                    .or_else(|| {
                        else_block
                            .as_ref()
                            .and_then(|b| b.tail_expr.as_ref().map(|e| self.expr_type(e)))
                    })
                    .or_else(|| {
                        else_ifs
                            .first()
                            .and_then(|(_, b)| b.tail_expr.as_ref().map(|e| self.expr_type(e)))
                    })
                    .unwrap_or(Type::I32)
            }
            Expr::IfLet {
                then_block,
                else_block,
                ..
            } => {
                // Result type from then/else blocks
                then_block
                    .tail_expr
                    .as_ref()
                    .map(|e| self.expr_type(e))
                    .or_else(|| {
                        else_block
                            .as_ref()
                            .and_then(|b| b.tail_expr.as_ref().map(|e| self.expr_type(e)))
                    })
                    .unwrap_or(Type::Unit)
            }
            Expr::Match { arms, .. } => {
                // Result type from first arm (all arms must have same type)
                arms.first()
                    .and_then(|arm| arm.body.tail_expr.as_ref())
                    .map(|e| self.expr_type(e))
                    .unwrap_or(Type::Unit)
            }
            Expr::Call { callee, args } => {
                // Handle builtin slice intrinsics
                if let Some(ret_ty) = self.slice_intrinsic_return_type(callee, args) {
                    return ret_ty;
                }
                // Handle __builtin_size_of intrinsic
                if callee == "__builtin_size_of" && args.len() == 1 {
                    return Type::Usize;
                }
                let arg_types: Vec<Type> = args.iter().map(|a| self.expr_type(a)).collect();
                // Try direct lookup, then overloaded
                let name = if self.module.get_function(callee).is_some() {
                    callee.clone()
                } else if let Some(mangled) = self.overloads.get(callee).and_then(|overloads| {
                    overloads
                        .iter()
                        .find(|(_, params)| params == &arg_types)
                        .map(|(mangled, _)| mangled.clone())
                }) {
                    mangled
                } else {
                    callee.clone()
                };
                self.fn_return_types
                    .get(&name)
                    .cloned()
                    .unwrap_or(Type::I32)
            }
            Expr::QualifiedCall {
                module,
                callee,
                args,
            } => {
                let qualified_name = format!("{}::{}", module, callee);
                let mangled_name = format!("{}::{}/{}", module, callee, args.len());
                let mut name = if self.module.get_function(&qualified_name).is_some() {
                    qualified_name
                } else if self.module.get_function(&mangled_name).is_some() {
                    mangled_name
                } else {
                    qualified_name
                };
                if let Some(methods) = self.impl_methods.get(module)
                    && let Some((_, mangled)) = methods.iter().find(|(name, _)| name == callee)
                {
                    name = mangled.clone();
                }
                self.fn_return_types
                    .get(&name)
                    .cloned()
                    .unwrap_or(Type::I32)
            }
            Expr::Binary { op, lhs, rhs, .. }
                if !matches!(
                    op,
                    BinOp::Eq
                        | BinOp::Neq
                        | BinOp::Lt
                        | BinOp::Gt
                        | BinOp::Le
                        | BinOp::Ge
                        | BinOp::And
                        | BinOp::Or
                ) =>
            {
                let lhs_ty = self.expr_type(lhs);
                let rhs_ty = self.expr_type(rhs);
                fn type_width(ty: &Type) -> u32 {
                    match ty {
                        Type::Bool => 1,
                        Type::I8 | Type::U8 => 8,
                        Type::I16 | Type::U16 => 16,
                        Type::I32 | Type::U32 => 32,
                        Type::I64 | Type::U64 | Type::Isize | Type::Usize => 64,
                        _ => 32,
                    }
                }
                if type_width(&lhs_ty) >= type_width(&rhs_ty) {
                    lhs_ty
                } else {
                    rhs_ty
                }
            }
            Expr::EnumLit {
                enum_name, payload, ..
            } => {
                let actual_name: String;
                if let Some(payload_expr) = payload {
                    if self.generic_enum_defs.contains_key(enum_name.as_str()) {
                        let payload_ty = self.expr_type(payload_expr);
                        if Self::is_concrete_type(&payload_ty) {
                            let mangled = Self::mangle_generic_instance(enum_name, &[payload_ty]);
                            if self.struct_types.contains_key(&mangled) {
                                actual_name = mangled;
                            } else {
                                actual_name = self
                                    .monomorphized_names
                                    .get(enum_name.as_str())
                                    .cloned()
                                    .unwrap_or(mangled);
                            }
                        } else {
                            actual_name = self
                                .monomorphized_names
                                .get(enum_name.as_str())
                                .cloned()
                                .unwrap_or_else(|| enum_name.clone());
                        }
                    } else {
                        actual_name = enum_name.clone();
                    }
                } else {
                    actual_name = self
                        .monomorphized_names
                        .get(enum_name.as_str())
                        .cloned()
                        .unwrap_or_else(|| enum_name.clone());
                }
                Type::Struct(actual_name)
            }
            Expr::Index { array, .. } => {
                let array_ty = self.expr_type(array);
                match &array_ty {
                    Type::Array { inner, .. } => *inner.clone(),
                    Type::Ref { inner, .. } => match inner.as_ref() {
                        Type::Array { inner: ai, .. } => *ai.clone(),
                        _ => Type::I32,
                    },
                    _ => Type::I32,
                }
            }
            Expr::Array(elems) => {
                if elems.is_empty() {
                    Type::I32
                } else {
                    let elem_ty = self.expr_type(&elems[0]);
                    Type::Array {
                        inner: Box::new(elem_ty),
                        len: elems.len(),
                    }
                }
            }
            Expr::Repeat(expr, count) => Type::Array {
                inner: Box::new(self.expr_type(expr)),
                len: *count,
            },
            Expr::Deref(inner) => {
                let inner_ty = self.expr_type(inner);
                match inner_ty {
                    Type::Ptr { inner: pointee, .. } => *pointee,
                    Type::Ref { inner: pointee, .. } => *pointee,
                    _ => Type::I32,
                }
            }
            Expr::Ref {
                expr: inner,
                is_mut,
            } => Type::Ref {
                inner: Box::new(self.expr_type(inner)),
                is_mut: *is_mut,
            },
            Expr::UnaryNot(inner) => self.expr_type(inner),
            Expr::UnaryMinus(inner) => self.expr_type(inner),
            Expr::Tuple(elems) => Type::Tuple(elems.iter().map(|e| self.expr_type(e)).collect()),
            _ => Self::literal_type(expr),
        }
    }

    fn literal_type(expr: &Expr) -> Type {
        match expr {
            Expr::BoolLit(_) => Type::Bool,
            Expr::IntLit(_) => Type::I32,
            Expr::FloatLit(_) => Type::F64,
            Expr::StrLit(_) => Type::Ref {
                inner: Box::new(Type::Str),
                is_mut: false,
            },
            Expr::Ref { expr, is_mut } => Type::Ref {
                inner: Box::new(Self::literal_type(expr)),
                is_mut: *is_mut,
            },
            Expr::Deref(expr) => {
                let inner_ty = Self::literal_type(expr);
                match inner_ty {
                    Type::Ptr { inner, .. } => *inner,
                    Type::Ref { inner, .. } => *inner,
                    _ => Type::I32,
                }
            }
            Expr::UnaryMinus(expr) => Self::literal_type(expr),
            Expr::UnaryNot(expr) => Self::literal_type(expr),
            Expr::Assign { value, .. } => Self::literal_type(value),
            Expr::Cast { to_type, .. } => to_type.clone(),
            Expr::Tuple(elems) => Type::Tuple(elems.iter().map(Self::literal_type).collect()),
            Expr::Unit => Type::Unit,
            Expr::MethodCall { expr, method, .. } => {
                let receiver_ty = Self::literal_type(expr);
                // clone() returns Self
                if method == "clone" {
                    return receiver_ty;
                }
                // cmp returns i32
                if method == "cmp" {
                    return Type::I32;
                }
                // eq/ne return bool
                if method == "eq" || method == "ne" {
                    return Type::Bool;
                }
                // default() returns Self
                if method == "default" {
                    return receiver_ty;
                }
                Type::Usize
            }
            Expr::StructLit { struct_name, .. } => Type::Struct(struct_name.clone()),
            Expr::EnumLit { enum_name, .. } => Type::Struct(enum_name.clone()),
            Expr::Binary {
                op:
                    BinOp::Eq
                    | BinOp::Neq
                    | BinOp::Lt
                    | BinOp::Gt
                    | BinOp::Le
                    | BinOp::Ge
                    | BinOp::And
                    | BinOp::Or,
                ..
            } => Type::Bool,
            Expr::Array(elems) => {
                if elems.is_empty() {
                    Type::I32 // fallback; empty arrays are rejected by parser
                } else {
                    let elem_ty = Self::literal_type(&elems[0]);
                    Type::Array {
                        inner: Box::new(elem_ty),
                        len: elems.len(),
                    }
                }
            }
            Expr::Repeat(expr, count) => Type::Array {
                inner: Box::new(Self::literal_type(expr)),
                len: *count,
            },
            Expr::Index { array, .. } => {
                let array_ty = Self::literal_type(array);
                match &array_ty {
                    Type::Array { inner, .. } => *inner.clone(),
                    _ => Type::I32,
                }
            }
            _ => Type::I32,
        }
    }

    fn declare_function(&mut self, func: &Function) -> Result<(), String> {
        let param_types: Vec<BasicMetadataTypeEnum<'ctx>> = func
            .params
            .iter()
            .map(|p| self.type_to_metadata_type(&p.ty))
            .collect();
        let is_never = matches!(&func.return_type, Some(Type::Never));

        // Determine if variadic — currently only printf
        let is_variadic = func.name == "printf";

        let use_void_return = is_never || (func.name != "main" && func.return_type.is_none());
        let fn_type = if use_void_return {
            let void_type = self.context.void_type();
            void_type.fn_type(&param_types, is_variadic)
        } else {
            let ret_type: BasicTypeEnum = match &func.return_type {
                Some(ty) => self.type_to_llvm(ty),
                None => self.i32_type.into(),
            };
            if is_variadic {
                ret_type.fn_type(&param_types, true)
            } else {
                ret_type.fn_type(&param_types, false)
            }
        };

        if self.module.get_function(&func.name).is_none() {
            let fn_val = self.module.add_function(&func.name, fn_type, None);
            if is_never {
                let kind_id = Attribute::get_named_enum_kind_id("noreturn");
                let attr = self.context.create_enum_attribute(kind_id, 0);
                fn_val.add_attribute(inkwell::attributes::AttributeLoc::Function, attr);
            }
        }
        // Store return type for expr_type resolution
        let ret_type = func.return_type.clone().unwrap_or(Type::I32);
        self.fn_return_types.insert(func.name.clone(), ret_type);
        // Store parameter types for argument coercion
        let param_tys: Vec<Type> = func.params.iter().map(|p| p.ty.clone()).collect();
        self.fn_param_types.insert(func.name.clone(), param_tys);
        Ok(())
    }

    fn compile_function_body(&mut self, func: &Function) -> Result<(), String> {
        let saved_bb = self.builder.get_insert_block();
        let saved_symbols = self.symbols.clone();
        let saved_module_path = self.current_module_path.clone();

        let result = self.compile_function_body_inner(func);

        if let Some(bb) = saved_bb {
            self.builder.position_at_end(bb);
        }
        self.symbols = saved_symbols;
        self.current_module_path = saved_module_path;

        result
    }

    fn compile_function_body_inner(&mut self, func: &Function) -> Result<(), String> {
        self.current_module_path = self.get_module_path_for_item_name(&func.name);

        // Validate visibility of parameter and return types
        for param in &func.params {
            self.check_visibility_of_type(&self.current_module_path, &param.ty)?;
        }
        if let Some(ref ret) = func.return_type {
            self.check_visibility_of_type(&self.current_module_path, ret)?;
        }

        let fn_val = self
            .module
            .get_function(&func.name)
            .ok_or_else(|| format!("function '{}' not declared", func.name))?;

        let entry = self.context.append_basic_block(fn_val, "entry");
        self.builder.position_at_end(entry);

        // Allocate and store parameters
        let param_values = fn_val.get_params();
        for (i, param) in func.params.iter().enumerate() {
            let llvm_ty = self.type_to_llvm(&param.ty);
            let alloca = self
                .builder
                .build_alloca(llvm_ty, &param.name)
                .map_err(|e| format!("failed to build alloca for param '{}': {}", param.name, e))?;
            self.builder
                .build_store(alloca, param_values[i])
                .map_err(|e| format!("failed to store param '{}': {}", param.name, e))?;
            self.symbols
                .insert(param.name.clone(), (alloca, false, param.ty.clone()));
        }

        for stmt in &func.body.stmts {
            self.compile_stmt(stmt)?;
        }

        // Check if the current block is already terminated (e.g. by a return statement)
        let already_terminated = self
            .builder
            .get_insert_block()
            .is_none_or(|bb| bb.get_terminator().is_some());

        if !already_terminated {
            if let Some(tail_expr) = &func.body.tail_expr {
                let val = self.compile_expr(tail_expr)?;
                self.builder
                    .build_return(Some(&val))
                    .map_err(|e| format!("failed to build return for tail expr: {}", e))?;
            } else {
                let is_never = matches!(&func.return_type, Some(Type::Never));
                if is_never {
                    self.builder
                        .build_unreachable()
                        .map_err(|e| format!("failed to build unreachable: {}", e))?;
                } else if func.name == "main" {
                    let zero_value: BasicValueEnum<'ctx> = self.i32_type.const_zero().into();
                    self.builder
                        .build_return(Some(&zero_value))
                        .map_err(|e| format!("failed to build return: {}", e))?;
                } else if let Some(ret_ty) = &func.return_type {
                    let llvm_ret_ty = self.type_to_llvm(ret_ty);
                    let default_ret: BasicValueEnum<'ctx> = match llvm_ret_ty {
                        BasicTypeEnum::IntType(int_ty) => int_ty.const_zero().into(),
                        BasicTypeEnum::FloatType(float_ty) => float_ty.const_float(0.0).into(),
                        BasicTypeEnum::StructType(struct_ty) => struct_ty.get_undef().into(),
                        _ => self.i32_type.const_zero().into(),
                    };
                    self.builder
                        .build_return(Some(&default_ret))
                        .map_err(|e| format!("failed to build return: {}", e))?;
                } else {
                    self.builder
                        .build_return(None)
                        .map_err(|e| format!("failed to build return: {}", e))?;
                }
            }
        }

        Ok(())
    }

    fn compile_stmt(&mut self, stmt: &Stmt) -> Result<(), String> {
        match stmt {
            Stmt::Let {
                name,
                is_mut,
                type_ann,
                init,
                ..
            } => {
                if let Some(ann_ty) = type_ann {
                    self.check_visibility_of_type(&self.current_module_path, ann_ty)?;
                }
                let ty = type_ann.clone().unwrap_or_else(|| self.expr_type(init));
                let llvm_ty = self.type_to_llvm(&ty);
                let alloca = self
                    .builder
                    .build_alloca(llvm_ty, name)
                    .map_err(|e| format!("failed to build alloca: {}", e))?;
                let mut value = if let Some(ann_ty) = type_ann {
                    self.with_expected_type(ann_ty, |this| this.compile_expr(init))?
                } else {
                    self.compile_expr(init)?
                };
                // Only attempt type coercion when the LLVM types differ,
                // to handle cases like StructLit vs GenericInstance types
                if value.get_type() != llvm_ty
                    && let Some(ann_ty) = type_ann
                {
                    let inferred = self.expr_type(init);
                    if ann_ty != &inferred {
                        value = self.emit_cast(value, ann_ty)?;
                    }
                }
                self.builder
                    .build_store(alloca, value)
                    .map_err(|e| format!("failed to build store: {}", e))?;
                self.symbols.insert(name.clone(), (alloca, *is_mut, ty));
                Ok(())
            }
            Stmt::Const { name, init, .. } => {
                let raw = self.const_eval(init)?;
                self.consts.insert(name.clone(), raw);
                Ok(())
            }
            Stmt::Expr(expr) => {
                self.compile_expr(expr)?;
                Ok(())
            }
            Stmt::Return { value, .. } => {
                match value {
                    Some(expr) => {
                        let val = self.compile_expr(expr)?;
                        self.builder
                            .build_return(Some(&val))
                            .map_err(|e| format!("failed to build return: {}", e))?;
                    }
                    None => {
                        self.builder
                            .build_return(None)
                            .map_err(|e| format!("failed to build return: {}", e))?;
                    }
                }
                // After return, insert unreachable so LLVM knows subsequent code is dead
                self.builder
                    .build_unreachable()
                    .map_err(|e| format!("failed to build unreachable after return: {}", e))?;
                Ok(())
            }
        }
    }

    /// Compile a block's statements and return the value of its tail expression (if any).
    fn compile_block_get_value(
        &mut self,
        block: &Block,
    ) -> Result<Option<BasicValueEnum<'ctx>>, String> {
        for stmt in &block.stmts {
            self.compile_stmt(stmt)?;
        }
        if let Some(tail_expr) = &block.tail_expr {
            let val = self.compile_expr(tail_expr)?;
            Ok(Some(val))
        } else {
            Ok(None)
        }
    }

    /// Compile a comparison operator via trait method calls (Eq::eq, Eq::ne, Ord::cmp).
    fn compile_trait_comparison(
        &mut self,
        op: &BinOp,
        lhs: &Expr,
        rhs: &Expr,
        lhs_val: BasicValueEnum<'ctx>,
        rhs_val: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let lhs_type = self.expr_type(lhs);
        let rhs_type = self.expr_type(rhs);

        // Promote both to the wider type for comparison
        let (promoted_lhs, promoted_rhs, common_type) =
            if let (BasicValueEnum::IntValue(lhs_i), BasicValueEnum::IntValue(rhs_i)) =
                (lhs_val, rhs_val)
            {
                let lhs_w = lhs_i.get_type().get_bit_width();
                let rhs_w = rhs_i.get_type().get_bit_width();
                if lhs_w > rhs_w {
                    let ext_rhs = self
                        .builder
                        .build_int_z_extend(rhs_i, lhs_i.get_type(), "cmp_ext")
                        .map_err(|e| format!("failed to extend cmp rhs: {}", e))?;
                    (lhs_i.into(), ext_rhs.into(), lhs_type.clone())
                } else if rhs_w > lhs_w {
                    let ext_lhs = self
                        .builder
                        .build_int_z_extend(lhs_i, rhs_i.get_type(), "cmp_ext")
                        .map_err(|e| format!("failed to extend cmp lhs: {}", e))?;
                    (ext_lhs.into(), rhs_i.into(), rhs_type.clone())
                } else {
                    (lhs_val, rhs_val, lhs_type.clone())
                }
            } else {
                (lhs_val, rhs_val, lhs_type.clone())
            };

        // Store both values to allocas for passing as references
        let common_llvm_ty = self.type_to_llvm(&common_type);
        let lhs_alloca = self
            .builder
            .build_alloca(common_llvm_ty, "cmp_lhs")
            .map_err(|e| format!("failed to build cmp lhs alloca: {}", e))?;
        self.builder
            .build_store(lhs_alloca, promoted_lhs)
            .map_err(|e| format!("failed to store cmp lhs: {}", e))?;
        let rhs_alloca = self
            .builder
            .build_alloca(common_llvm_ty, "cmp_rhs")
            .map_err(|e| format!("failed to build cmp rhs alloca: {}", e))?;
        self.builder
            .build_store(rhs_alloca, promoted_rhs)
            .map_err(|e| format!("failed to store cmp rhs: {}", e))?;

        // Use the common type for trait method lookup
        let lhs_type = common_type;

        match op {
            BinOp::Eq | BinOp::Neq => {
                let trait_name = "Eq";
                let method_name = match op {
                    BinOp::Eq => "eq",
                    BinOp::Neq => "ne",
                    _ => unreachable!(),
                };
                let fn_name = Self::trait_method_name(&lhs_type, trait_name, method_name);
                let type_name = Self::type_to_mangled_name(&lhs_type);
                let trait_mangled_name =
                    format!("__trait_{}_{}_{}", trait_name, method_name, type_name);
                let fn_val = self
                    .module
                    .get_function(&fn_name)
                    .or_else(|| self.module.get_function(&trait_mangled_name))
                    .ok_or_else(|| {
                        format!(
                            "internal: {}::{} not found for type {:?}",
                            trait_name, method_name, lhs_type
                        )
                    })?;
                let result = self
                    .builder
                    .build_call(
                        fn_val,
                        &[lhs_alloca.into(), rhs_alloca.into()],
                        "trait_eq_call",
                    )
                    .map_err(|e| {
                        format!("failed to call {}::{}: {}", trait_name, method_name, e)
                    })?;
                Ok(self.try_extract_result(result))
            }
            _ => {
                // Lt, Gt, Le, Ge via Ord::cmp
                let fn_name = Self::trait_method_name(&lhs_type, "Ord", "cmp");
                let type_name = Self::type_to_mangled_name(&lhs_type);
                let trait_mangled_name = format!("__trait_Ord_cmp_{}", type_name);
                let fn_val = self
                    .module
                    .get_function(&fn_name)
                    .or_else(|| self.module.get_function(&trait_mangled_name))
                    .ok_or_else(|| {
                        format!("internal: Ord::cmp not found for type {:?}", lhs_type)
                    })?;
                let result = self
                    .builder
                    .build_call(
                        fn_val,
                        &[lhs_alloca.into(), rhs_alloca.into()],
                        "trait_cmp_call",
                    )
                    .map_err(|e| format!("failed to call Ord::cmp: {}", e))?;
                let cmp_result = self.try_extract_result(result).into_int_value();
                let pred = match op {
                    BinOp::Lt => inkwell::IntPredicate::SLT,
                    BinOp::Gt => inkwell::IntPredicate::SGT,
                    BinOp::Le => inkwell::IntPredicate::SLE,
                    BinOp::Ge => inkwell::IntPredicate::SGE,
                    _ => unreachable!(),
                };
                let result = self
                    .builder
                    .build_int_compare(pred, cmp_result, self.i32_type.const_zero(), "cmp_result")
                    .map_err(|e| format!("failed to build cmp compare with zero: {}", e))?;
                Ok(result.into())
            }
        }
    }

    /// Compile an arithmetic operator via trait method calls (Add::add, Sub::sub, Mul::mul, Div::div).
    fn compile_trait_arithmetic(
        &mut self,
        op: &BinOp,
        lhs: &Expr,
        rhs: &Expr,
        lhs_val: BasicValueEnum<'ctx>,
        rhs_val: BasicValueEnum<'ctx>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let lhs_type = self.expr_type(lhs);
        let rhs_type = self.expr_type(rhs);

        // Promote both to the wider type if they are integers
        let (promoted_lhs, promoted_rhs, common_type) =
            if let (BasicValueEnum::IntValue(lhs_i), BasicValueEnum::IntValue(rhs_i)) =
                (lhs_val, rhs_val)
            {
                let lhs_w = lhs_i.get_type().get_bit_width();
                let rhs_w = rhs_i.get_type().get_bit_width();
                if lhs_w > rhs_w {
                    let ext_rhs = self
                        .builder
                        .build_int_z_extend(rhs_i, lhs_i.get_type(), "arith_ext")
                        .map_err(|e| format!("failed to extend arith rhs: {}", e))?;
                    (lhs_i.into(), ext_rhs.into(), lhs_type.clone())
                } else if rhs_w > lhs_w {
                    let ext_lhs = self
                        .builder
                        .build_int_z_extend(lhs_i, rhs_i.get_type(), "arith_ext")
                        .map_err(|e| format!("failed to extend arith lhs: {}", e))?;
                    (ext_lhs.into(), rhs_i.into(), rhs_type.clone())
                } else {
                    (lhs_val, rhs_val, lhs_type.clone())
                }
            } else {
                (lhs_val, rhs_val, lhs_type.clone())
            };

        // Store both values to allocas for passing as references
        let common_llvm_ty = self.type_to_llvm(&common_type);
        let lhs_alloca = self
            .builder
            .build_alloca(common_llvm_ty, "arith_lhs")
            .map_err(|e| format!("failed to build arith lhs alloca: {}", e))?;
        self.builder
            .build_store(lhs_alloca, promoted_lhs)
            .map_err(|e| format!("failed to store arith lhs: {}", e))?;
        let rhs_alloca = self
            .builder
            .build_alloca(common_llvm_ty, "arith_rhs")
            .map_err(|e| format!("failed to build arith rhs alloca: {}", e))?;
        self.builder
            .build_store(rhs_alloca, promoted_rhs)
            .map_err(|e| format!("failed to store arith rhs: {}", e))?;

        let (trait_name, method_name) = match op {
            BinOp::Add => ("Add", "add"),
            BinOp::Sub => ("Sub", "sub"),
            BinOp::Mul => ("Mul", "mul"),
            BinOp::Div => ("Div", "div"),
            _ => return Err(format!("unsupported arithmetic operator {:?}", op)),
        };

        let type_name = Self::type_to_mangled_name(&common_type);
        let builtin_name = format!("__builtin_{}_{}_{}", trait_name, method_name, type_name);
        let trait_name_mangled = format!("__trait_{}_{}_{}", trait_name, method_name, type_name);

        let fn_val = self
            .module
            .get_function(&builtin_name)
            .or_else(|| self.module.get_function(&trait_name_mangled))
            .ok_or_else(|| {
                format!(
                    "type '{:?}' does not implement trait '{}' (method '{}' not found)",
                    common_type, trait_name, method_name
                )
            })?;

        let result = self
            .builder
            .build_call(
                fn_val,
                &[lhs_alloca.into(), rhs_alloca.into()],
                "trait_arith_call",
            )
            .map_err(|e| format!("failed to call {}::{}: {}", trait_name, method_name, e))?;
        Ok(self.try_extract_result(result))
    }

    /// Compile the else-if chain storing results into result_alloca.
    fn compile_else_chain_store(
        &mut self,
        else_ifs: &[(Expr, Block)],
        else_block: &Option<Block>,
        parent_fn: inkwell::values::FunctionValue<'ctx>,
        merge_bb: inkwell::basic_block::BasicBlock<'ctx>,
        result_alloca: inkwell::values::PointerValue<'ctx>,
        _result_type: &Type,
    ) -> Result<(), String> {
        if let Some((elif_cond, elif_body)) = else_ifs.first() {
            let cond_val = self.compile_expr(elif_cond)?;
            let cond_i1 = match cond_val {
                BasicValueEnum::IntValue(v) => {
                    if v.get_type().get_bit_width() != 1 {
                        self.builder
                            .build_int_truncate(v, self.bool_type, "elif_cond_i1")
                            .map_err(|e| format!("failed to trunc elif cond to i1: {}", e))?
                    } else {
                        v
                    }
                }
                _ => {
                    return Err("else-if condition must be a boolean or integer".to_string());
                }
            };

            let then_bb = self.context.append_basic_block(parent_fn, "elif_then");
            let rest_bb = self.context.append_basic_block(parent_fn, "elif_rest");

            self.builder
                .build_conditional_branch(cond_i1, then_bb, rest_bb)
                .map_err(|e| format!("failed to build elif branch: {}", e))?;

            // Compile the elif body
            self.builder.position_at_end(then_bb);
            let saved_symbols = self.symbols.clone();
            if let Some(val) = self.compile_block_get_value(elif_body)? {
                self.builder
                    .build_store(result_alloca, val)
                    .map_err(|e| format!("failed to store elif result: {}", e))?;
            }
            self.symbols = saved_symbols;
            let then_terminated = self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_some();
            if !then_terminated {
                self.builder
                    .build_unconditional_branch(merge_bb)
                    .map_err(|e| format!("failed to branch from elif to merge: {}", e))?;
            }

            // Compile remaining else-ifs / else block
            self.builder.position_at_end(rest_bb);
            self.compile_else_chain_store(
                &else_ifs[1..],
                else_block,
                parent_fn,
                merge_bb,
                result_alloca,
                _result_type,
            )?;
        } else if let Some(el_block) = else_block {
            let saved_symbols = self.symbols.clone();
            if let Some(val) = self.compile_block_get_value(el_block)? {
                self.builder
                    .build_store(result_alloca, val)
                    .map_err(|e| format!("failed to store else result: {}", e))?;
            }
            self.symbols = saved_symbols;
        }
        Ok(())
    }

    /// Get the key used for looking up array methods in impl_methods.
    fn array_type_key(inner: &Type, len: usize) -> String {
        let elem_name = if let Some(prim) = Self::primitive_type_name(inner) {
            prim.to_string()
        } else {
            match inner {
                Type::Struct(name) => name.clone(),
                Type::GenericInstance(name, args) => Self::mangle_generic_instance(name, args),
                _ => format!("{:?}", inner),
            }
        };
        format!("array_{}_{}", elem_name, len)
    }

    /// Ensure builtin slice methods (index, index_mut, as_ptr, len) exist for the given array type.
    /// First processes GenericArray impl blocks from slice_impls (which compile method bodies
    /// calling __builtin_slice_* intrinsics), then falls back to compiler intrinsics for any
    /// remaining methods not provided by the stdlib.
    fn ensure_slice_methods(&mut self, elem_ty: &Type, len: usize) -> Result<(), String> {
        let key = Self::array_type_key(elem_ty, len);
        if self.impl_methods.contains_key(&key) {
            return Ok(()); // already generated
        }
        // Register empty entry to avoid recursion
        self.impl_methods.entry(key.clone()).or_default();
        self.trait_impls
            .entry(key.clone())
            .or_default()
            .extend(["Index".to_string(), "IndexMut".to_string()]);

        let elem_llvm = self.type_to_llvm(elem_ty);
        let array_llvm: BasicTypeEnum<'ctx> = match elem_llvm {
            BasicTypeEnum::IntType(it) => it.array_type(len as u32).into(),
            BasicTypeEnum::FloatType(ft) => ft.array_type(len as u32).into(),
            BasicTypeEnum::StructType(st) => st.array_type(len as u32).into(),
            BasicTypeEnum::ArrayType(at) => at.array_type(len as u32).into(),
            _ => panic!("unsupported array element type: {:?}", elem_llvm),
        };
        let ptr_type = self.context.ptr_type(AddressSpace::default());
        let self_array_type = Type::Array {
            inner: Box::new(elem_ty.clone()),
            len,
        };

        // --- Phase 1: Process GenericArray impl blocks (stdlib bodies calling __builtin_slice_*) ---
        let slice_impls = self.slice_impls.clone();
        for impl_decl in &slice_impls {
            let _len_var = match &impl_decl.impl_type {
                Type::GenericArray { len_var, .. } => len_var.clone(),
                _ => continue,
            };
            let type_params = &impl_decl.type_params;
            let const_params = &impl_decl.const_params;
            let inner_type_name = match &impl_decl.impl_type {
                Type::GenericArray { inner, .. } => match inner.as_ref() {
                    Type::Struct(name) => name.clone(),
                    _ => String::new(),
                },
                _ => String::new(),
            };
            let type_args: Vec<Type> = type_params
                .iter()
                .map(|p| {
                    if p == &inner_type_name {
                        elem_ty.clone()
                    } else {
                        Type::I32
                    }
                })
                .collect();
            let const_values: Vec<i64> = const_params.iter().map(|_| len as i64).collect();

            for method in &impl_decl.methods {
                let mut method_func = method.clone();
                Self::substitute_type_params_in_func(&mut method_func, type_params, &type_args);
                let self_type = self_array_type.clone();
                Self::resolve_self_type(&mut method_func, &self_type);
                Self::m_substitute_const_block(&mut method_func.body, const_params, &const_values);
                Self::m_substitute_block(&mut method_func.body, type_params, &type_args);

                let body_instances = Self::collect_generic_instances_from_method(&method_func);
                for (sub_base, sub_args) in &body_instances {
                    if self.generic_struct_defs.contains_key(sub_base)
                        || self.generic_enum_defs.contains_key(sub_base)
                    {
                        self.ensure_monomorphized(sub_base, sub_args)?;
                    }
                }

                let mangled_name = if let Some(ref trait_name) = impl_decl.trait_name {
                    format!("__trait_{}_{}_{}", trait_name, method.name, key)
                } else {
                    format!("{}::{}/{}", key, method.name, method.params.len())
                };
                method_func.name = mangled_name.clone();

                if self.module.get_function(&mangled_name).is_some() {
                    self.impl_methods
                        .get_mut(&key)
                        .unwrap()
                        .push((method.name.clone(), mangled_name));
                    continue;
                }

                self.declare_function(&method_func)?;
                self.compile_function_body(&method_func)?;

                self.impl_methods
                    .get_mut(&key)
                    .unwrap()
                    .push((method.name.clone(), mangled_name));
            }
        }

        // --- Phase 2: Fallback intrinsic generation for any missing methods ---
        // This ensures array methods work even when stdlib slice.u is not loaded.
        let existing_methods: Vec<String> = self
            .impl_methods
            .get(&key)
            .map(|m| m.iter().map(|(n, _)| n.clone()).collect())
            .unwrap_or_default();

        if !existing_methods.contains(&"len".to_string()) {
            let len_fn_name = format!("{}::len/1", key);
            if self.module.get_function(&len_fn_name).is_none() {
                let param_types: [BasicMetadataTypeEnum<'ctx>; 1] = [ptr_type.into()];
                let ret_type: BasicTypeEnum<'ctx> = self.ptr_int_type.into();
                let fn_type = ret_type.fn_type(&param_types, false);
                let fn_val = self.module.add_function(&len_fn_name, fn_type, None);
                let entry = self.context.append_basic_block(fn_val, "entry");
                let saved_bb = self.builder.get_insert_block();
                self.builder.position_at_end(entry);
                let len_val = self.ptr_int_type.const_int(len as u64, false);
                self.builder
                    .build_return(Some(&len_val))
                    .map_err(|e| format!("failed to build return for len: {}", e))?;
                if let Some(bb) = saved_bb {
                    self.builder.position_at_end(bb);
                }
            }
            self.impl_methods
                .get_mut(&key)
                .unwrap()
                .push(("len".to_string(), len_fn_name.clone()));
            self.fn_return_types
                .insert(len_fn_name.clone(), Type::Usize);
            self.fn_param_types.insert(
                len_fn_name.clone(),
                vec![Type::Ref {
                    inner: Box::new(self_array_type.clone()),
                    is_mut: false,
                }],
            );
        }

        if !existing_methods.contains(&"index".to_string()) {
            let index_fn_name = format!("__builtin_Index_index_{}", key);
            if self.module.get_function(&index_fn_name).is_none() {
                let param_types: [BasicMetadataTypeEnum<'ctx>; 2] =
                    [ptr_type.into(), self.ptr_int_type.into()];
                let ret_type: BasicTypeEnum<'ctx> = ptr_type.into();
                let fn_type = ret_type.fn_type(&param_types, false);
                let fn_val = self.module.add_function(&index_fn_name, fn_type, None);
                let entry = self.context.append_basic_block(fn_val, "entry");
                let saved_bb = self.builder.get_insert_block();
                self.builder.position_at_end(entry);
                let self_ptr = fn_val.get_first_param().unwrap().into_pointer_value();
                let idx_val = fn_val.get_nth_param(1).unwrap().into_int_value();
                let zero = self.ptr_int_type.const_zero();
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(array_llvm, self_ptr, &[zero, idx_val], "index_elem")
                        .map_err(|e| format!("failed to build GEP for index: {}", e))?
                };
                self.builder
                    .build_return(Some(&elem_ptr))
                    .map_err(|e| format!("failed to build return for index: {}", e))?;
                if let Some(bb) = saved_bb {
                    self.builder.position_at_end(bb);
                }
            }
            self.impl_methods
                .get_mut(&key)
                .unwrap()
                .push(("index".to_string(), index_fn_name.clone()));
            self.fn_return_types.insert(
                index_fn_name.clone(),
                Type::Ptr {
                    inner: Box::new(elem_ty.clone()),
                    is_mut: false,
                },
            );
            self.fn_param_types.insert(
                index_fn_name.clone(),
                vec![
                    Type::Ref {
                        inner: Box::new(self_array_type.clone()),
                        is_mut: false,
                    },
                    Type::Usize,
                ],
            );
        }

        if !existing_methods.contains(&"index_mut".to_string()) {
            let index_mut_fn_name = format!("__builtin_IndexMut_index_mut_{}", key);
            if self.module.get_function(&index_mut_fn_name).is_none() {
                let param_types: [BasicMetadataTypeEnum<'ctx>; 2] =
                    [ptr_type.into(), self.ptr_int_type.into()];
                let ret_type: BasicTypeEnum<'ctx> = ptr_type.into();
                let fn_type = ret_type.fn_type(&param_types, false);
                let fn_val = self.module.add_function(&index_mut_fn_name, fn_type, None);
                let entry = self.context.append_basic_block(fn_val, "entry");
                let saved_bb = self.builder.get_insert_block();
                self.builder.position_at_end(entry);
                let self_ptr = fn_val.get_first_param().unwrap().into_pointer_value();
                let idx_val = fn_val.get_nth_param(1).unwrap().into_int_value();
                let zero = self.ptr_int_type.const_zero();
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(
                            array_llvm,
                            self_ptr,
                            &[zero, idx_val],
                            "index_mut_elem",
                        )
                        .map_err(|e| format!("failed to build GEP for index_mut: {}", e))?
                };
                self.builder
                    .build_return(Some(&elem_ptr))
                    .map_err(|e| format!("failed to build return for index_mut: {}", e))?;
                if let Some(bb) = saved_bb {
                    self.builder.position_at_end(bb);
                }
            }
            self.impl_methods
                .get_mut(&key)
                .unwrap()
                .push(("index_mut".to_string(), index_mut_fn_name.clone()));
            self.fn_return_types.insert(
                index_mut_fn_name.clone(),
                Type::Ptr {
                    inner: Box::new(elem_ty.clone()),
                    is_mut: true,
                },
            );
            self.fn_param_types.insert(
                index_mut_fn_name.clone(),
                vec![
                    Type::Ref {
                        inner: Box::new(self_array_type.clone()),
                        is_mut: true,
                    },
                    Type::Usize,
                ],
            );
        }

        if !existing_methods.contains(&"as_ptr".to_string()) {
            let as_ptr_fn_name = format!("{}::as_ptr/1", key);
            if self.module.get_function(&as_ptr_fn_name).is_none() {
                let param_types: [BasicMetadataTypeEnum<'ctx>; 1] = [ptr_type.into()];
                let ret_type: BasicTypeEnum<'ctx> = ptr_type.into();
                let fn_type = ret_type.fn_type(&param_types, false);
                let fn_val = self.module.add_function(&as_ptr_fn_name, fn_type, None);
                let entry = self.context.append_basic_block(fn_val, "entry");
                let saved_bb = self.builder.get_insert_block();
                self.builder.position_at_end(entry);
                let self_ptr = fn_val.get_first_param().unwrap().into_pointer_value();
                let ptr = self
                    .builder
                    .build_bit_cast(self_ptr, ptr_type, "as_ptr")
                    .map_err(|e| format!("failed to build bitcast for as_ptr: {}", e))?;
                self.builder
                    .build_return(Some(&ptr))
                    .map_err(|e| format!("failed to build return for as_ptr: {}", e))?;
                if let Some(bb) = saved_bb {
                    self.builder.position_at_end(bb);
                }
            }
            self.impl_methods
                .get_mut(&key)
                .unwrap()
                .push(("as_ptr".to_string(), as_ptr_fn_name.clone()));
            self.fn_return_types.insert(
                as_ptr_fn_name.clone(),
                Type::Ptr {
                    inner: Box::new(elem_ty.clone()),
                    is_mut: false,
                },
            );
            self.fn_param_types.insert(
                as_ptr_fn_name.clone(),
                vec![Type::Ref {
                    inner: Box::new(self_array_type.clone()),
                    is_mut: false,
                }],
            );
        }

        Ok(())
    }

    /// Substitute type params in a function's param types and return type.
    fn substitute_type_params_in_func(func: &mut Function, params: &[String], args: &[Type]) {
        for param in &mut func.params {
            Self::substitute_type_params(&mut param.ty, params, args);
        }
        if let Some(ref mut ret_ty) = func.return_type {
            Self::substitute_type_params(ret_ty, params, args);
        }
    }

    fn compile_expr(&mut self, expr: &Expr) -> Result<BasicValueEnum<'ctx>, String> {
        match expr {
            Expr::BoolLit(val) => Ok(self
                .bool_type
                .const_int(if *val { 1 } else { 0 }, false)
                .into()),
            Expr::IntLit(val) => Ok(self.i32_type.const_int(*val as u64, true).into()),
            Expr::FloatLit(val) => Ok(self.f64_type.const_float(*val).into()),
            Expr::StrLit(s) => {
                let ptr = self
                    .builder
                    .build_global_string_ptr(s, "str")
                    .map_err(|e| format!("failed to build string literal: {}", e))?;
                let ptr_val = ptr.as_pointer_value();
                // Build { ptr, len } struct
                let len = s.len() as u64;
                let len_val = self.i64_type.const_int(len, false);
                let elems: [BasicTypeEnum; 2] = [
                    self.context.ptr_type(AddressSpace::default()).into(),
                    self.i64_type.into(),
                ];
                let struct_ty = self.context.struct_type(&elems, false);
                let str_struct = struct_ty.get_undef();
                let str_struct = self
                    .builder
                    .build_insert_value(str_struct, ptr_val, 0, "str_ptr")
                    .map_err(|e| format!("failed to build str ptr insert: {}", e))?;
                let str_struct = self
                    .builder
                    .build_insert_value(str_struct, len_val, 1, "str_len")
                    .map_err(|e| format!("failed to build str len insert: {}", e))?;
                Ok(str_struct.into_struct_value().into())
            }
            Expr::Ident(name) => {
                if let Some(val) = self.consts.get(name) {
                    return Ok(self.i32_type.const_int(*val as u64, true).into());
                }
                if let Some((ptr, _, ty)) = self.symbols.get(name) {
                    let llvm_ty = self.type_to_llvm(ty);
                    let val = self
                        .builder
                        .build_load(llvm_ty, *ptr, name)
                        .map_err(|e| format!("failed to build load: {}", e))?;
                    Ok(val)
                } else {
                    Err(format!("undefined variable '{}'", name))
                }
            }
            Expr::Assign { target, value } => match target.as_ref() {
                Expr::Ident(name) => {
                    if self.consts.contains_key(name) {
                        return Err(format!("cannot assign to constant '{}'", name));
                    }
                    let (ptr, is_mut, _) = self
                        .symbols
                        .get(name)
                        .map(|(p, m, t)| (*p, *m, t.clone()))
                        .ok_or_else(|| format!("undefined variable '{}'", name))?;
                    if !is_mut {
                        return Err(format!("cannot assign to immutable variable '{}'", name));
                    }
                    let val = self.compile_expr(value)?;
                    self.builder
                        .build_store(ptr, val)
                        .map_err(|e| format!("failed to build store: {}", e))?;
                    Ok(val)
                }
                Expr::Deref(inner) => {
                    let ptr = self.compile_expr(inner)?;
                    let ptr = ptr.into_pointer_value();
                    let val = self.compile_expr(value)?;
                    self.builder
                        .build_store(ptr, val)
                        .map_err(|e| format!("failed to build store through deref: {}", e))?;
                    Ok(val)
                }
                Expr::Member {
                    expr: member_expr,
                    index,
                    field,
                } => {
                    let parent_ptr_val = self.compile_expr(member_expr)?;
                    let parent_ptr = match parent_ptr_val {
                        BasicValueEnum::PointerValue(p) => p,
                        _ => {
                            if let Expr::Ident(name) = member_expr.as_ref() {
                                if let Some((ptr, _, _)) = self.symbols.get(name) {
                                    *ptr
                                } else {
                                    return Err(format!("undefined variable '{}'", name));
                                }
                            } else {
                                return Err("expected pointer for member assignment".to_string());
                            }
                        }
                    };
                    let parent_type = self.expr_type(member_expr);

                    let (struct_ty, struct_name) = match &parent_type {
                        Type::Ref { inner, is_mut } => {
                            if !is_mut {
                                return Err("cannot assign to field through immutable reference"
                                    .to_string());
                            }
                            match inner.as_ref() {
                                Type::Struct(name) => {
                                    let st =
                                        self.struct_types.get(name).copied().ok_or_else(|| {
                                            format!("unknown struct type '{}'", name)
                                        })?;
                                    (st, name.clone())
                                }
                                Type::GenericInstance(name, args) => {
                                    let mangled = Self::mangle_generic_instance(name, args);
                                    let st = self.struct_types.get(&mangled).copied().ok_or_else(
                                        || format!("unknown struct type '{}'", mangled),
                                    )?;
                                    (st, mangled)
                                }
                                _ => {
                                    return Err(
                                        "cannot access field on non-struct type".to_string()
                                    );
                                }
                            }
                        }
                        Type::Struct(name) => {
                            let st = self
                                .struct_types
                                .get(name)
                                .copied()
                                .ok_or_else(|| format!("unknown struct type '{}'", name))?;
                            (st, name.clone())
                        }
                        Type::GenericInstance(name, args) => {
                            let mangled = Self::mangle_generic_instance(name, args);
                            let st = self
                                .struct_types
                                .get(&mangled)
                                .copied()
                                .ok_or_else(|| format!("unknown struct type '{}'", mangled))?;
                            (st, mangled)
                        }
                        _ => return Err("cannot assign to field on non-struct type".to_string()),
                    };

                    if let Some(field_name) = field {
                        self.check_field_visibility(
                            &self.current_module_path,
                            &struct_name,
                            field_name,
                        )?;
                    }

                    let field_idx = if let Some(field_name) = field {
                        let fields = self
                            .struct_fields
                            .get(&struct_name)
                            .ok_or_else(|| format!("unknown struct '{}'", struct_name))?;
                        fields
                            .iter()
                            .position(|f| f.name == *field_name)
                            .ok_or_else(|| {
                                format!("struct '{}' has no field '{}'", struct_name, field_name)
                            })? as u32
                    } else {
                        *index as u32
                    };

                    let field_ptr = self
                        .builder
                        .build_struct_gep(struct_ty, parent_ptr, field_idx, "field_ptr")
                        .map_err(|e| {
                            format!("failed to get field pointer for assignment: {}", e)
                        })?;
                    let mut val = self.compile_expr(value)?;
                    // Coerce to the field's declared type
                    if let Some(fields) = self.struct_fields.get(&struct_name)
                        && let Some(field_def) = fields.get(field_idx as usize)
                    {
                        let field_llvm_ty = self.type_to_llvm(&field_def.ty);
                        if val.get_type() != field_llvm_ty {
                            val = self.emit_cast(val, &field_def.ty)?;
                        }
                    }
                    self.builder
                        .build_store(field_ptr, val)
                        .map_err(|e| format!("failed to store field value: {}", e))?;
                    Ok(val)
                }
                Expr::Index { array, index } => {
                    let array_ty = self.expr_type(array);
                    let (elem_ty, len) = match &array_ty {
                        Type::Array { inner, len } => (*inner.clone(), *len),
                        Type::Ref { inner, .. } => match inner.as_ref() {
                            Type::Array { inner: ai, len } => (*ai.clone(), *len),
                            _ => {
                                return Err(format!(
                                    "cannot index into non-array type {:?}",
                                    array_ty
                                ));
                            }
                        },
                        _ => {
                            return Err(format!("cannot index into non-array type {:?}", array_ty));
                        }
                    };
                    self.ensure_slice_methods(&elem_ty, len)?;
                    let key = Self::array_type_key(&elem_ty, len);
                    let methods = self
                        .impl_methods
                        .get(&key)
                        .ok_or_else(|| format!("no methods registered for array type '{}'", key))?;
                    let fn_name = methods
                        .iter()
                        .find(|(name, _)| name == "index_mut")
                        .map(|(_, mangled)| mangled.clone())
                        .ok_or_else(|| format!("no 'index_mut' method for array type '{}'", key))?;
                    let fn_val = self.module.get_function(&fn_name).ok_or_else(|| {
                        format!("internal: index_mut function '{}' not found", fn_name)
                    })?;
                    // Compile array to a pointer — use original alloca if it's a variable
                    let array_ptr = if let Expr::Ident(name) = array.as_ref() {
                        if let Some((ptr, _, _)) = self.symbols.get(name) {
                            *ptr
                        } else {
                            return Err(format!("undefined variable '{}'", name));
                        }
                    } else {
                        let array_val = self.compile_expr(array)?;
                        match array_val {
                            BasicValueEnum::PointerValue(p) => p,
                            _ => {
                                let alloca = self
                                    .builder
                                    .build_alloca(array_val.get_type(), "array_mut_ref")
                                    .map_err(|e| {
                                        format!("failed to build alloca for array mut ref: {}", e)
                                    })?;
                                self.builder
                                    .build_store(alloca, array_val)
                                    .map_err(|e| format!("failed to store array: {}", e))?;
                                alloca
                            }
                        }
                    };
                    let idx_val = self.compile_expr(index)?;
                    let result = self
                        .builder
                        .build_call(
                            fn_val,
                            &[array_ptr.into(), idx_val.into()],
                            "index_mut_call",
                        )
                        .map_err(|e| format!("failed to call index_mut: {}", e))?;
                    let result_ptr = self.try_extract_result(result).into_pointer_value();
                    let val = self.compile_expr(value)?;
                    self.builder
                        .build_store(result_ptr, val)
                        .map_err(|e| format!("failed to store through index_mut: {}", e))?;
                    Ok(val)
                }
                _ => Err("invalid assignment target".to_string()),
            },
            Expr::Ref { expr, .. } => {
                if let Expr::Ident(name) = expr.as_ref() {
                    if let Some((ptr, _, _)) = self.symbols.get(name) {
                        Ok((*ptr).into())
                    } else {
                        Err(format!("undefined variable '{}'", name))
                    }
                } else if let Expr::Deref(inner) = expr.as_ref() {
                    self.compile_expr(inner)
                } else {
                    let ty = Self::literal_type(expr);
                    let llvm_ty = self.type_to_llvm(&ty);
                    let alloca = self
                        .builder
                        .build_alloca(llvm_ty, "ref_temp")
                        .map_err(|e| format!("failed to build ref temp alloca: {}", e))?;
                    let val = self.compile_expr(expr)?;
                    self.builder
                        .build_store(alloca, val)
                        .map_err(|e| format!("failed to store ref temp: {}", e))?;
                    Ok(alloca.into())
                }
            }
            Expr::Deref(expr) => {
                let ptr_val = self.compile_expr(expr)?;
                let ptr = ptr_val.into_pointer_value();
                // Determine pointee type
                let pointee_ty = self.expr_type(expr);
                let pointee_llvm_ty = self.type_to_llvm(match &pointee_ty {
                    Type::Ref { inner, .. } => inner,
                    Type::Ptr { inner, .. } => inner,
                    _ => &pointee_ty,
                });
                let val = self
                    .builder
                    .build_load(pointee_llvm_ty, ptr, "deref")
                    .map_err(|e| format!("failed to build deref load: {}", e))?;
                Ok(val)
            }
            Expr::Call { callee, args } => {
                if callee != "__builtin_size_of" && !callee.starts_with("__builtin_slice_") {
                    self.check_visibility_of_path(&self.current_module_path, callee)?;
                }
                // Handle __builtin_size_of intrinsic
                if callee == "__builtin_size_of" && args.len() == 1 {
                    let arg_ty = self.expr_type(&args[0]);
                    let llvm_ty = self.type_to_llvm(&arg_ty);
                    let size = match llvm_ty {
                        inkwell::types::BasicTypeEnum::IntType(i) => i.get_bit_width() as u64 / 8,
                        inkwell::types::BasicTypeEnum::FloatType(f) => {
                            if f == self.f32_type {
                                4
                            } else {
                                8
                            }
                        }
                        inkwell::types::BasicTypeEnum::PointerType(_) => 8,
                        inkwell::types::BasicTypeEnum::StructType(s) => {
                            let mut total = 0u64;
                            for i in 0..s.count_fields() {
                                if let Some(ft) = s.get_field_type_at_index(i) {
                                    match ft {
                                        inkwell::types::BasicTypeEnum::IntType(i) => {
                                            total += i.get_bit_width() as u64 / 8
                                        }
                                        inkwell::types::BasicTypeEnum::FloatType(f) => {
                                            total += if f == self.f32_type { 4 } else { 8 }
                                        }
                                        inkwell::types::BasicTypeEnum::PointerType(_) => total += 8,
                                        _ => total += 4,
                                    }
                                }
                            }
                            if total == 0 {
                                total = 4;
                            }
                            total
                        }
                        _ => 4,
                    };
                    return Ok(self.ptr_int_type.const_int(size, false).into());
                }
                // Handle builtin slice intrinsics
                if let Some(result) = self.compile_slice_intrinsic(callee, args)? {
                    return Ok(result);
                }
                let arg_types: Vec<Type> = args.iter().map(|a| self.expr_type(a)).collect();
                let fn_val = self
                    .resolve_function(callee, &arg_types)
                    .ok_or_else(|| format!("unknown function '{}'", callee))?;
                let mut arg_values = self.compile_args_vec(args)?;
                // Coerce arguments to match declared parameter types
                if let Some(param_tys) = self.fn_param_types.get(callee) {
                    for (i, arg_val) in arg_values.iter_mut().enumerate() {
                        if let Some(param_ty) = param_tys.get(i) {
                            let param_llvm_ty = self.type_to_llvm(param_ty);
                            // Compare types by extracting BasicValueEnum
                            let bv = match arg_val {
                                BasicMetadataValueEnum::IntValue(v) => BasicValueEnum::IntValue(*v),
                                BasicMetadataValueEnum::FloatValue(v) => {
                                    BasicValueEnum::FloatValue(*v)
                                }
                                BasicMetadataValueEnum::PointerValue(v) => {
                                    BasicValueEnum::PointerValue(*v)
                                }
                                BasicMetadataValueEnum::StructValue(v) => {
                                    BasicValueEnum::StructValue(*v)
                                }
                                &mut _ => continue,
                            };
                            if bv.get_type() != param_llvm_ty {
                                let cast = self.emit_cast(bv, param_ty)?;
                                *arg_val = match cast {
                                    BasicValueEnum::IntValue(v) => v.into(),
                                    BasicValueEnum::FloatValue(v) => v.into(),
                                    BasicValueEnum::PointerValue(v) => v.into(),
                                    BasicValueEnum::StructValue(v) => v.into(),
                                    _ => continue,
                                };
                            }
                        }
                    }
                }
                let result = self
                    .builder
                    .build_call(fn_val, &arg_values, "call")
                    .map_err(|e| format!("failed to build call to '{}': {}", callee, e))?;
                Ok(self.try_extract_result(result))
            }
            Expr::QualifiedCall {
                module,
                callee,
                args,
            } => {
                let qualified_name = format!("{}::{}", module, callee);
                self.check_visibility_of_path(&self.current_module_path, &qualified_name)?;
                let mangled_name = format!("{}::{}/{}", module, callee, args.len());
                let arg_types: Vec<Type> = args.iter().map(|a| self.expr_type(a)).collect();
                // For generic structs, also try the monomorphized name
                let monomorphized_name = Some(self.resolve_mangled_name(module))
                    .map(|mangled| format!("{}::{}/{}", mangled, callee, args.len()));
                // Try direct lookup first, then mangled name (for impl methods), then overload map
                let mut resolved_fn_val = self
                    .module
                    .get_function(&qualified_name)
                    .or_else(|| self.module.get_function(&mangled_name))
                    .or_else(|| {
                        monomorphized_name
                            .as_ref()
                            .and_then(|n| self.module.get_function(n))
                    })
                    .or_else(|| self.resolve_function(&qualified_name, &arg_types))
                    .or_else(|| self.resolve_function(callee, &arg_types))
                    .or_else(|| self.resolve_function(&mangled_name, &arg_types));

                // If not resolved directly, try looking up in impl_methods
                if resolved_fn_val.is_none()
                    && let Some(methods) = self.impl_methods.get(module)
                    && let Some((_, mangled)) = methods.iter().find(|(name, _)| name == callee)
                {
                    resolved_fn_val = self.module.get_function(mangled);
                }

                let fn_val = resolved_fn_val
                    .ok_or_else(|| format!("unknown function '{}'", qualified_name))?;
                let mut arg_values = self.compile_args_vec(args)?;
                // Coerce arguments to match declared parameter types
                let resolved_name = fn_val.get_name().to_string_lossy().to_string();
                if let Some(param_tys) = self.fn_param_types.get(&resolved_name) {
                    for (i, arg_val) in arg_values.iter_mut().enumerate() {
                        if let Some(param_ty) = param_tys.get(i) {
                            let param_llvm_ty = self.type_to_llvm(param_ty);
                            let bv = match arg_val {
                                BasicMetadataValueEnum::IntValue(v) => BasicValueEnum::IntValue(*v),
                                BasicMetadataValueEnum::FloatValue(v) => {
                                    BasicValueEnum::FloatValue(*v)
                                }
                                BasicMetadataValueEnum::PointerValue(v) => {
                                    BasicValueEnum::PointerValue(*v)
                                }
                                BasicMetadataValueEnum::StructValue(v) => {
                                    BasicValueEnum::StructValue(*v)
                                }
                                &mut _ => continue,
                            };
                            if bv.get_type() != param_llvm_ty {
                                let cast = self.emit_cast(bv, param_ty)?;
                                *arg_val = match cast {
                                    BasicValueEnum::IntValue(v) => v.into(),
                                    BasicValueEnum::FloatValue(v) => v.into(),
                                    BasicValueEnum::PointerValue(v) => v.into(),
                                    BasicValueEnum::StructValue(v) => v.into(),
                                    _ => continue,
                                };
                            }
                        }
                    }
                }
                let result = self
                    .builder
                    .build_call(fn_val, &arg_values, "call")
                    .map_err(|e| format!("failed to build call to '{}': {}", qualified_name, e))?;
                Ok(self.try_extract_result(result))
            }
            Expr::Binary { op, lhs, rhs } => {
                if *op == BinOp::And {
                    let parent_fn = self
                        .builder
                        .get_insert_block()
                        .unwrap()
                        .get_parent()
                        .unwrap();

                    let rhs_bb = self.context.append_basic_block(parent_fn, "and_rhs");
                    let merge_bb = self.context.append_basic_block(parent_fn, "and_merge");

                    let result_alloca = self
                        .builder
                        .build_alloca(self.bool_type, "and_result")
                        .map_err(|e| format!("failed to build and result alloca: {}", e))?;

                    let lhs_val = self.compile_expr(lhs)?;
                    let lhs_i1 = match lhs_val {
                        BasicValueEnum::IntValue(v) => {
                            if v.get_type().get_bit_width() != 1 {
                                let zero = v.get_type().const_zero();
                                self.builder
                                    .build_int_compare(inkwell::IntPredicate::NE, v, zero, "lhs_i1")
                                    .map_err(|e| format!("failed to compare lhs != 0: {}", e))?
                            } else {
                                v
                            }
                        }
                        _ => {
                            return Err(
                                "logical AND operands must be booleans or integers".to_string()
                            );
                        }
                    };

                    self.builder
                        .build_store(result_alloca, lhs_i1)
                        .map_err(|e| format!("failed to store lhs: {}", e))?;

                    let lhs_terminated = self
                        .builder
                        .get_insert_block()
                        .unwrap()
                        .get_terminator()
                        .is_some();

                    if lhs_terminated {
                        return Ok(lhs_val);
                    }

                    self.builder
                        .build_conditional_branch(lhs_i1, rhs_bb, merge_bb)
                        .map_err(|e| {
                            format!("failed to build conditional branch for AND: {}", e)
                        })?;

                    // Compile RHS
                    self.builder.position_at_end(rhs_bb);
                    let rhs_val = self.compile_expr(rhs)?;
                    let rhs_i1 = match rhs_val {
                        BasicValueEnum::IntValue(v) => {
                            if v.get_type().get_bit_width() != 1 {
                                let zero = v.get_type().const_zero();
                                self.builder
                                    .build_int_compare(inkwell::IntPredicate::NE, v, zero, "rhs_i1")
                                    .map_err(|e| format!("failed to compare rhs != 0: {}", e))?
                            } else {
                                v
                            }
                        }
                        _ => {
                            return Err(
                                "logical AND operands must be booleans or integers".to_string()
                            );
                        }
                    };

                    self.builder
                        .build_store(result_alloca, rhs_i1)
                        .map_err(|e| format!("failed to store rhs: {}", e))?;

                    let rhs_terminated = self
                        .builder
                        .get_insert_block()
                        .unwrap()
                        .get_terminator()
                        .is_some();

                    if !rhs_terminated {
                        self.builder
                            .build_unconditional_branch(merge_bb)
                            .map_err(|e| {
                                format!(
                                    "failed to build unconditional branch to merge for AND: {}",
                                    e
                                )
                            })?;
                    }

                    // Merge Block
                    self.builder.position_at_end(merge_bb);
                    let res = self
                        .builder
                        .build_load(self.bool_type, result_alloca, "and_res")
                        .map_err(|e| format!("failed to load AND result: {}", e))?;
                    return Ok(res);
                }

                if *op == BinOp::Or {
                    let parent_fn = self
                        .builder
                        .get_insert_block()
                        .unwrap()
                        .get_parent()
                        .unwrap();

                    let rhs_bb = self.context.append_basic_block(parent_fn, "or_rhs");
                    let merge_bb = self.context.append_basic_block(parent_fn, "or_merge");

                    let result_alloca = self
                        .builder
                        .build_alloca(self.bool_type, "or_result")
                        .map_err(|e| format!("failed to build or result alloca: {}", e))?;

                    let lhs_val = self.compile_expr(lhs)?;
                    let lhs_i1 = match lhs_val {
                        BasicValueEnum::IntValue(v) => {
                            if v.get_type().get_bit_width() != 1 {
                                let zero = v.get_type().const_zero();
                                self.builder
                                    .build_int_compare(inkwell::IntPredicate::NE, v, zero, "lhs_i1")
                                    .map_err(|e| format!("failed to compare lhs != 0: {}", e))?
                            } else {
                                v
                            }
                        }
                        _ => {
                            return Err(
                                "logical OR operands must be booleans or integers".to_string()
                            );
                        }
                    };

                    self.builder
                        .build_store(result_alloca, lhs_i1)
                        .map_err(|e| format!("failed to store lhs: {}", e))?;

                    let lhs_terminated = self
                        .builder
                        .get_insert_block()
                        .unwrap()
                        .get_terminator()
                        .is_some();

                    if lhs_terminated {
                        return Ok(lhs_val);
                    }

                    self.builder
                        .build_conditional_branch(lhs_i1, merge_bb, rhs_bb)
                        .map_err(|e| format!("failed to build conditional branch for OR: {}", e))?;

                    // Compile RHS
                    self.builder.position_at_end(rhs_bb);
                    let rhs_val = self.compile_expr(rhs)?;
                    let rhs_i1 = match rhs_val {
                        BasicValueEnum::IntValue(v) => {
                            if v.get_type().get_bit_width() != 1 {
                                let zero = v.get_type().const_zero();
                                self.builder
                                    .build_int_compare(inkwell::IntPredicate::NE, v, zero, "rhs_i1")
                                    .map_err(|e| format!("failed to compare rhs != 0: {}", e))?
                            } else {
                                v
                            }
                        }
                        _ => {
                            return Err(
                                "logical OR operands must be booleans or integers".to_string()
                            );
                        }
                    };

                    self.builder
                        .build_store(result_alloca, rhs_i1)
                        .map_err(|e| format!("failed to store rhs: {}", e))?;

                    let rhs_terminated = self
                        .builder
                        .get_insert_block()
                        .unwrap()
                        .get_terminator()
                        .is_some();

                    if !rhs_terminated {
                        self.builder
                            .build_unconditional_branch(merge_bb)
                            .map_err(|e| {
                                format!(
                                    "failed to build unconditional branch to merge for OR: {}",
                                    e
                                )
                            })?;
                    }

                    // Merge Block
                    self.builder.position_at_end(merge_bb);
                    let res = self
                        .builder
                        .build_load(self.bool_type, result_alloca, "or_res")
                        .map_err(|e| format!("failed to load OR result: {}", e))?;
                    return Ok(res);
                }

                let lhs_val = self.compile_expr(lhs)?;
                let rhs_val = self.compile_expr(rhs)?;

                let result = match op {
                    BinOp::Add | BinOp::Sub | BinOp::Mul | BinOp::Div => {
                        self.compile_trait_arithmetic(op, lhs, rhs, lhs_val, rhs_val)?
                    }
                    _ => self.compile_trait_comparison(op, lhs, rhs, lhs_val, rhs_val)?,
                };
                Ok(result)
            }
            Expr::Tuple(elems) => {
                let compiled: Vec<BasicValueEnum<'ctx>> = elems
                    .iter()
                    .map(|e| self.compile_expr(e))
                    .collect::<Result<Vec<_>, _>>()?;
                let llvm_elems: Vec<BasicTypeEnum<'ctx>> =
                    compiled.iter().map(|v| v.get_type()).collect();
                let struct_ty = self.context.struct_type(&llvm_elems, false);
                let initial: inkwell::values::AggregateValueEnum<'ctx> =
                    struct_ty.get_undef().into();
                let mut result = initial;
                for (i, val) in compiled.iter().enumerate() {
                    result = self
                        .builder
                        .build_insert_value(result, *val, i as u32, "tuple_elem")
                        .map_err(|e| format!("failed to build tuple elem {}: {}", i, e))?;
                }
                Ok(result.into_struct_value().into())
            }
            Expr::Unit => {
                let unit_ty = self.context.struct_type(&[], false);
                Ok(unit_ty.get_undef().into())
            }
            Expr::Member {
                expr, index, field, ..
            } => {
                let mut val = self.compile_expr(expr)?;
                // If the result is a pointer, load the struct first
                if val.is_pointer_value() {
                    let ptr = val.into_pointer_value();
                    // Determine struct type from the expression type
                    let ty = self.expr_type(expr);
                    match &ty {
                        Type::Ref { inner, .. } => {
                            let llvm_ty = self.type_to_llvm(inner);
                            val = self
                                .builder
                                .build_load(llvm_ty, ptr, "struct_deref")
                                .map_err(|e| {
                                    format!("failed to load struct for field access: {}", e)
                                })?;
                        }
                        Type::Struct(_) | Type::GenericInstance(_, _) => {
                            let llvm_ty = self.type_to_llvm(&ty);
                            val = self
                                .builder
                                .build_load(llvm_ty, ptr, "struct_deref")
                                .map_err(|e| {
                                    format!("failed to load struct for field access: {}", e)
                                })?;
                        }
                        _ => {}
                    }
                }
                match val {
                    BasicValueEnum::StructValue(sv) => {
                        let struct_ty = sv.get_type();
                        let effective_index = if let Some(field_name) = field {
                            // Look up field index from struct type definition
                            // Infer the struct name from the expression type
                            let parent_ty = self.expr_type(expr);
                            let struct_name = match &parent_ty {
                                Type::Struct(name) => name.clone(),
                                Type::GenericInstance(name, args) => {
                                    Self::mangle_generic_instance(name, args)
                                }
                                Type::Ref { inner, .. } => match inner.as_ref() {
                                    Type::Struct(name) => name.clone(),
                                    Type::GenericInstance(name, args) => {
                                        Self::mangle_generic_instance(name, args)
                                    }
                                    _ => {
                                        return Err(format!(
                                            "cannot access field '{}' on non-struct type",
                                            field_name
                                        ));
                                    }
                                },
                                _ => {
                                    return Err(format!(
                                        "cannot access field '{}' on non-struct type",
                                        field_name
                                    ));
                                }
                            };
                            if let Some(field_name) = field {
                                self.check_field_visibility(
                                    &self.current_module_path,
                                    &struct_name,
                                    field_name,
                                )?;
                            }
                            let fields = self
                                .struct_fields
                                .get(&struct_name)
                                .ok_or_else(|| format!("unknown struct '{}'", struct_name))?;
                            let idx = fields
                                .iter()
                                .position(|f| f.name == *field_name)
                                .ok_or_else(|| {
                                    format!(
                                        "struct '{}' has no field '{}'",
                                        struct_name, field_name
                                    )
                                })?;
                            idx as u32
                        } else {
                            *index as u32
                        };
                        let field_count = struct_ty.count_fields();
                        if (effective_index as usize) >= field_count as usize {
                            return Err(format!(
                                "field index {} out of bounds; struct has {} fields",
                                effective_index, field_count
                            ));
                        }
                        let extracted = self
                            .builder
                            .build_extract_value(sv, effective_index, "member")
                            .map_err(|e| format!("failed to extract struct field: {}", e))?;
                        Ok(extracted)
                    }
                    _ => Err("cannot access member of non-struct value".to_string()),
                }
            }
            Expr::MethodCall {
                expr: receiver,
                method,
                args,
            } => {
                // Special case: .len() on &str
                if method == "len" && args.is_empty() {
                    let rt = self.expr_type(receiver);
                    let is_str = matches!(&rt, Type::Ref { inner, .. } if **inner == Type::Str);
                    if is_str {
                        let val = self.compile_expr(receiver)?;
                        match val {
                            BasicValueEnum::StructValue(sv) => {
                                let len = self
                                    .builder
                                    .build_extract_value(sv, 1, "str_len")
                                    .map_err(|e| format!("failed to extract str length: {}", e))?;
                                return Ok(len);
                            }
                            _ => return Err("cannot call .len() on a non-string value".to_string()),
                        }
                    }
                }

                // Determine receiver type
                let receiver_type = self.expr_type(receiver);
                let (type_name, _is_ref) = match &receiver_type {
                    Type::Array { inner, len } => {
                        self.ensure_slice_methods(inner, *len)?;
                        (Self::array_type_key(inner, *len), false)
                    }
                    Type::Struct(name) => (name.clone(), false),
                    Type::GenericInstance(name, args) => {
                        (Self::mangle_generic_instance(name, args), false)
                    }
                    Type::Ref { inner, .. } => match inner.as_ref() {
                        Type::Array { inner, len } => {
                            self.ensure_slice_methods(inner, *len)?;
                            (Self::array_type_key(inner, *len), true)
                        }
                        Type::Struct(name) => (name.clone(), true),
                        Type::GenericInstance(name, args) => {
                            (Self::mangle_generic_instance(name, args), true)
                        }
                        Type::Bool => ("bool".to_string(), true),
                        Type::I8 => ("i8".to_string(), true),
                        Type::I16 => ("i16".to_string(), true),
                        Type::I32 => ("i32".to_string(), true),
                        Type::I64 => ("i64".to_string(), true),
                        Type::U8 => ("u8".to_string(), true),
                        Type::U16 => ("u16".to_string(), true),
                        Type::U32 => ("u32".to_string(), true),
                        Type::U64 => ("u64".to_string(), true),
                        Type::Usize => ("usize".to_string(), true),
                        Type::Isize => ("isize".to_string(), true),
                        Type::F32 => ("f32".to_string(), true),
                        Type::F64 => ("f64".to_string(), true),
                        _ => {
                            return Err(format!(
                                "cannot call method '{}' on type {:?}",
                                method, receiver_type
                            ));
                        }
                    },
                    Type::Bool => ("bool".to_string(), false),
                    Type::I8 => ("i8".to_string(), false),
                    Type::I16 => ("i16".to_string(), false),
                    Type::I32 => ("i32".to_string(), false),
                    Type::I64 => ("i64".to_string(), false),
                    Type::U8 => ("u8".to_string(), false),
                    Type::U16 => ("u16".to_string(), false),
                    Type::U32 => ("u32".to_string(), false),
                    Type::U64 => ("u64".to_string(), false),
                    Type::Usize => ("usize".to_string(), false),
                    Type::Isize => ("isize".to_string(), false),
                    Type::F32 => ("f32".to_string(), false),
                    Type::F64 => ("f64".to_string(), false),
                    _ => {
                        return Err(format!(
                            "cannot call method '{}' on type {:?}",
                            method, receiver_type
                        ));
                    }
                };

                // Look up method
                let methods = self.impl_methods.get(&type_name).ok_or_else(|| {
                    format!("type '{}' has no methods (no impl block found)", type_name)
                })?;

                let mangled = methods
                    .iter()
                    .find(|(name, _)| name == method)
                    .map(|(_, mangled)| mangled.clone())
                    .ok_or_else(|| {
                        eprintln!("DEBUG methods for {}: {:?}", type_name, methods);
                        format!("type '{}' has no method '{}'", type_name, method)
                    })?;

                self.check_visibility_of_path(&self.current_module_path, &mangled)?;

                let fn_val = self
                    .module
                    .get_function(&mangled)
                    .ok_or_else(|| format!("internal: method '{}' not found in module", mangled))?;

                // Compile receiver as first arg (pass pointer for &self)
                let receiver_val = self.compile_expr(receiver)?;
                let receiver_ptr = match receiver_val {
                    BasicValueEnum::PointerValue(p) => p,
                    _ => {
                        if let Expr::Ident(name) = receiver.as_ref() {
                            if let Some((ptr, _, _)) = self.symbols.get(name) {
                                *ptr
                            } else {
                                return Err(format!("undefined variable '{}'", name));
                            }
                        } else {
                            if let Expr::Member {
                                expr: member_expr,
                                index: _,
                                field: Some(field_name),
                                ..
                            } = receiver.as_ref()
                            {
                                let parent_ty = self.expr_type(member_expr);
                                if let Type::Ref { inner, .. } = &parent_ty {
                                    match inner.as_ref() {
                                        Type::Struct(_) | Type::GenericInstance(_, _) => {
                                            let struct_name = match inner.as_ref() {
                                                Type::Struct(name) => name.clone(),
                                                Type::GenericInstance(name, args) => {
                                                    Self::mangle_generic_instance(name, args)
                                                }
                                                _ => unreachable!(),
                                            };
                                            let fields =
                                                self.struct_fields.get(&struct_name).ok_or_else(
                                                    || format!("unknown struct '{}'", struct_name),
                                                )?;
                                            let idx = fields
                                                .iter()
                                                .position(|f| f.name == *field_name)
                                                .ok_or_else(|| {
                                                    format!(
                                                        "struct '{}' has no field '{}'",
                                                        struct_name, field_name
                                                    )
                                                })?;
                                            let llvm_ty = self.type_to_llvm(inner);
                                            let member_ptr = self.compile_expr(member_expr)?;
                                            let base_ptr = match member_ptr {
                                                BasicValueEnum::PointerValue(p) => p,
                                                ptr_val => {
                                                    let a = self
                                                        .builder
                                                        .build_alloca(
                                                            ptr_val.get_type(),
                                                            "ref_base",
                                                        )
                                                        .map_err(|e| {
                                                            format!("failed to build alloca: {}", e)
                                                        })?;
                                                    self.builder.build_store(a, ptr_val).map_err(
                                                        |e| format!("failed to store: {}", e),
                                                    )?;
                                                    a
                                                }
                                            };
                                            let field_ptr = self
                                                .builder
                                                .build_struct_gep(
                                                    llvm_ty,
                                                    base_ptr,
                                                    idx as u32,
                                                    "field_ptr",
                                                )
                                                .map_err(|e| {
                                                    format!(
                                                        "failed to build struct GEP for field '{}': {}",
                                                        field_name, e
                                                    )
                                                })?;

                                            let mut all_args = vec![field_ptr.into()];
                                            for arg in args {
                                                let val = self.compile_expr(arg)?;
                                                all_args.push(val.into());
                                            }
                                            let result = self
                                                .builder
                                                .build_call(fn_val, &all_args, "method_call")
                                                .map_err(|e| {
                                                    format!(
                                                        "failed to build method call '{}': {}",
                                                        method, e
                                                    )
                                                })?;
                                            return Ok(self.try_extract_result(result));
                                        }
                                        _ => {}
                                    }
                                }
                            }

                            // If receiver is a value, store to alloca and pass pointer
                            let alloca = self
                                .builder
                                .build_alloca(receiver_val.get_type(), "method_self")
                                .map_err(|e| {
                                    format!("failed to build alloca for method receiver: {}", e)
                                })?;
                            self.builder
                                .build_store(alloca, receiver_val)
                                .map_err(|e| {
                                    format!("failed to store receiver for method call: {}", e)
                                })?;
                            alloca
                        }
                    }
                };

                let mut all_args = vec![receiver_ptr.into()];
                for arg in args {
                    let val = self.compile_expr(arg)?;
                    all_args.push(val.into());
                }

                let result = self
                    .builder
                    .build_call(fn_val, &all_args, "method_call")
                    .map_err(|e| format!("failed to build method call '{}': {}", method, e))?;
                Ok(self.try_extract_result(result))
            }
            Expr::StructLit {
                struct_name,
                fields,
                ..
            } => {
                let actual_name = self.resolve_mangled_name(struct_name);
                self.check_visibility_of_path(&self.current_module_path, &actual_name)?;
                self.check_struct_literal_construction(&self.current_module_path, &actual_name)?;

                let struct_type = self
                    .struct_types
                    .get(&actual_name)
                    .copied()
                    .ok_or_else(|| format!("unknown struct '{}'", actual_name))?;
                let struct_val: inkwell::values::AggregateValueEnum<'ctx> =
                    struct_type.get_undef().into();
                let mut result = struct_val;
                // Pre-fetch struct field types for coercion
                let struct_def_fields: Vec<Type> = self
                    .struct_fields
                    .get(&actual_name)
                    .map(|f| f.iter().map(|fd| fd.ty.clone()).collect())
                    .unwrap_or_default();
                for (i, (field_name, field_expr)) in fields.iter().enumerate() {
                    let mut field_val = if let Some(field_ty) = struct_def_fields.get(i) {
                        self.with_expected_type(field_ty, |this| this.compile_expr(field_expr))?
                    } else {
                        self.compile_expr(field_expr)?
                    };
                    // Coerce field value to the declared field type
                    if let Some(field_ty) = struct_def_fields.get(i) {
                        let field_llvm_ty = self.type_to_llvm(field_ty);
                        if field_val.get_type() != field_llvm_ty {
                            field_val = self.emit_cast(field_val, field_ty)?;
                        }
                    }
                    result = self
                        .builder
                        .build_insert_value(result, field_val, i as u32, field_name)
                        .map_err(|e| {
                            format!("failed to insert struct field '{}': {}", field_name, e)
                        })?;
                }
                match result {
                    inkwell::values::AggregateValueEnum::StructValue(sv) => Ok(sv.into()),
                    _ => Err(format!("expected struct value for '{}'", struct_name)),
                }
            }
            Expr::Array(elems) => {
                if elems.is_empty() {
                    return Err("empty array literals are not supported".to_string());
                }
                let elem_ty = self.expr_type(&elems[0]);
                let elem_llvm = self.type_to_llvm(&elem_ty);
                let len = elems.len() as u32;
                let array_llvm: BasicTypeEnum<'ctx> = match elem_llvm {
                    BasicTypeEnum::IntType(it) => it.array_type(len).into(),
                    BasicTypeEnum::FloatType(ft) => ft.array_type(len).into(),
                    BasicTypeEnum::StructType(st) => st.array_type(len).into(),
                    BasicTypeEnum::ArrayType(at) => at.array_type(len).into(),
                    _ => panic!("unsupported array element type: {:?}", elem_llvm),
                };
                let alloca = self
                    .builder
                    .build_alloca(array_llvm, "array")
                    .map_err(|e| format!("failed to build array alloca: {}", e))?;
                for (i, elem_expr) in elems.iter().enumerate() {
                    let elem_val = self.compile_expr(elem_expr)?;
                    let zero = self.ptr_int_type.const_zero();
                    let idx = self.ptr_int_type.const_int(i as u64, false);
                    let elem_ptr = unsafe {
                        self.builder
                            .build_in_bounds_gep(
                                array_llvm,
                                alloca,
                                &[zero, idx],
                                &format!("array_elem_{}", i),
                            )
                            .map_err(|e| {
                                format!("failed to build GEP for array elem {}: {}", i, e)
                            })?
                    };
                    self.builder
                        .build_store(elem_ptr, elem_val)
                        .map_err(|e| format!("failed to store array elem {}: {}", i, e))?;
                }
                // Generate Index/IndexMut impls for this array type on demand
                self.ensure_slice_methods(&elem_ty, elems.len())?;
                let result = self
                    .builder
                    .build_load(array_llvm, alloca, "array_val")
                    .map_err(|e| format!("failed to load array: {}", e))?;
                Ok(result)
            }
            Expr::Repeat(expr, count) => {
                let elem_ty = self.expr_type(expr);
                let elem_llvm = self.type_to_llvm(&elem_ty);
                let len = *count as u32;
                let array_llvm: BasicTypeEnum<'ctx> = match elem_llvm {
                    BasicTypeEnum::IntType(it) => it.array_type(len).into(),
                    BasicTypeEnum::FloatType(ft) => ft.array_type(len).into(),
                    BasicTypeEnum::StructType(st) => st.array_type(len).into(),
                    BasicTypeEnum::ArrayType(at) => at.array_type(len).into(),
                    _ => panic!("unsupported array element type: {:?}", elem_llvm),
                };
                let alloca = self
                    .builder
                    .build_alloca(array_llvm, "array_repeat")
                    .map_err(|e| format!("failed to build array repeat alloca: {}", e))?;
                let elem_val = self.compile_expr(expr)?;
                for i in 0..*count {
                    let zero = self.ptr_int_type.const_zero();
                    let idx = self.ptr_int_type.const_int(i as u64, false);
                    let elem_ptr = unsafe {
                        self.builder
                            .build_in_bounds_gep(
                                array_llvm,
                                alloca,
                                &[zero, idx],
                                &format!("rep_elem_{}", i),
                            )
                            .map_err(|e| {
                                format!("failed to build GEP for repeat elem {}: {}", i, e)
                            })?
                    };
                    self.builder
                        .build_store(elem_ptr, elem_val)
                        .map_err(|e| format!("failed to store repeat elem {}: {}", i, e))?;
                }
                // Generate Index/IndexMut impls for this array type on demand
                self.ensure_slice_methods(&elem_ty, *count)?;
                let result = self
                    .builder
                    .build_load(array_llvm, alloca, "array_val")
                    .map_err(|e| format!("failed to load array: {}", e))?;
                Ok(result)
            }
            Expr::Index { array, index } => {
                let array_ty = self.expr_type(array);
                let (elem_ty, len) = match &array_ty {
                    Type::Array { inner, len } => (*inner.clone(), *len),
                    Type::Ref { inner, .. } => match inner.as_ref() {
                        Type::Array { inner: ai, len } => (*ai.clone(), *len),
                        _ => {
                            return Err(format!("cannot index into non-array type {:?}", array_ty));
                        }
                    },
                    _ => {
                        return Err(format!("cannot index into non-array type {:?}", array_ty));
                    }
                };
                self.ensure_slice_methods(&elem_ty, len)?;
                let key = Self::array_type_key(&elem_ty, len);
                let methods = self
                    .impl_methods
                    .get(&key)
                    .ok_or_else(|| format!("no methods registered for array type '{}'", key))?;
                let fn_name = methods
                    .iter()
                    .find(|(name, _)| name == "index")
                    .map(|(_, mangled)| mangled.clone())
                    .ok_or_else(|| format!("no 'index' method for array type '{}'", key))?;
                let fn_val = self
                    .module
                    .get_function(&fn_name)
                    .ok_or_else(|| format!("internal: index function '{}' not found", fn_name))?;
                // Compile array to a pointer — use original alloca if it's a variable
                let array_ptr = if let Expr::Ident(name) = array.as_ref() {
                    if let Some((ptr, _, _)) = self.symbols.get(name) {
                        *ptr
                    } else {
                        return Err(format!("undefined variable '{}'", name));
                    }
                } else {
                    let array_val = self.compile_expr(array)?;
                    match array_val {
                        BasicValueEnum::PointerValue(p) => p,
                        _ => {
                            let alloca = self
                                .builder
                                .build_alloca(array_val.get_type(), "array_ref")
                                .map_err(|e| {
                                    format!("failed to build alloca for array ref: {}", e)
                                })?;
                            self.builder
                                .build_store(alloca, array_val)
                                .map_err(|e| format!("failed to store array: {}", e))?;
                            alloca
                        }
                    }
                };
                let idx_val = self.compile_expr(index)?;
                let result = self
                    .builder
                    .build_call(fn_val, &[array_ptr.into(), idx_val.into()], "index_call")
                    .map_err(|e| format!("failed to call index: {}", e))?;
                // The result is a pointer to the element; dereference it
                let result_ptr = self.try_extract_result(result).into_pointer_value();
                let elem_llvm = self.type_to_llvm(&elem_ty);
                let loaded = self
                    .builder
                    .build_load(elem_llvm, result_ptr, "index_load")
                    .map_err(|e| format!("failed to load index result: {}", e))?;
                Ok(loaded)
            }
            Expr::Cast { expr, to_type } => {
                self.check_visibility_of_type(&self.current_module_path, to_type)?;
                let val = self.compile_expr(expr)?;
                self.emit_cast(val, to_type)
            }
            Expr::If {
                cond,
                then_block,
                else_ifs,
                else_block,
            } => {
                let parent_fn = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let cond_type = self.expr_type(cond);
                if cond_type != Type::Bool {
                    return Err(format!("if condition must be bool, found {:?}", cond_type));
                }
                let cond_val = self.compile_expr(cond)?;
                let cond_i1 = match cond_val {
                    BasicValueEnum::IntValue(v) => v,
                    _ => {
                        return Err("if condition must be a boolean".to_string());
                    }
                };

                // Determine result type from then/else blocks
                let result_type = then_block
                    .tail_expr
                    .as_ref()
                    .map(|e| self.expr_type(e))
                    .or_else(|| {
                        else_block
                            .as_ref()
                            .and_then(|b| b.tail_expr.as_ref().map(|e| self.expr_type(e)))
                    })
                    .or_else(|| {
                        else_ifs
                            .first()
                            .and_then(|(_, b)| b.tail_expr.as_ref().map(|e| self.expr_type(e)))
                    })
                    .unwrap_or(Type::I32);
                let result_llvm_ty = self.type_to_llvm(&result_type);

                // Allocate a stack slot for the if result
                let result_alloca = self
                    .builder
                    .build_alloca(result_llvm_ty, "if_result")
                    .map_err(|e| format!("failed to build if result alloca: {}", e))?;

                let then_bb = self.context.append_basic_block(parent_fn, "if_then");
                let else_bb = self.context.append_basic_block(parent_fn, "if_else");
                let merge_bb = self.context.append_basic_block(parent_fn, "if_merge");

                self.builder
                    .build_conditional_branch(cond_i1, then_bb, else_bb)
                    .map_err(|e| format!("failed to build if branch: {}", e))?;

                // Compile then block
                self.builder.position_at_end(then_bb);
                let saved_symbols = self.symbols.clone();
                if let Some(then_val) = self.compile_block_get_value(then_block)? {
                    self.builder
                        .build_store(result_alloca, then_val)
                        .map_err(|e| format!("failed to store then result: {}", e))?;
                }
                self.symbols = saved_symbols;
                let then_terminated = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_some();
                if !then_terminated {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| format!("failed to branch to merge: {}", e))?;
                }

                // Compile else_ifs and else block
                self.builder.position_at_end(else_bb);
                self.compile_else_chain_store(
                    else_ifs,
                    else_block,
                    parent_fn,
                    merge_bb,
                    result_alloca,
                    &result_type,
                )?;
                let else_terminated = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_some();
                if !else_terminated {
                    self.builder
                        .build_unconditional_branch(merge_bb)
                        .map_err(|e| format!("failed to branch to merge from else: {}", e))?;
                }

                // Position at merge block and load result
                self.builder.position_at_end(merge_bb);
                let result = self
                    .builder
                    .build_load(result_llvm_ty, result_alloca, "if_result")
                    .map_err(|e| format!("failed to load if result: {}", e))?;
                Ok(result)
            }
            Expr::Loop { body } => {
                let parent_fn = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let header_bb = self.context.append_basic_block(parent_fn, "loop_header");
                let after_bb = self.context.append_basic_block(parent_fn, "loop_after");

                self.builder
                    .build_unconditional_branch(header_bb)
                    .map_err(|e| format!("failed to branch to loop header: {}", e))?;

                self.builder.position_at_end(header_bb);
                let saved_symbols = self.symbols.clone();
                self.compile_block_get_value(body)?;
                self.symbols = saved_symbols;

                // Branch back to header
                let terminated = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_some();
                if !terminated {
                    self.builder
                        .build_unconditional_branch(header_bb)
                        .map_err(|e| format!("failed to branch back in loop: {}", e))?;
                }

                // Position at after block for subsequent code
                self.builder.position_at_end(after_bb);

                // Loop produces unit value
                let unit_ty = self.context.struct_type(&[], false);
                Ok(unit_ty.get_undef().into())
            }
            Expr::While { cond, body } => {
                let parent_fn = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_parent()
                    .unwrap();

                let cond_bb = self.context.append_basic_block(parent_fn, "while_cond");
                let body_bb = self.context.append_basic_block(parent_fn, "while_body");
                let after_bb = self.context.append_basic_block(parent_fn, "while_after");

                self.builder
                    .build_unconditional_branch(cond_bb)
                    .map_err(|e| format!("failed to branch to while cond: {}", e))?;

                // Condition block
                self.builder.position_at_end(cond_bb);
                let cond_type = self.expr_type(cond);
                if cond_type != Type::Bool {
                    return Err(format!(
                        "while condition must be bool, found {:?}",
                        cond_type
                    ));
                }
                let cond_val = self.compile_expr(cond)?;
                let cond_i1 = match cond_val {
                    BasicValueEnum::IntValue(v) => v,
                    _ => {
                        return Err("while condition must be a boolean".to_string());
                    }
                };
                self.builder
                    .build_conditional_branch(cond_i1, body_bb, after_bb)
                    .map_err(|e| format!("failed to build while branch: {}", e))?;

                // Body block
                self.builder.position_at_end(body_bb);
                let saved_symbols = self.symbols.clone();
                self.compile_block_get_value(body)?;
                self.symbols = saved_symbols;
                let terminated = self
                    .builder
                    .get_insert_block()
                    .unwrap()
                    .get_terminator()
                    .is_some();
                if !terminated {
                    self.builder
                        .build_unconditional_branch(cond_bb)
                        .map_err(|e| format!("failed to branch back in while: {}", e))?;
                }

                // Position at after block
                self.builder.position_at_end(after_bb);

                let unit_ty = self.context.struct_type(&[], false);
                Ok(unit_ty.get_undef().into())
            }
            Expr::UnaryNot(inner_expr) => {
                let val = self.compile_expr(inner_expr)?;
                match val {
                    BasicValueEnum::IntValue(v) => Ok(self
                        .builder
                        .build_not(v, "not")
                        .map_err(|e| format!("failed to build unary not: {}", e))?
                        .into()),
                    _ => Err("unary ! requires integer operand".to_string()),
                }
            }
            Expr::UnaryMinus(inner_expr) => {
                let val = self.compile_expr(inner_expr)?;
                match val {
                    BasicValueEnum::IntValue(v) => Ok(self
                        .builder
                        .build_int_neg(v, "neg")
                        .map_err(|e| format!("failed to build unary minus: {}", e))?
                        .into()),
                    BasicValueEnum::FloatValue(v) => Ok(self
                        .builder
                        .build_float_neg(v, "neg")
                        .map_err(|e| format!("failed to build unary minus: {}", e))?
                        .into()),
                    _ => Err("unary - requires numeric operand".to_string()),
                }
            }
            Expr::EnumLit {
                enum_name,
                variant,
                payload,
            } => {
                // Compute the actual type name. For generic enums with a payload,
                // derive from the payload type (e.g., Option::Some(42) → Option__i32).
                // Avoid monomorphized_names because it's a global map that gets
                // overwritten by recursive monomorphization.
                let actual_name = if let Some(payload_expr) = payload {
                    if self.generic_enum_defs.contains_key(enum_name.as_str()) {
                        let payload_ty = self.expr_type(payload_expr);
                        if Self::is_concrete_type(&payload_ty) {
                            let mangled = Self::mangle_generic_instance(enum_name, &[payload_ty]);
                            // If the mangled type exists, use it.
                            // Otherwise fall back to monomorphized_names
                            // (recursive monomorphization hasn't run yet).
                            if self.struct_types.contains_key(&mangled) {
                                mangled
                            } else {
                                self.monomorphized_names
                                    .get(enum_name.as_str())
                                    .cloned()
                                    .unwrap_or(mangled)
                            }
                        } else {
                            // Non-concrete payload (e.g., type param in generic fn).
                            // Fall back to monomorphized_names.
                            self.monomorphized_names
                                .get(enum_name.as_str())
                                .cloned()
                                .unwrap_or_else(|| enum_name.clone())
                        }
                    } else {
                        enum_name.clone()
                    }
                } else {
                    self.monomorphized_names
                        .get(enum_name.as_str())
                        .cloned()
                        .unwrap_or_else(|| enum_name.clone())
                };
                self.check_visibility_of_path(&self.current_module_path, &actual_name)?;
                let struct_type = self
                    .struct_types
                    .get(&actual_name)
                    .copied()
                    .ok_or_else(|| format!("unknown enum type '{}'", actual_name))?;
                let decl = self
                    .enum_defs
                    .get(&actual_name)
                    .ok_or_else(|| format!("unknown enum '{}'", actual_name))?;
                let variant_idx = decl
                    .variants
                    .iter()
                    .position(|v| v.name == *variant)
                    .ok_or_else(|| {
                        format!("unknown variant '{}' in enum '{}'", variant, actual_name)
                    })? as u32;
                let fields = self
                    .struct_fields
                    .get(&actual_name)
                    .ok_or_else(|| format!("unknown enum '{}'", actual_name))?;
                let payload_field_name = format!("__{}", variant);
                let mut payload_field_idx: Option<u32> = None;
                for (idx, field) in fields.iter().enumerate() {
                    if field.name == payload_field_name {
                        payload_field_idx = Some(idx as u32);
                        break;
                    }
                }
                let struct_val: inkwell::values::AggregateValueEnum<'ctx> =
                    struct_type.get_undef().into();
                let mut result = struct_val;
                result = self
                    .builder
                    .build_insert_value(
                        result,
                        self.context.i8_type().const_int(variant_idx as u64, false),
                        0,
                        "__tag",
                    )
                    .map_err(|e| format!("failed to insert enum tag: {}", e))?;
                if let Some(payload_expr) = payload
                    && let Some(field_idx) = payload_field_idx
                {
                    let payload_val = self.compile_expr(payload_expr)?;
                    result = self
                        .builder
                        .build_insert_value(result, payload_val, field_idx, &payload_field_name)
                        .map_err(|e| format!("failed to insert enum payload: {}", e))?;
                }
                match result {
                    inkwell::values::AggregateValueEnum::StructValue(sv) => Ok(sv.into()),
                    _ => Err(format!("expected struct value for enum '{}'", enum_name)),
                }
            }
            Expr::IfLet {
                pattern,
                scrutinee,
                then_block,
                else_block,
            } => self.compile_if_let(pattern, scrutinee, then_block, else_block),
            Expr::Match { scrutinee, arms } => self.compile_match(scrutinee, arms),
        }
    }

    /// Compile `if let` pattern matching.
    fn compile_if_let(
        &mut self,
        pattern: &Pattern,
        scrutinee: &Expr,
        then_block: &Block,
        else_block: &Option<Block>,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let scrutinee_val = self.compile_expr(scrutinee)?;
        let scrutinee_ty = self.expr_type(scrutinee);
        let result_type = then_block
            .tail_expr
            .as_ref()
            .map(|e| self.expr_type(e))
            .or_else(|| {
                else_block
                    .as_ref()
                    .and_then(|b| b.tail_expr.as_ref().map(|e| self.expr_type(e)))
            })
            .unwrap_or(Type::Unit);
        let result_llvm_ty = self.type_to_llvm(&result_type);

        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();

        // Allocate result slot
        let result_alloca = self
            .builder
            .build_alloca(result_llvm_ty, "if_let_result")
            .map_err(|e| format!("failed to build if_let result alloca: {}", e))?;

        // Generate pattern match check
        let (matches_val, bindings) =
            self.gen_pattern_check(pattern, scrutinee_val, &scrutinee_ty)?;

        let then_bb = self.context.append_basic_block(parent_fn, "if_let_then");
        let else_bb = self.context.append_basic_block(parent_fn, "if_let_else");
        let merge_bb = self.context.append_basic_block(parent_fn, "if_let_merge");

        self.builder
            .build_conditional_branch(matches_val, then_bb, else_bb)
            .map_err(|e| format!("failed to build if_let branch: {}", e))?;

        // === Then block: bind variables, execute body ===
        self.builder.position_at_end(then_bb);
        let saved = self.symbols.clone();
        for (name, ptr, ty) in &bindings {
            self.symbols.insert(name.clone(), (*ptr, false, ty.clone()));
        }
        if let Some(val) = self.compile_block_get_value(then_block)? {
            self.builder
                .build_store(result_alloca, val)
                .map_err(|e| format!("failed to store if_let then result: {}", e))?;
        }
        self.symbols = saved;
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            self.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| format!("failed to branch from if_let then: {}", e))?;
        }

        // === Else block ===
        self.builder.position_at_end(else_bb);
        let saved = self.symbols.clone();
        if let Some(el_block) = else_block
            && let Some(val) = self.compile_block_get_value(el_block)?
        {
            self.builder
                .build_store(result_alloca, val)
                .map_err(|e| format!("failed to store if_let else result: {}", e))?;
        }
        self.symbols = saved;
        if self
            .builder
            .get_insert_block()
            .unwrap()
            .get_terminator()
            .is_none()
        {
            self.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| format!("failed to branch from if_let else: {}", e))?;
        }

        // === Merge ===
        self.builder.position_at_end(merge_bb);
        if result_type == Type::Unit {
            let unit_ty = self.context.struct_type(&[], false);
            Ok(unit_ty.get_undef().into())
        } else {
            let result = self
                .builder
                .build_load(result_llvm_ty, result_alloca, "if_let_result")
                .map_err(|e| format!("failed to load if_let result: {}", e))?;
            Ok(result)
        }
    }

    /// Compile `match` expression with sequential arm checking.
    fn compile_match(
        &mut self,
        scrutinee: &Expr,
        arms: &[MatchArm],
    ) -> Result<BasicValueEnum<'ctx>, String> {
        let scrutinee_val = self.compile_expr(scrutinee)?;
        let scrutinee_ty = self.expr_type(scrutinee);
        let result_type = arms
            .first()
            .and_then(|arm| arm.body.tail_expr.as_ref())
            .map(|e| self.expr_type(e))
            .unwrap_or(Type::Unit);
        let result_llvm_ty = self.type_to_llvm(&result_type);

        let parent_fn = self
            .builder
            .get_insert_block()
            .unwrap()
            .get_parent()
            .unwrap();

        let result_alloca = self
            .builder
            .build_alloca(result_llvm_ty, "match_result")
            .map_err(|e| format!("failed to build match result alloca: {}", e))?;

        let merge_bb = self.context.append_basic_block(parent_fn, "match_merge");

        // Handle empty match (shouldn't happen but be safe)
        if arms.is_empty() {
            self.builder
                .build_unconditional_branch(merge_bb)
                .map_err(|e| format!("failed to branch to merge: {}", e))?;
            self.builder.position_at_end(merge_bb);
            let unit_ty = self.context.struct_type(&[], false);
            return Ok(unit_ty.get_undef().into());
        }

        // Pre-allocate all check blocks so we can branch to the first one
        let mut check_bbs: Vec<inkwell::basic_block::BasicBlock<'ctx>> = Vec::new();
        let mut body_bbs: Vec<inkwell::basic_block::BasicBlock<'ctx>> = Vec::new();
        for i in 0..arms.len() {
            check_bbs.push(
                self.context
                    .append_basic_block(parent_fn, &format!("match_check_{}", i)),
            );
            body_bbs.push(
                self.context
                    .append_basic_block(parent_fn, &format!("match_body_{}", i)),
            );
        }

        // Branch from current block to first check block
        self.builder
            .build_unconditional_branch(check_bbs[0])
            .map_err(|e| format!("failed to branch to match entry: {}", e))?;

        for (i, arm) in arms.iter().enumerate() {
            let is_last = i == arms.len() - 1;
            let check_bb = check_bbs[i];
            let body_bb = body_bbs[i];

            // Position at check block
            self.builder.position_at_end(check_bb);

            // Generate pattern match check
            let (mut matches_val, bindings) =
                self.gen_pattern_check(&arm.pattern, scrutinee_val, &scrutinee_ty)?;

            // Apply guard if present
            if let Some(ref guard) = arm.guard {
                let guard_val = self.compile_expr(guard)?;
                let guard_i1 = match guard_val {
                    BasicValueEnum::IntValue(v) => {
                        if v.get_type().get_bit_width() != 1 {
                            self.builder
                                .build_int_truncate(v, self.bool_type, "guard_i1")
                                .map_err(|e| format!("failed to trunc guard: {}", e))?
                        } else {
                            v
                        }
                    }
                    _ => {
                        return Err("match guard must be a boolean".to_string());
                    }
                };
                matches_val = self
                    .builder
                    .build_and(matches_val, guard_i1, "guard_and_match")
                    .map_err(|e| format!("failed to build guard and: {}", e))?;
            }

            if is_last {
                // Last arm: if matches → body, else → merge (no match = no value)
                self.builder
                    .build_conditional_branch(matches_val, body_bb, merge_bb)
                    .map_err(|e| format!("failed to build last match branch: {}", e))?;
            } else {
                let next_check = check_bbs[i + 1];
                self.builder
                    .build_conditional_branch(matches_val, body_bb, next_check)
                    .map_err(|e| format!("failed to build match branch: {}", e))?;
            }

            // Compile arm body
            self.builder.position_at_end(body_bb);
            let saved = self.symbols.clone();
            for (name, ptr, ty) in &bindings {
                self.symbols.insert(name.clone(), (*ptr, false, ty.clone()));
            }
            if let Some(val) = self.compile_block_get_value(&arm.body)? {
                self.builder
                    .build_store(result_alloca, val)
                    .map_err(|e| format!("failed to store match arm result: {}", e))?;
            }
            self.symbols = saved;
            if self
                .builder
                .get_insert_block()
                .unwrap()
                .get_terminator()
                .is_none()
            {
                self.builder
                    .build_unconditional_branch(merge_bb)
                    .map_err(|e| format!("failed to branch from match arm: {}", e))?;
            }
        }

        // === Merge ===
        self.builder.position_at_end(merge_bb);
        if result_type == Type::Unit {
            let unit_ty = self.context.struct_type(&[], false);
            Ok(unit_ty.get_undef().into())
        } else {
            let result = self
                .builder
                .build_load(result_llvm_ty, result_alloca, "match_result")
                .map_err(|e| format!("failed to load match result: {}", e))?;
            Ok(result)
        }
    }

    /// Generate LLVM IR for pattern matching against a value.
    /// Returns (matches: i1, bindings: Vec<(name, alloca_ptr, type)>).
    fn gen_pattern_check(
        &mut self,
        pattern: &Pattern,
        scrutinee_val: BasicValueEnum<'ctx>,
        scrutinee_ty: &Type,
    ) -> Result<
        (
            inkwell::values::IntValue<'ctx>,
            Vec<(String, inkwell::values::PointerValue<'ctx>, Type)>,
        ),
        String,
    > {
        match pattern {
            Pattern::Wildcard => Ok((self.bool_type.const_int(1, false), Vec::new())),
            Pattern::Binding(name) => {
                // Create alloca for the binding and store the scrutinee value
                let llvm_ty = self.type_to_llvm(scrutinee_ty);
                let alloca = self
                    .builder
                    .build_alloca(llvm_ty, &format!("bind_{}", name))
                    .map_err(|e| format!("failed to build binding alloca: {}", e))?;
                self.builder
                    .build_store(alloca, scrutinee_val)
                    .map_err(|e| format!("failed to store binding: {}", e))?;
                Ok((
                    self.bool_type.const_int(1, false),
                    vec![(name.clone(), alloca, scrutinee_ty.clone())],
                ))
            }
            Pattern::IntLit(n) => {
                // Compare scrutinee (must be integer) with literal
                let scrutinee_int = match scrutinee_val {
                    BasicValueEnum::IntValue(v) => v,
                    _ => return Err("integer pattern requires integer scrutinee".to_string()),
                };
                let lit_val = scrutinee_int.get_type().const_int(*n as u64, true);
                let cmp = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::EQ,
                        scrutinee_int,
                        lit_val,
                        "pat_lit_cmp",
                    )
                    .map_err(|e| format!("failed to build literal pattern cmp: {}", e))?;
                Ok((cmp, Vec::new()))
            }
            Pattern::BoolLit(b) => {
                let scrutinee_int = match scrutinee_val {
                    BasicValueEnum::IntValue(v) => v,
                    _ => return Err("bool pattern requires integer scrutinee".to_string()),
                };
                let lit_val = scrutinee_int
                    .get_type()
                    .const_int(if *b { 1 } else { 0 }, false);
                let cmp = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::EQ,
                        scrutinee_int,
                        lit_val,
                        "pat_bool_cmp",
                    )
                    .map_err(|e| format!("failed to build bool pattern cmp: {}", e))?;
                Ok((cmp, Vec::new()))
            }
            Pattern::EnumVariant {
                enum_name: _,
                variant,
                payload,
            } => {
                // Derive the concrete enum type name from the scrutinee type.
                let enum_type_name = match scrutinee_ty {
                    Type::Struct(name) => {
                        // For generic enums, the base name may need monomorphized lookup
                        self.monomorphized_names
                            .get(name.as_str())
                            .cloned()
                            .unwrap_or_else(|| name.clone())
                    }
                    Type::GenericInstance(name, args) => Self::mangle_generic_instance(name, args),
                    Type::Ref { inner, .. } => match inner.as_ref() {
                        Type::Struct(name) => self
                            .monomorphized_names
                            .get(name.as_str())
                            .cloned()
                            .unwrap_or_else(|| name.clone()),
                        Type::GenericInstance(name, args) => {
                            Self::mangle_generic_instance(name, args)
                        }
                        _ => {
                            return Err(format!("cannot pattern-match on type {:?}", scrutinee_ty));
                        }
                    },
                    _ => {
                        return Err(format!("cannot pattern-match on type {:?}", scrutinee_ty));
                    }
                };

                // Look up variant index from enum definition
                let decl = self
                    .enum_defs
                    .get(&enum_type_name)
                    .ok_or_else(|| format!("unknown enum '{}'", enum_type_name))?;
                let variant_idx = decl
                    .variants
                    .iter()
                    .position(|v| v.name == *variant)
                    .ok_or_else(|| {
                        format!("unknown variant '{}' in enum '{}'", variant, enum_type_name)
                    })? as u64;

                // Extract the tag field from the scrutinee
                let scrutinee_struct = match scrutinee_val {
                    BasicValueEnum::StructValue(sv) => sv,
                    _ => {
                        return Err("enum pattern matching requires struct scrutinee".to_string());
                    }
                };
                let tag_val = self
                    .builder
                    .build_extract_value(scrutinee_struct, 0, "enum_tag")
                    .map_err(|e| format!("failed to extract enum tag: {}", e))?;
                let tag_int = match tag_val {
                    BasicValueEnum::IntValue(v) => v,
                    _ => return Err("enum tag is not an integer".to_string()),
                };

                // Compare tag with variant index
                let tag_matches = self
                    .builder
                    .build_int_compare(
                        inkwell::IntPredicate::EQ,
                        tag_int,
                        self.context.i8_type().const_int(variant_idx, false),
                        "tag_cmp",
                    )
                    .map_err(|e| format!("failed to build tag compare: {}", e))?;

                let mut bindings = Vec::new();

                // If variant has payload, extract it and match inner pattern
                if let Some(inner_pattern) = payload {
                    // Find the payload field index and type (clone to end immutable borrow)
                    let (payload_field_idx, payload_ty) = {
                        let fields = self
                            .struct_fields
                            .get(&enum_type_name)
                            .ok_or_else(|| format!("unknown enum '{}'", enum_type_name))?;
                        let payload_field_name = format!("__{}", variant);
                        let idx = fields
                            .iter()
                            .position(|f| f.name == payload_field_name)
                            .ok_or_else(|| {
                                format!(
                                    "payload field '{}' not found in enum '{}'",
                                    payload_field_name, enum_type_name
                                )
                            })? as u32;
                        (idx, fields[idx as usize].ty.clone())
                    };

                    // Extract the payload value
                    let payload_val = self
                        .builder
                        .build_extract_value(
                            scrutinee_struct,
                            payload_field_idx,
                            &format!("__{}", variant),
                        )
                        .map_err(|e| format!("failed to extract enum payload: {}", e))?;

                    // Recursively match inner pattern against the payload
                    let (inner_matches, inner_bindings) =
                        self.gen_pattern_check(inner_pattern, payload_val, &payload_ty)?;

                    // Combine: tag matches AND inner pattern matches
                    let combined = self
                        .builder
                        .build_and(tag_matches, inner_matches, "pat_and")
                        .map_err(|e| format!("failed to build pattern and: {}", e))?;

                    bindings.extend(inner_bindings);
                    Ok((combined, bindings))
                } else {
                    // Unit variant: just tag comparison
                    Ok((tag_matches, bindings))
                }
            }
        }
    }

    /// Resolve a function name by trying direct lookup first, then overload map.
    fn resolve_function(
        &self,
        name: &str,
        arg_types: &[Type],
    ) -> Option<inkwell::values::FunctionValue<'ctx>> {
        let n_args = arg_types.len();
        // Try direct lookup first
        if let Some(fn_val) = self.module.get_function(name) {
            return Some(fn_val);
        }
        // Try overload map
        if let Some(overloads) = self.overloads.get(name) {
            // Filter by arg count
            let candidates: Vec<&(String, Vec<Type>)> = overloads
                .iter()
                .filter(|(_, params)| params.len() == n_args)
                .collect();
            if candidates.len() == 1 {
                return self.module.get_function(&candidates[0].0);
            }
            // Multiple candidates with same arg count: match by types
            for (mangled, param_types) in candidates {
                if param_types
                    .iter()
                    .zip(arg_types.iter())
                    .all(|(p, a)| p == a)
                {
                    return self.module.get_function(mangled);
                }
            }
        }
        None
    }

    /// Determine return type of builtin slice intrinsics (__builtin_slice_*).
    fn slice_intrinsic_return_type(&self, callee: &str, args: &[Expr]) -> Option<Type> {
        match callee {
            "__builtin_slice_index" if args.len() == 2 => {
                let self_ty = self.expr_type(&args[0]);
                if let Type::Ref { inner, .. } = &self_ty
                    && let Type::Array { inner: elem_ty, .. } = inner.as_ref()
                {
                    return Some(Type::Ref {
                        inner: elem_ty.clone(),
                        is_mut: false,
                    });
                }
                None
            }
            "__builtin_slice_index_mut" if args.len() == 2 => {
                let self_ty = self.expr_type(&args[0]);
                if let Type::Ref { inner, .. } = &self_ty
                    && let Type::Array { inner: elem_ty, .. } = inner.as_ref()
                {
                    return Some(Type::Ref {
                        inner: elem_ty.clone(),
                        is_mut: true,
                    });
                }
                None
            }
            "__builtin_slice_as_ptr" if args.len() == 1 => {
                let self_ty = self.expr_type(&args[0]);
                if let Type::Ref { inner, .. } = &self_ty
                    && let Type::Array { inner: elem_ty, .. } = inner.as_ref()
                {
                    return Some(Type::Ptr {
                        inner: elem_ty.clone(),
                        is_mut: false,
                    });
                }
                None
            }
            _ => None,
        }
    }

    /// Compile a builtin slice intrinsic call, returning the LLVM value.
    fn compile_slice_intrinsic(
        &mut self,
        callee: &str,
        args: &[Expr],
    ) -> Result<Option<BasicValueEnum<'ctx>>, String> {
        match callee {
            "__builtin_slice_index" if args.len() == 2 => {
                let self_val = self.compile_expr(&args[0])?;
                let idx_val = self.compile_expr(&args[1])?;
                let self_ptr = self_val.into_pointer_value();
                let idx = idx_val.into_int_value();

                // Get element type and length from self's type
                let self_ty = self.expr_type(&args[0]);
                let (elem_ty, len) = match &self_ty {
                    Type::Ref { inner, .. } => match inner.as_ref() {
                        Type::Array { inner, len } => (*inner.clone(), *len),
                        _ => {
                            return Err(format!(
                                "__builtin_slice_index: expected reference to array, got {:?}",
                                self_ty
                            ));
                        }
                    },
                    _ => {
                        return Err(format!(
                            "__builtin_slice_index: expected reference to array, got {:?}",
                            self_ty
                        ));
                    }
                };

                let elem_llvm = self.type_to_llvm(&elem_ty);
                let array_llvm: BasicTypeEnum<'ctx> = match elem_llvm {
                    BasicTypeEnum::IntType(it) => it.array_type(len as u32).into(),
                    BasicTypeEnum::FloatType(ft) => ft.array_type(len as u32).into(),
                    BasicTypeEnum::StructType(st) => st.array_type(len as u32).into(),
                    BasicTypeEnum::ArrayType(at) => at.array_type(len as u32).into(),
                    _ => panic!("unsupported array element type: {:?}", elem_llvm),
                };

                let zero = self.ptr_int_type.const_zero();
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(array_llvm, self_ptr, &[zero, idx], "slice_index")
                        .map_err(|e| format!("failed to build GEP for slice index: {}", e))?
                };
                Ok(Some(elem_ptr.into()))
            }
            "__builtin_slice_index_mut" if args.len() == 2 => {
                let self_val = self.compile_expr(&args[0])?;
                let idx_val = self.compile_expr(&args[1])?;
                let self_ptr = self_val.into_pointer_value();
                let idx = idx_val.into_int_value();

                let self_ty = self.expr_type(&args[0]);
                let (elem_ty, len) = match &self_ty {
                    Type::Ref { inner, .. } => match inner.as_ref() {
                        Type::Array { inner, len } => (*inner.clone(), *len),
                        _ => {
                            return Err(format!(
                                "__builtin_slice_index_mut: expected reference to array, got {:?}",
                                self_ty
                            ));
                        }
                    },
                    _ => {
                        return Err(format!(
                            "__builtin_slice_index_mut: expected reference to array, got {:?}",
                            self_ty
                        ));
                    }
                };

                let elem_llvm = self.type_to_llvm(&elem_ty);
                let array_llvm: BasicTypeEnum<'ctx> = match elem_llvm {
                    BasicTypeEnum::IntType(it) => it.array_type(len as u32).into(),
                    BasicTypeEnum::FloatType(ft) => ft.array_type(len as u32).into(),
                    BasicTypeEnum::StructType(st) => st.array_type(len as u32).into(),
                    BasicTypeEnum::ArrayType(at) => at.array_type(len as u32).into(),
                    _ => panic!("unsupported array element type: {:?}", elem_llvm),
                };

                let zero = self.ptr_int_type.const_zero();
                let elem_ptr = unsafe {
                    self.builder
                        .build_in_bounds_gep(array_llvm, self_ptr, &[zero, idx], "slice_index_mut")
                        .map_err(|e| format!("failed to build GEP for slice index_mut: {}", e))?
                };
                Ok(Some(elem_ptr.into()))
            }
            "__builtin_slice_as_ptr" if args.len() == 1 => {
                let self_val = self.compile_expr(&args[0])?;
                let self_ptr = self_val.into_pointer_value();
                let ptr_type = self.context.ptr_type(AddressSpace::default());
                let ptr = self
                    .builder
                    .build_bit_cast(self_ptr, ptr_type, "slice_as_ptr")
                    .map_err(|e| format!("failed to build bitcast for as_ptr: {}", e))?;
                Ok(Some(ptr))
            }
            _ => Ok(None),
        }
    }

    fn compile_args_vec(
        &mut self,
        args: &[Expr],
    ) -> Result<Vec<BasicMetadataValueEnum<'ctx>>, String> {
        let mut values = Vec::with_capacity(args.len());
        for arg in args {
            let val = self.compile_expr(arg)?;
            values.push(val.into());
        }
        Ok(values)
    }

    fn try_extract_result(
        &self,
        result: inkwell::values::CallSiteValue<'ctx>,
    ) -> BasicValueEnum<'ctx> {
        let any = result.as_any_value_enum();
        match any {
            inkwell::values::AnyValueEnum::IntValue(v) => v.into(),
            inkwell::values::AnyValueEnum::FloatValue(v) => v.into(),
            inkwell::values::AnyValueEnum::PointerValue(v) => v.into(),
            inkwell::values::AnyValueEnum::StructValue(v) => v.into(),
            _ => self.i32_type.const_zero().into(),
        }
    }

    fn const_eval(&self, expr: &Expr) -> Result<i64, String> {
        match expr {
            Expr::BoolLit(val) => Ok(if *val { 1 } else { 0 }),
            Expr::IntLit(val) => Ok(*val),
            Expr::FloatLit(_) => {
                Err("float literals are not supported in const initializers".to_string())
            }
            Expr::StrLit(_) => {
                Err("string literals are not supported in const initializers".to_string())
            }
            Expr::Ident(name) => self
                .consts
                .get(name)
                .copied()
                .ok_or_else(|| format!("undefined constant '{}'", name)),
            Expr::Binary { op, lhs, rhs } => {
                let lhs = self.const_eval(lhs)?;
                let rhs = self.const_eval(rhs)?;
                match op {
                    BinOp::Add => Ok(lhs + rhs),
                    BinOp::Sub => Ok(lhs - rhs),
                    BinOp::Mul => Ok(lhs * rhs),
                    BinOp::Div => {
                        if rhs == 0 {
                            Err("division by zero in const expression".to_string())
                        } else {
                            Ok(lhs / rhs)
                        }
                    }
                    BinOp::Eq => Ok(if lhs == rhs { 1 } else { 0 }),
                    BinOp::Neq => Ok(if lhs != rhs { 1 } else { 0 }),
                    BinOp::Lt => Ok(if lhs < rhs { 1 } else { 0 }),
                    BinOp::Gt => Ok(if lhs > rhs { 1 } else { 0 }),
                    BinOp::Le => Ok(if lhs <= rhs { 1 } else { 0 }),
                    BinOp::Ge => Ok(if lhs >= rhs { 1 } else { 0 }),
                    BinOp::And => Ok(if lhs != 0 && rhs != 0 { 1 } else { 0 }),
                    BinOp::Or => Ok(if lhs != 0 || rhs != 0 { 1 } else { 0 }),
                }
            }
            Expr::Ref { .. } => {
                Err("reference expressions are not supported in const initializers".to_string())
            }
            Expr::UnaryMinus(expr) => {
                let val = self.const_eval(expr)?;
                Ok(-val)
            }
            Expr::Deref(_) => {
                Err("dereference expressions are not supported in const initializers".to_string())
            }
            Expr::Assign { .. } => {
                Err("assignment is not supported in const initializers".to_string())
            }
            Expr::MethodCall { expr, method, args } => {
                if method == "len" && args.is_empty() {
                    if let Expr::StrLit(s) = expr.as_ref() {
                        Ok(s.len() as i64)
                    } else {
                        Err("non-constant receiver for method call in const initializer"
                            .to_string())
                    }
                } else {
                    Err("non-constant method call in const initializer".to_string())
                }
            }
            Expr::Cast { expr, .. } => self.const_eval(expr),
            _ => Err("non-constant expression in const initializer".to_string()),
        }
    }

    fn emit_cast(
        &self,
        val: BasicValueEnum<'ctx>,
        to_type: &Type,
    ) -> Result<BasicValueEnum<'ctx>, String> {
        // Handle bool destination: truncate int to i1, compare float != 0.0
        if *to_type == Type::Bool {
            match val {
                BasicValueEnum::IntValue(src_int) => {
                    let src_ty = src_int.get_type();
                    if src_ty.get_bit_width() == 1 {
                        // Already i1, no-op
                        return Ok(val);
                    }
                    return self
                        .builder
                        .build_int_truncate(src_int, self.bool_type, "cast")
                        .map(|v| v.into())
                        .map_err(|e| format!("int-to-bool trunc cast failed: {}", e));
                }
                BasicValueEnum::FloatValue(src_float) => {
                    // Float != 0.0 → true
                    let zero = self.f64_type.const_float(0.0);
                    let src_f = src_float.get_type();
                    let cmp = if src_f == self.f32_type {
                        let zero_f32 = self.f32_type.const_float(0.0);
                        self.builder
                            .build_float_compare(
                                inkwell::FloatPredicate::ONE,
                                src_float,
                                zero_f32,
                                "cast",
                            )
                            .map_err(|e| format!("float-to-bool cast failed: {}", e))?
                    } else {
                        self.builder
                            .build_float_compare(
                                inkwell::FloatPredicate::ONE,
                                src_float,
                                zero,
                                "cast",
                            )
                            .map_err(|e| format!("float-to-bool cast failed: {}", e))?
                    };
                    return Ok(cmp.into());
                }
                _ => {
                    return Err("unsupported value type for cast to bool".to_string());
                }
            }
        }

        // Handle pointer↔integer conversions
        if let BasicValueEnum::PointerValue(ptr_val) = val {
            let is_int_dest = matches!(
                to_type,
                Type::I8
                    | Type::I16
                    | Type::I32
                    | Type::I64
                    | Type::U8
                    | Type::U16
                    | Type::U32
                    | Type::U64
                    | Type::Isize
                    | Type::Usize
            );
            if is_int_dest {
                // Always ptrtoint to ptr_int_type first (LLVM requirement)
                let as_ptr_int = self
                    .builder
                    .build_ptr_to_int(ptr_val, self.ptr_int_type, "ptr_to_int")
                    .map_err(|e| format!("ptr-to-int cast failed: {}", e))?;
                let dst_width = match to_type {
                    Type::I8 | Type::U8 => 8,
                    Type::I16 | Type::U16 => 16,
                    Type::I32 | Type::U32 => 32,
                    Type::I64 | Type::U64 | Type::Isize | Type::Usize => 64,
                    _ => 64,
                };
                let ptr_width = 64u32;
                let result: BasicValueEnum<'ctx> = if dst_width < ptr_width {
                    self.builder
                        .build_int_truncate(
                            as_ptr_int,
                            match to_type {
                                Type::I8 | Type::U8 => self.i8_type,
                                Type::I16 | Type::U16 => self.i16_type,
                                Type::I32 | Type::U32 => self.i32_type,
                                _ => self.ptr_int_type,
                            },
                            "ptr_trunc",
                        )
                        .map(|v| v.into())
                        .map_err(|e| format!("ptr-to-int truncation failed: {}", e))?
                } else {
                    as_ptr_int.into()
                };
                return Ok(result);
            }
        }
        if matches!(to_type, Type::Ptr { .. }) {
            if let BasicValueEnum::IntValue(src_int) = val {
                let src_width = src_int.get_type().get_bit_width();
                let ptr_width = self.ptr_int_type.get_bit_width();
                let ptr_ty = self.context.ptr_type(AddressSpace::default());
                if src_width < ptr_width {
                    let extended = self
                        .builder
                        .build_int_z_extend(src_int, self.ptr_int_type, "int_to_ptr_ext")
                        .map_err(|e| format!("failed to extend int for ptr cast: {}", e))?;
                    return self
                        .builder
                        .build_int_to_ptr(extended, ptr_ty, "int_to_ptr")
                        .map(|v| v.into())
                        .map_err(|e| format!("int-to-ptr cast failed: {}", e));
                } else {
                    return self
                        .builder
                        .build_int_to_ptr(src_int, ptr_ty, "int_to_ptr")
                        .map(|v| v.into())
                        .map_err(|e| format!("int-to-ptr cast failed: {}", e));
                }
            }
            // Pointer → same pointer (identity)
            return Ok(val);
        }

        let is_dst_float = Self::is_float(to_type);
        let dst_signed = Self::is_signed(to_type);
        let dst_llvm = self.type_to_llvm(to_type);

        match val {
            BasicValueEnum::IntValue(src_int) => {
                let src_ty = src_int.get_type();
                let src_width = src_ty.get_bit_width();

                // Handle bool source (i1) → wider int: always zext (unsigned)
                if src_width == 1 && !is_dst_float {
                    let dst_int = dst_llvm.into_int_type();
                    return self
                        .builder
                        .build_int_z_extend(src_int, dst_int, "cast")
                        .map(|v| v.into())
                        .map_err(|e| format!("bool-to-int zext cast failed: {}", e));
                }
                // Handle bool source (i1) → float: zext to i32 first
                if src_width == 1 && is_dst_float {
                    let dst_f = match to_type {
                        Type::F32 => self.f32_type,
                        _ => self.f64_type,
                    };
                    let as_i32 = self
                        .builder
                        .build_int_z_extend(src_int, self.i32_type, "bool_to_i32")
                        .map_err(|e| format!("bool-to-i32 cast failed: {}", e))?;
                    return self
                        .builder
                        .build_unsigned_int_to_float(as_i32, dst_f, "cast")
                        .map(|v| v.into())
                        .map_err(|e| format!("bool-to-float cast failed: {}", e));
                }

                let dst_width = match to_type {
                    Type::I8 | Type::U8 => 8,
                    Type::I16 | Type::U16 => 16,
                    Type::I32 | Type::U32 => 32,
                    Type::I64 | Type::U64 | Type::Isize | Type::Usize => 64,
                    Type::F32 | Type::F64 => 0,
                    Type::Never => {
                        return Err("cast to never type is not allowed".to_string());
                    }
                    Type::Tuple(_) | Type::Unit => {
                        return Err("cannot cast to tuple or unit type".to_string());
                    }
                    Type::Str => {
                        return Err("cannot cast to string type".to_string());
                    }
                    Type::Ref { .. } => return Err("cannot cast to reference type".to_string()),
                    Type::Ptr { .. } => unreachable!(),
                    Type::Struct(_) | Type::GenericInstance(_, _) | Type::Alias(_, _) => {
                        return Err("cannot cast to struct type".to_string());
                    }
                    Type::Array { .. } => {
                        return Err("cannot cast to array type".to_string());
                    }
                    Type::Slice { .. } | Type::GenericArray { .. } => {
                        return Err("cannot cast to slice or generic array type".to_string());
                    }
                    Type::SelfType => return Err("cannot cast to Self type".to_string()),
                    // Bool caught by early return above, but keep for completeness
                    Type::Bool => 1,
                };

                if is_dst_float {
                    let dst_f = match to_type {
                        Type::F32 => self.f32_type,
                        _ => self.f64_type,
                    };
                    if src_ty == self.i32_type || dst_signed {
                        self.builder
                            .build_signed_int_to_float(src_int, dst_f, "cast")
                            .map(|v| v.into())
                            .map_err(|e| format!("int-to-float cast failed: {}", e))
                    } else {
                        self.builder
                            .build_unsigned_int_to_float(src_int, dst_f, "cast")
                            .map(|v| v.into())
                            .map_err(|e| format!("uint-to-float cast failed: {}", e))
                    }
                } else if src_width < dst_width {
                    if dst_signed {
                        self.builder
                            .build_int_s_extend(src_int, dst_llvm.into_int_type(), "cast")
                            .map(|v| v.into())
                            .map_err(|e| format!("int sext cast failed: {}", e))
                    } else {
                        self.builder
                            .build_int_z_extend(src_int, dst_llvm.into_int_type(), "cast")
                            .map(|v| v.into())
                            .map_err(|e| format!("int zext cast failed: {}", e))
                    }
                } else if src_width > dst_width {
                    self.builder
                        .build_int_truncate(src_int, dst_llvm.into_int_type(), "cast")
                        .map(|v| v.into())
                        .map_err(|e| format!("int trunc cast failed: {}", e))
                } else {
                    self.builder
                        .build_bit_cast(src_int, dst_llvm.into_int_type(), "cast")
                        .map_err(|e| format!("int bitcast failed: {}", e))
                }
            }
            BasicValueEnum::FloatValue(src_float) => {
                if is_dst_float {
                    let dst_f = match to_type {
                        Type::F32 => self.f32_type,
                        _ => self.f64_type,
                    };
                    let src_f = src_float.get_type();
                    if src_f == dst_f {
                        Ok(src_float.into())
                    } else if src_f == self.f32_type {
                        self.builder
                            .build_float_ext(src_float, self.f64_type, "cast")
                            .map(|v| v.into())
                            .map_err(|e| format!("float ext cast failed: {}", e))
                    } else {
                        self.builder
                            .build_float_trunc(src_float, self.f32_type, "cast")
                            .map(|v| v.into())
                            .map_err(|e| format!("float trunc cast failed: {}", e))
                    }
                } else {
                    let dst_int = dst_llvm.into_int_type();
                    if dst_signed {
                        self.builder
                            .build_float_to_signed_int(src_float, dst_int, "cast")
                            .map(|v| v.into())
                            .map_err(|e| format!("float-to-sint cast failed: {}", e))
                    } else {
                        self.builder
                            .build_float_to_unsigned_int(src_float, dst_int, "cast")
                            .map(|v| v.into())
                            .map_err(|e| format!("float-to-uint cast failed: {}", e))
                    }
                }
            }
            _ => Err(format!("unsupported value type for cast: val={:?}", val)),
        }
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
        let mut cg = CodeGen::new_jit(&context, OptimizationLevel::None)?;
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
    fn test_jit_logical_and_tt() {
        assert_eq!(
            jit("fn main() -> i32 { (true && true) as i32 }").unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_logical_and_tf() {
        assert_eq!(
            jit("fn main() -> i32 { (true && false) as i32 }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_logical_and_ft() {
        assert_eq!(
            jit("fn main() -> i32 { (false && true) as i32 }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_logical_and_ff() {
        assert_eq!(
            jit("fn main() -> i32 { (false && false) as i32 }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_logical_and_ints_t() {
        assert_eq!(jit("fn main() -> i32 { (1 && 2) as i32 }").unwrap(), 1);
    }

    #[test]
    fn test_jit_logical_and_ints_f() {
        assert_eq!(jit("fn main() -> i32 { (0 && 2) as i32 }").unwrap(), 0);
    }

    #[test]
    fn test_jit_logical_or_tt() {
        assert_eq!(
            jit("fn main() -> i32 { (true || true) as i32 }").unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_logical_or_tf() {
        assert_eq!(
            jit("fn main() -> i32 { (true || false) as i32 }").unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_logical_or_ft() {
        assert_eq!(
            jit("fn main() -> i32 { (false || true) as i32 }").unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_logical_or_ff() {
        assert_eq!(
            jit("fn main() -> i32 { (false || false) as i32 }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_logical_or_ints_t() {
        assert_eq!(jit("fn main() -> i32 { (0 || 2) as i32 }").unwrap(), 1);
    }

    #[test]
    fn test_jit_logical_or_ints_f() {
        assert_eq!(jit("fn main() -> i32 { (0 || 0) as i32 }").unwrap(), 0);
    }

    #[test]
    fn test_jit_logical_short_circuit_and() {
        // AND short-circuiting: false && mutate(...) => mutate should not run
        let src_and = "
            fn mutate(ptr: *mut i32) -> bool {
                *ptr = 42;
                true
            }
            fn main() -> i32 {
                let mut x = 0;
                let res = false && mutate(&mut x);
                x
            }
        ";
        assert_eq!(jit(src_and).unwrap(), 0);
    }

    #[test]
    fn test_jit_logical_short_circuit_or() {
        // OR short-circuiting: true || mutate(...) => mutate should not run
        let src_or = "
            fn mutate(ptr: *mut i32) -> bool {
                *ptr = 42;
                true
            }
            fn main() -> i32 {
                let mut x = 0;
                let res = true || mutate(&mut x);
                x
            }
        ";
        assert_eq!(jit(src_or).unwrap(), 0);
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
        let mut cg = CodeGen::new_native(&context, OptimizationLevel::None);
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
    fn test_jit_printf_extern() {
        assert_eq!(
            jit(
                r#"extern "C" { fn printf(fmt: *const i8, ...) -> i32; } fn main() { printf("%d".0, 42); }"#
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_mutable_var_assignment() {
        assert_eq!(jit("fn main() { let mut x = 10; x = 20; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_immutable_var_cannot_assign() {
        let result = jit("fn main() { let x = 10; x = 20; }");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("cannot assign to immutable"),
            "error should mention cannot assign to immutable"
        );
    }

    #[test]
    fn test_jit_assign_undefined_var() {
        let result = jit("fn main() { x = 10; }");
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("undefined variable"),
            "error should mention undefined variable"
        );
    }

    #[test]
    fn test_jit_const_declaration() {
        assert_eq!(jit("fn main() { const X = 42; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_const_use() {
        let result = jit("fn main() { const X = 42; let y = X; }");
        assert!(
            result.is_ok(),
            "using const in expression failed: {:?}",
            result
        );
    }

    #[test]
    fn test_jit_const_in_arithmetic() {
        let result = jit("fn main() { const BASE = 10; const SUM = BASE + 5; let x = SUM; }");
        assert!(result.is_ok(), "const arithmetic failed: {:?}", result);
    }

    #[test]
    fn test_jit_cannot_assign_to_const() {
        let result = jit("fn main() { const X = 42; X = 99; }");
        assert!(result.is_err());
        let err = result.unwrap_err();
        assert!(
            err.contains("cannot assign to constant"),
            "error should mention 'cannot assign to constant': {}",
            err
        );
    }

    #[test]
    fn test_jit_mutable_var_reassign_and_read() {
        assert_eq!(jit("fn main() { let mut x = 5; x = x + 1; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_multiple_mutable_vars() {
        assert_eq!(
            jit("fn main() { let mut a = 1; let mut b = 2; a = 3; b = a; }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_assignment_inside_let_init() {
        let result = jit("fn main() { let mut x = 0; let y = x = 5; }");
        assert!(result.is_ok(), "assignment in let init: {:?}", result);
    }

    #[test]
    fn test_compile_module_const() {
        let src = "fn f() { const X = 42; }";
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
        let mut cg = CodeGen::new_native(&context, OptimizationLevel::None);
        assert!(cg.compile_module(&program).is_ok());
    }

    #[test]
    fn test_jit_typed_let() {
        assert_eq!(jit("fn main() { let x: i32 = 42; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_float_literal() {
        assert_eq!(jit("fn main() { let x: f64 = 3.14; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_cast_int_to_i64() {
        assert_eq!(jit("fn main() { let x: i64 = 42 as i64; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_cast_int_to_u8() {
        assert_eq!(jit("fn main() { let x: u8 = 300 as u8; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_chain_cast() {
        assert_eq!(
            jit("fn main() { let x: u8 = 42 as i64 as u8; }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_float_add_as_i32() {
        assert_eq!(
            jit("fn main() { let x: i32 = (3.14 as i32) + 1; }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_u8_type() {
        assert_eq!(jit("fn main() { let x: u8 = 200; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_i16_type() {
        assert_eq!(jit("fn main() { let x: i16 = 42; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_u64_type() {
        assert_eq!(jit("fn main() { let x: u64 = 42; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_f32_type() {
        assert_eq!(jit("fn main() { let x: f32 = 1.5; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_float_to_int_cast() {
        assert_eq!(jit("fn main() { let x: i32 = 3.99 as i32; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_int_to_f64_cast() {
        assert_eq!(jit("fn main() { let x: f64 = 42 as f64; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_usize_type() {
        assert_eq!(jit("fn main() { let x: usize = 42; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_isize_type() {
        assert_eq!(jit("fn main() { let x: isize = 42; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_suffix_int_i32() {
        assert_eq!(jit("fn main() { let x = 42i32; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_suffix_int_u8() {
        assert_eq!(jit("fn main() { let x = 255u8; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_suffix_int_i64() {
        assert_eq!(jit("fn main() { let x = 1000i64; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_suffix_float_f64() {
        assert_eq!(jit("fn main() { let x = 3.14f64; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_suffix_float_f32() {
        assert_eq!(jit("fn main() { let x = 1.5f32; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_suffix_in_arithmetic() {
        assert_eq!(jit("fn main() { let x = 10i64 + 20i32; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_string_literal() {
        assert_eq!(
            jit(r#"extern "C" { fn printf(fmt: *const i8, ...) -> i32; } fn main() { printf("%s".0, "hello".0); }"#)
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_compile_module_typed() {
        let src = "fn f() { let x: f64 = 3.14; }";
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
        let mut cg = CodeGen::new_native(&context, OptimizationLevel::None);
        assert!(cg.compile_module(&program).is_ok());
    }

    #[test]
    fn test_jit_ref_deref_i32() {
        assert_eq!(
            jit("fn main() { let x = 42; let r = &x; let v = *r; }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_mut_ref_deref_assign() {
        assert_eq!(
            jit("fn main() { let mut x = 10; let r = &mut x; *r = 20; }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_ref_to_temp_expr() {
        assert_eq!(jit("fn main() { let r = &(1+2); let v = *r; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_deref_assign_through_mut_ref() {
        let result = jit("fn main() { let mut x = 5; let r = &mut x; *r = *r + 1; }");
        assert!(
            result.is_ok(),
            "deref assign through mut ref failed: {:?}",
            result
        );
    }

    #[test]
    fn test_jit_str_literal_type() {
        // String literals compile successfully as &str
        assert_eq!(
            jit(r#"extern "C" { fn printf(fmt: *const i8, ...) -> i32; } fn main() { printf("%s".0, "hello".0); }"#)
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_str_len_literal() {
        // "hello".len() should compile and execute
        assert!(jit("fn main() { \"hello\".len(); }").is_ok());
    }

    #[test]
    fn test_jit_str_len_variable() {
        assert!(jit("fn main() { let s = \"hello\"; s.len(); }").is_ok());
    }

    #[test]
    fn test_jit_struct_create() {
        assert_eq!(
            jit("struct Point { x: i32, y: i32, }\nfn main() { let p = Point { x: 10, y: 20 }; }")
                .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_struct_method_call() {
        assert_eq!(
            jit(
                "struct Point { x: i32, y: i32, }\nimpl Point {\n    fn new(x: i32, y: i32) -> Point { Point { x: x, y: y }; }\n    fn area(&self) -> i32 { self.x * self.y; }\n}\nfn main() {\n    let p = Point { x: 3, y: 4 };\n    p.area();\n}\n"
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_option_pattern_matching() {
        assert_eq!(
            jit("
                enum Option<T> { Some(T), None, }
                impl<T> Option<T> {
                    fn unwrap_or(&self, default: T) -> T {
                        match *self {
                            Option::Some(val) => val,
                            Option::None => default,
                        }
                    }
                }
                fn main() -> i32 {
                    let opt = Option::Some(42);
                    opt.unwrap_or(0)
                }
            ")
            .unwrap(),
            42
        );
        assert_eq!(
            jit("
                enum Option<T> { Some(T), None, }
                impl<T> Option<T> {
                    fn unwrap_or(&self, default: T) -> T {
                        match *self {
                            Option::Some(val) => val,
                            Option::None => default,
                        }
                    }
                }
                fn main() -> i32 {
                    let opt: Option<i32> = Option::None;
                    opt.unwrap_or(42)
                }
            ")
            .unwrap(),
            42
        );
    }

    #[test]
    fn test_jit_result_pattern_matching() {
        assert_eq!(
            jit("
                enum Result<T, E> { Ok(T), Err(E), }
                impl<T, E> Result<T, E> {
                    fn is_ok(&self) -> bool {
                        match *self {
                            Result::Ok(_) => true,
                            Result::Err(_) => false,
                        }
                    }
                }
                fn main() -> i32 {
                    let res: Result<i32, i32> = Result::Ok(100);
                    if res.is_ok() { 42 } else { 0 }
                }
            ")
            .unwrap(),
            42
        );
    }

    #[test]
    fn test_jit_builtin_default() {
        assert_eq!(jit("fn main() { i32::default(); }").unwrap(), 0);
    }

    #[test]
    fn test_jit_static_method() {
        assert_eq!(
            jit("fn main() { let x: i32 = i32::default(); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_builtin_clone() {
        assert_eq!(
            jit("fn main() { let x: i32 = 42; let y: i32 = x.clone(); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_builtin_eq() {
        assert_eq!(
            jit("fn main() { let a: i32 = 42; let b: i32 = 42; a.eq(&b); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_builtin_ne() {
        assert_eq!(
            jit("fn main() { let a: i32 = 42; let b: i32 = 43; a.ne(&b); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_builtin_cmp() {
        assert_eq!(
            jit("fn main() { let a: i32 = 42; let b: i32 = 43; a.cmp(&b); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_builtin_cmp_partial_cmp_removed() {
        // partial_cmp was removed, but cmp still works
    }

    #[test]
    fn test_jit_derive_default() {
        assert_eq!(
            jit(
                "#[derive(Default)]\nstruct Point { x: i32, y: i32, }\nfn main() { let p = Point { x: 10, y: 20 }; p.default(); }"
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_derive_clone() {
        assert_eq!(
            jit(
                "#[derive(Clone)]\nstruct Point { x: i32, y: i32, }\nfn main() { let p = Point { x: 10, y: 20 }; let c = p.clone(); }"
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_derive_eq() {
        assert_eq!(
            jit(
                "#[derive(Eq)]\nstruct Point { x: i32, y: i32, }\nfn main() { let a = Point { x: 10, y: 20 }; let b = Point { x: 10, y: 20 }; let r = a.eq(&b); }"
            )
            .unwrap(),
            0
        );
        assert_eq!(
            jit(
                "#[derive(Eq)]\nstruct Point { x: i32, y: i32, }\nfn main() { let a = Point { x: 10, y: 20 }; let b = Point { x: 99, y: 20 }; let r = a.ne(&b); }"
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_derive_ord() {
        assert_eq!(
            jit(
                "#[derive(Ord)]\nstruct Point { x: i32, y: i32, }\nfn main() { let a = Point { x: 10, y: 20 }; let b = Point { x: 5, y: 30 }; let r = a.cmp(&b); }"
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_derive_multi_trait() {
        assert_eq!(
            jit(
                "#[derive(Default, Clone, Eq)]\nstruct Point { x: i32, y: i32, }\nfn main() { let p = Point { x: 10, y: 20 }; let c = p.clone(); c.eq(&p); }"
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_derive_default_zeroed() {
        let result = jit(
            "#[derive(Default)]\nstruct Point { x: i32, y: i32, }\nfn main() { let p = Point { x: 10, y: 20 }; p.default(); }",
        );
        assert!(result.is_ok());
    }

    #[test]
    fn test_jit_derive_non_default_field_fails() {
        // Testing that it errors at compile time.
        // We need to check the error message too.
        let result = jit(
            "struct NonDefault { x: i32, }\n#[derive(Default)]\nstruct Wrapper { f: NonDefault, }\nfn main() {}",
        );
        assert!(result.is_err());
        assert!(
            result.unwrap_err().contains("does not implement Default"),
            "error should mention field doesn't implement Default"
        );
    }

    #[test]
    fn test_jit_bool_decl() {
        assert_eq!(
            jit("fn main() { let x: bool = true; let y: bool = false; }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_bool_eq() {
        assert_eq!(
            jit("fn main() { let a: bool = true; let b: bool = true; a.eq(&b); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_bool_ne() {
        assert_eq!(
            jit("fn main() { let a: bool = true; let b: bool = false; a.ne(&b); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_bool_cmp() {
        assert_eq!(
            jit("fn main() { let a: bool = true; let b: bool = false; a.cmp(&b); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_bool_default() {
        assert_eq!(
            jit("fn main() { let x: bool = true; x.default(); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_bool_clone() {
        assert_eq!(
            jit("fn main() { let x: bool = true; let y: bool = x.clone(); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_i32_eq_returns_bool() {
        assert_eq!(
            jit("fn main() { let a: i32 = 42; let b: i32 = 42; a.eq(&b); }").unwrap(),
            0
        );
        assert_eq!(
            jit("fn main() { let a: i32 = 42; let b: i32 = 43; a.ne(&b); }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_bool_in_struct() {
        assert_eq!(
            jit(
                "#[derive(Eq)]\nstruct Pair { a: bool, b: bool, }\nfn main() { let p = Pair { a: true, b: false }; let q = Pair { a: true, b: false }; p.eq(&q); }"
            )
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_bool_cast_to_i32() {
        assert_eq!(jit("fn main() { let x: i32 = true as i32; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_i32_cast_to_bool() {
        assert_eq!(jit("fn main() { let x: bool = 1 as bool; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_comparison_eq() {
        assert_eq!(jit("fn main() { 1 == 1; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_comparison_neq() {
        assert_eq!(jit("fn main() { 1 != 2; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_comparison_lt() {
        assert_eq!(jit("fn main() { 1 < 2; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_comparison_gt() {
        assert_eq!(jit("fn main() { 2 > 1; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_comparison_le() {
        assert_eq!(jit("fn main() { 1 <= 2; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_comparison_ge() {
        assert_eq!(jit("fn main() { 2 >= 2; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_if_as_stmt() {
        assert_eq!(jit("fn main() { if true { 2 } else { 3 }; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_if_else_as_stmt() {
        // if as expression statement with ; — result discarded, main returns 0
        assert_eq!(jit("fn main() { if false { 2 } else { 3 }; }").unwrap(), 0);
    }

    #[test]
    fn test_jit_if_else_if() {
        // if as expression statement with ;
        assert_eq!(
            jit("fn main() { if false { 1 } else if true { 2 } else { 3 }; }").unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_loop_infinite() {
        // loop with return to escape
        assert_eq!(jit("fn main() { return 42; }").unwrap(), 42);
    }

    #[test]
    fn test_jit_while_as_stmt() {
        assert_eq!(jit("fn main() { while false { 1 } }").unwrap(), 0);
    }

    #[test]
    fn test_jit_explicit_return() {
        assert_eq!(jit("fn main() -> i32 { return 42; }").unwrap(), 42);
    }

    #[test]
    fn test_jit_implicit_return() {
        assert_eq!(jit("fn main() -> i32 { 42 }").unwrap(), 42);
    }

    #[test]
    fn test_jit_return_in_if() {
        assert_eq!(
            jit("fn main() -> i32 { if true { return 42; } else { return 0; }; 0 }").unwrap(),
            42
        );
    }

    #[test]
    fn test_jit_while_with_if_break() {
        // while with a return inside
        assert_eq!(
            jit("fn main() -> i32 { while true { return 42; } }").unwrap(),
            42
        );
    }

    #[test]
    fn test_jit_nested_if() {
        // nested if as expression statement with ;
        assert_eq!(
            jit("fn main() { if true { if false { 1 } else { 2 } }; }").unwrap(),
            0
        );
    }

    const STRING_MALLOC: &str = r#"extern "C" {
    fn malloc(size: usize) -> *mut u8;
    fn realloc(ptr: *mut u8, new_size: usize) -> *mut u8;
    fn free(ptr: *mut u8);
    fn memcpy(dest: *mut u8, src: *const u8, n: usize) -> *mut u8;
}"#;

    const STRING_STRUCT: &str = r#"struct String {
    ptr: *mut u8,
    len: usize,
    cap: usize,
}"#;

    const STRING_IMPL: &str = concat!(
        "impl String {\n",
        "    fn new() -> String {\n",
        "        String { ptr: 0 as *mut u8, len: 0, cap: 0 }\n",
        "    }\n",
        "    fn with_capacity(cap: usize) -> String {\n",
        "        let ptr = if cap == 0 { 0 as *mut u8 } else { malloc(cap) };\n",
        "        String { ptr: ptr, len: 0, cap: cap }\n",
        "    }\n",
        "    fn len(&self) -> usize {\n",
        "        self.len\n",
        "    }\n",
        "    fn capacity(&self) -> usize {\n",
        "        self.cap\n",
        "    }\n",
        "    fn is_empty(&self) -> bool {\n",
        "        self.len == 0\n",
        "    }\n",
        "    fn clear(&mut self) {\n",
        "        self.len = 0;\n",
        "    }\n",
        "    fn push_str(&mut self, s: &str) {\n",
        "        let new_len = self.len + s.1;\n",
        "        if new_len > self.cap {\n",
        "            let ptr = malloc(new_len);\n",
        "            memcpy(ptr, self.ptr, self.len);\n",
        "            if self.cap > 0 {\n",
        "                free(self.ptr);\n",
        "            };\n",
        "            self.ptr = ptr;\n",
        "            self.cap = new_len;\n",
        "        };\n",
        "        memcpy((self.ptr as usize + self.len) as *mut u8, s.0, s.1);\n",
        "        self.len = new_len;\n",
        "    }\n",
        "    fn drop(&mut self) {\n",
        "        if self.cap > 0 {\n",
        "            free(self.ptr);\n",
        "            self.ptr = 0 as *mut u8;\n",
        "            self.len = 0;\n",
        "            self.cap = 0;\n",
        "        };\n",
        "    }\n",
        "}\n",
    );

    fn string_prog(body: &str) -> String {
        format!(
            r#"{} {} {} fn main() {{ {} }}"#,
            STRING_MALLOC, STRING_STRUCT, STRING_IMPL, body
        )
    }

    #[test]
    fn test_jit_string_new() {
        assert_eq!(jit(&string_prog("let s = String::new();")).unwrap(), 0);
    }

    #[test]
    fn test_jit_string_with_capacity() {
        assert_eq!(
            jit(&string_prog("let s = String::with_capacity(10);")).unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_string_len_capacity() {
        assert_eq!(
            jit(&string_prog(
                "let s = String::new(); s.len(); s.capacity();"
            ))
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_string_push_str() {
        assert_eq!(
            jit(&string_prog(
                "let mut s = String::new(); s.push_str(\"hello\");"
            ))
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_string_clear() {
        assert_eq!(
            jit(&string_prog(
                "let mut s = String::with_capacity(10); s.clear();"
            ))
            .unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_string_is_empty() {
        assert_eq!(
            jit(&string_prog("let s = String::new(); s.is_empty();")).unwrap(),
            0
        );
    }

    #[test]
    fn test_jit_same_scope_shadowing() {
        // Same-scope shadowing: let x = 1; let x = x + 1; → x is 2
        assert_eq!(
            jit("fn main() -> i32 { let x = 1; let x = x + 1; x }").unwrap(),
            2
        );
    }

    #[test]
    fn test_jit_if_block_shadowing_restores_outer() {
        // Nested if-block shadowing: outer x must be restored after the block
        assert_eq!(
            jit("fn main() -> i32 { let x = 1; if true { let x = 2; }; x }").unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_loop_block_shadowing_restores_outer() {
        // Nested loop-block shadowing
        assert_eq!(
            jit("fn main() -> i32 { loop { let x = 2; return x; }; 0 }").unwrap(),
            2
        );
    }

    #[test]
    fn test_jit_while_block_shadowing_restores_outer() {
        // Nested while-block shadowing
        assert_eq!(
            jit("fn main() -> i32 { let x = 1; while false { let x = 2; }; x }").unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_if_else_block_shadowing_restores_outer() {
        // Shadowing in else block
        assert_eq!(
            jit("fn main() -> i32 { let x = 1; if false { let x = 2; } else { let x = 3; }; x }")
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_else_if_block_shadowing_restores_outer() {
        // Shadowing in else-if block
        assert_eq!(
            jit("fn main() -> i32 { let x = 1; if false { } else if true { let x = 4; }; x }")
                .unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_shadowing_changes_type() {
        // Shadowing can change type (like Rust)
        assert_eq!(
            jit("fn main() -> i32 { let x = 1; let x = true; x as i32 }").unwrap(),
            1
        );
    }

    // --- Array slice method tests ---

    #[test]
    fn test_jit_array_len() {
        // a.len() returns the compile-time length
        assert_eq!(
            jit("fn main() -> usize { let a = [10, 20, 30]; a.len() }").unwrap(),
            3
        );
    }

    #[test]
    fn test_jit_array_len_zero_size() {
        // [0; 0] is rejected by parser, but repeat with positive len works
        assert_eq!(
            jit("fn main() -> usize { let a = [0; 1]; a.len() }").unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_array_len_large() {
        assert_eq!(
            jit("fn main() -> usize { let a = [0; 100]; a.len() }").unwrap(),
            100
        );
    }

    #[test]
    fn test_jit_array_len_different_elem_types() {
        // len() works regardless of element type
        assert_eq!(
            jit("fn main() -> usize { let a = [true, false, true]; a.len() }").unwrap(),
            3
        );
    }

    #[test]
    fn test_jit_array_index_read() {
        // a[i] reads element via slice index intrinsic
        assert_eq!(
            jit("fn main() -> i32 { let a = [10, 20, 30]; a[1] }").unwrap(),
            20
        );
    }

    #[test]
    fn test_jit_array_index_first() {
        assert_eq!(
            jit("fn main() -> i32 { let a = [7, 8, 9]; a[0] }").unwrap(),
            7
        );
    }

    #[test]
    fn test_jit_array_index_last() {
        assert_eq!(
            jit("fn main() -> i32 { let a = [1, 2, 3, 4]; a[3] }").unwrap(),
            4
        );
    }

    #[test]
    fn test_jit_array_index_mut_write() {
        // a[i] = v mutates element via slice index_mut intrinsic
        assert_eq!(
            jit("fn main() -> i32 { let mut a = [10, 20, 30]; a[1] = 99; a[1] }").unwrap(),
            99
        );
    }

    #[test]
    fn test_jit_array_index_mut_then_read_other() {
        assert_eq!(
            jit("fn main() -> i32 { let mut a = [1, 2, 3]; a[2] = 42; a[0] }").unwrap(),
            1
        );
    }

    #[test]
    fn test_jit_nested_array_index() {
        assert_eq!(
            jit("fn main() -> i32 { let a = [[1, 2], [3, 4]]; a[1][0] }").unwrap(),
            3
        );
    }

    #[test]
    fn test_jit_array_float_elements() {
        // Arrays of floats work
        assert_eq!(
            jit("fn main() -> usize { let a = [1.0, 2.0, 3.0]; a.len() }").unwrap(),
            3
        );
    }
}
