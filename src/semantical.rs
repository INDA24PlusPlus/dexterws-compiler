use std::collections::HashMap;

use crate::{
    chainmap::ChainMap,
    parsing::{
        self, Assignment, Ast, BinOpKind, Block, Expr, ExprKind, Function, FunctionSignature, Identifier, ItemKind, LiteralKind, NodeSpan, Statement, StatementKind, Struct, Type, UnaryOpKind
    },
    tokenizing::TokenLocation,
};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Variable {
    pub name: String,
    pub ty: Type,
}

#[derive(Debug, Clone, PartialEq)]
pub struct ExtendedFunction {
    pub inner: Function,
    pub variables: Vec<Variable>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Module {
    pub name: String,
    pub functions: Vec<ExtendedFunction>,
    pub structs: HashMap<String, Struct>,
}

pub struct SemanticAnalyzer {
    variables: ChainMap<String, Type>,
    function_variables: Vec<Variable>,
    functions: HashMap<String, FunctionSignature>,
    structs: HashMap<String, Struct>,
    loop_depth: usize,
    current_function: Option<String>,
}

const BUILTINS: &[(&str, Type, Type)] = &[
    ("iprint", Type::Int, Type::Int),
    ("fprint", Type::Float, Type::Int),
    ("bprint", Type::Bool, Type::Int),
    ("printf", Type::String, Type::Int),
];

#[derive(Debug, Clone, PartialEq)]
pub enum SemanticError {
    ExpressionTypeMismatch {
        expected: Type,
        found: Type,
        expr: Expr,
    },
    FunctionNotFound {
        name: String,
        expr: Expr,
    },
    ArgumentCountMismatch {
        expected: usize,
        found: usize,
        expr: Expr,
    },
    ArgumentTypeMismatch {
        expected: Type,
        found: Type,
        expr: Expr,
    },
    BreakOutsideOfLoop {
        statement: Statement,
    },
    DoubleVariableDeclaration {
        name: Identifier,
        expr: Expr,
    },
    DoubleFunctionDeclaration {
        name: Identifier,
    },
    DoubleStructDeclaration {
        name: Identifier,
    },
    VariableNotFound {
        name: Identifier,
        expr: Expr,
    },
    ReturnOutsideOfFunction {
        stmt: Statement,
    },
    ReturnTypeMismatch {
        expected: Type,
        found: Type,
        statement: Statement,
    },
    UnsupportedBinOp {
        ty: Type,
        expr: Expr,
    },
    UnreachableCode {
        statement: Statement,
    },
}

impl SemanticError {
    pub fn to_string(self, source: &str) -> String {
        let mut string = String::new();
        let lines = source.lines().collect::<Vec<_>>();
        let span = match &self {
            SemanticError::ExpressionTypeMismatch { expr, .. } => expr.span,
            SemanticError::FunctionNotFound { expr, .. } => expr.span,
            SemanticError::ArgumentCountMismatch { expr, .. } => expr.span,
            SemanticError::ArgumentTypeMismatch { expr, .. } => expr.span,
            SemanticError::BreakOutsideOfLoop { statement } => statement.span,
            SemanticError::DoubleVariableDeclaration { expr, .. } => expr.span,
            SemanticError::DoubleFunctionDeclaration { name } => name.span,
            SemanticError::DoubleStructDeclaration { name } => name.span,
            SemanticError::VariableNotFound { expr, .. } => expr.span,
            SemanticError::ReturnOutsideOfFunction { stmt } => stmt.span,
            SemanticError::ReturnTypeMismatch { statement, .. } => statement.span,
            SemanticError::UnsupportedBinOp { expr, .. } => expr.span,
            SemanticError::UnreachableCode { statement } => statement.span,
        };

        let error_text = match &self {
            SemanticError::ExpressionTypeMismatch {
                expected, found, ..
            } => {
                format!(
                    "invalid expression type, expected {}, found {}",
                    expected, found
                )
            }
            SemanticError::FunctionNotFound { name, .. } => {
                format!("function '{}' is not defined", name)
            }
            SemanticError::ArgumentCountMismatch {
                expected, found, ..
            } => {
                format!(
                    "invalid argument count, expected {} arguments, found {}",
                    expected, found
                )
            }
            SemanticError::ArgumentTypeMismatch {
                expected, found, ..
            } => {
                format!(
                    "invalid argument type, expected {}, found {}",
                    expected, found
                )
            }
            SemanticError::BreakOutsideOfLoop { .. } => "cannot break outside of loop".to_string(),
            SemanticError::DoubleVariableDeclaration { name, .. } => {
                format!("variable '{}' already declared", name.value)
            }
            SemanticError::DoubleFunctionDeclaration { name } => {
                format!("function '{}' already declared", name.value)
            }
            SemanticError::DoubleStructDeclaration { name } => {
                format!("struct '{}' already declared", name.value)
            }
            SemanticError::VariableNotFound { name, .. } => {
                format!("variable '{}' is not defined", name.value)
            }
            SemanticError::ReturnOutsideOfFunction { .. } => {
                "cannot return outside of function".to_string()
            }
            SemanticError::ReturnTypeMismatch {
                expected, found, ..
            } => {
                format!(
                    "invalid return type, expected {}, found {}",
                    expected, found
                )
            }
            SemanticError::UnsupportedBinOp { ty, .. } => {
                format!("unsupported binary operation for type {}", ty)
            }
            SemanticError::UnreachableCode { .. } => "this code is unreachable".to_string(),
        };

        string.push_str(&format!("\x1b[31merror\x1b[0m: {}\n", error_text));
        let first_line = span.start.line_span.0 - 1;
        let last_line = span.end.line_span.0 - 1;
        let pre_first_line = first_line.checked_sub(1);
        let post_last_line = last_line + 1;
        let max_line_num_len = post_last_line.to_string().len();

        if let Some(idx) = pre_first_line {
            let line = lines[idx as usize];
            let line_num = idx + 1;
            let line_num_str = line_num.to_string();
            let padding = max_line_num_len - line_num_str.len();
            string.push_str(&format!(
                " \x1b[34m{}{} | \x1b[0m",
                " ".repeat(padding),
                line_num_str
            ));
            string.push_str(line);
            string.push('\n');
        }

        for i in first_line..=last_line {
            let line = lines[i as usize];
            let line_num = i + 1;
            let line_num_str = line_num.to_string();
            let padding = max_line_num_len - line_num_str.len();
            let col_start = if i == first_line {
                span.start.col_span.0 - 1
            } else {
                0
            };
            let col_end = if i == last_line {
                span.end.col_span.1 - 1
            } else {
                line.len() as u32
            };
            let mut caret = String::new();
            for _ in 0..col_start {
                caret.push(' ');
            }
            for _ in col_start..col_end {
                caret.push('^');
            }
            string.push_str(&format!(
                " \x1b[34m{}{} | \x1b[0m",
                " ".repeat(padding),
                line_num_str
            ));
            string.push_str(line);
            string.push('\n');
            string.push_str(&format!(
                " \x1b[34m{} | \x1b[0m",
                " ".repeat(max_line_num_len)
            ));
            string.push_str(&format!("\x1b[31m{}\x1b[0m\n", caret));
        }

        if post_last_line < lines.len() as u32 {
            let line = lines[post_last_line as usize];
            let line_num = post_last_line + 1;
            let line_num_str = line_num.to_string();
            let padding = max_line_num_len - line_num_str.len();
            string.push_str(&format!(
                " \x1b[34m{}{} | \x1b[0m",
                " ".repeat(padding),
                line_num_str
            ));
            string.push_str(line);
            string.push('\n');
        }

        string
    }
}

type SemanticResult<T> = Result<T, SemanticError>;

impl SemanticAnalyzer {
    pub fn new() -> SemanticAnalyzer {
        SemanticAnalyzer {
            variables: ChainMap::new(),
            function_variables: vec![],
            functions: HashMap::new(),
            structs: HashMap::new(),
            loop_depth: 0,
            current_function: None,
        }
    }

    pub fn analyze_ast(&mut self, ast: Ast, name: &str) -> SemanticResult<Module> {
        let mut module = Module {
            name: name.to_string(),
            functions: vec![],
            structs: HashMap::new(),
        };
        
        self.declare_builtins();
        self.declare_structs(&ast)?;
        self.declare_functions(&ast)?;


        for item in ast.nodes {
            match item.kind {
                ItemKind::Function(func) => {
                    let extended_func = self.analyze_function(func)?;
                    module.functions.push(extended_func);
                }
                ItemKind::Import(_) => {}
                ItemKind::Struct(_) => {}
            }
        }
        module.structs = self.structs.clone();

        Ok(module)
    }

    fn declare_builtins(&mut self) {
        for (name, arg_ty, ret_ty) in BUILTINS {
            let sig = FunctionSignature {
                name: Identifier {
                    value: name.to_string(),
                    span: NodeSpan {
                        start: TokenLocation {
                            line_span: (0, 0),
                            col_span: (0, 0),
                        },
                        end: TokenLocation {
                            line_span: (0, 0),
                            col_span: (0, 0),
                        },
                    },
                },
                args: vec![("arg".to_string(), arg_ty.clone())],
                ret_ty: ret_ty.clone(),
            };
            self.functions.insert(sig.name.value.clone(), sig);
        }
    }

    fn declare_functions(&mut self, ast: &Ast) -> SemanticResult<()> {
        for item in &ast.nodes {
            match &item.kind {
                ItemKind::Function(func) => {
                    if self.functions.contains_key(&func.sig.name.value) {
                        return Err(SemanticError::DoubleFunctionDeclaration {
                            name: func.sig.name.clone(),
                        });
                    }
                    self.functions
                        .insert(func.sig.name.value.clone(), func.sig.clone());
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn declare_structs(&mut self, ast: &Ast) -> SemanticResult<()> {
        for item in &ast.nodes {
            match &item.kind {
                ItemKind::Struct(struct_) => {
                    if self.structs.contains_key(&struct_.name.value) {
                        return Err(SemanticError::DoubleStructDeclaration {
                            name: struct_.name.clone(),
                        });
                    }
                    let name = struct_.name.value.clone();
                    self.structs
                        .insert(name, struct_.clone());
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn analyze_function(&mut self, func: Function) -> SemanticResult<ExtendedFunction> {
        self.function_variables.clear();
        self.variables.push();
        for arg in &func.sig.args {
            let var = Variable {
                name: arg.0.clone(),
                ty: arg.1.clone(),
            };
            self.variables.insert(var.name.clone(), var.ty).unwrap();
        }

        self.current_function = Some(func.sig.name.value.clone());
        self.analyze_body(&func.body)?;
        self.current_function = None;
        self.variables.pop();

        let mut extended_func = ExtendedFunction {
            inner: func,
            variables: vec![],
        };

        extended_func.variables.append(&mut self.function_variables);

        Ok(extended_func)
    }

    fn analyze_statement(&mut self, stmt: &Statement) -> SemanticResult<bool> {
        match &stmt.kind {
            StatementKind::Expr(expr) => {
                self.analyze_expr(expr)?;
            }
            StatementKind::Decl(decl) => self.analyze_decl(decl)?,
            StatementKind::Assign(assign) => self.analyze_assignment(assign)?,
            StatementKind::If { cond, then, or } => self.analyze_if(cond, then, or)?,
            StatementKind::Loop { body } => self.analyze_loop(body)?,
            StatementKind::Break => {
                if self.loop_depth == 0 {
                    return Err(SemanticError::BreakOutsideOfLoop {
                        statement: stmt.clone(),
                    });
                }
                return Ok(true);
            }
            StatementKind::Return(_) => {
                self.analyze_return(stmt)?;
                return Ok(true);
            }
        }
        Ok(false)
    }

    fn analyze_body(&mut self, body: &Block) -> SemanticResult<()> {
        self.variables.push();

        for stmt in &body.stmts {
            self.analyze_statement(stmt)?;
        }

        self.variables.pop();
        Ok(())
    }

    fn analyze_assignment(&mut self, assign: &Assignment) -> SemanticResult<()> {
        let expr_result = self.analyze_expr(&assign.value)?;
        let expected = if let Some(ty) = self.variables.get(&assign.name.value) {
            ty
        } else {
            return Err(SemanticError::VariableNotFound {
                name: assign.name.clone(),
                expr: assign.value.clone(),
            });
        };
        if expr_result != *expected {
            return Err(SemanticError::ExpressionTypeMismatch {
                expected: expected.clone(),
                found: expr_result,
                expr: assign.value.clone(),
            });
        }
        Ok(())
    }

    fn analyze_decl(&mut self, decl: &Assignment) -> SemanticResult<()> {
        if self.variables.contains_key(&decl.name.value) {
            return Err(SemanticError::DoubleVariableDeclaration {
                name: decl.name.clone(),
                expr: decl.value.clone(),
            });
        }
        let expr_result = self.analyze_expr(&decl.value)?;
        let var = Variable {
            name: decl.name.value.clone(),
            ty: expr_result,
        };

        self.variables.insert(var.name.clone(), var.ty.clone());
        self.function_variables.push(var);
        Ok(())
    }

    fn analyze_if(&mut self, cond: &Expr, then: &Block, or: &Option<Block>) -> SemanticResult<()> {
        let cond_ty = self.analyze_expr(cond)?;
        if cond_ty != Type::Bool {
            return Err(SemanticError::ExpressionTypeMismatch {
                expected: Type::Bool,
                found: cond_ty,
                expr: cond.clone(),
            });
        }
        self.analyze_body(then)?;
        if let Some(or) = or {
            self.analyze_body(or)?;
        }
        Ok(())
    }

    fn analyze_loop(&mut self, body: &Block) -> SemanticResult<()> {
        self.loop_depth += 1;
        self.analyze_body(body)?;
        self.loop_depth -= 1;
        Ok(())
    }

    fn analyze_return(&mut self, statement: &Statement) -> SemanticResult<()> {
        let expr = match &statement.kind {
            StatementKind::Return(expr) => expr,
            _ => unreachable!("This function should only be called with an expression statement"),
        };

        let ret_ty = if let Some(expr) = expr {
            self.analyze_expr(expr)?
        } else {
            Type::Void
        };

        if let Some(func) = self
            .current_function
            .as_ref()
            .and_then(|name| self.functions.get(name))
        {
            if ret_ty != func.ret_ty {
                return Err(SemanticError::ReturnTypeMismatch {
                    expected: func.ret_ty.clone(),
                    found: ret_ty,
                    statement: statement.clone(),
                });
            }
        } else {
            return Err(SemanticError::ReturnOutsideOfFunction {
                stmt: statement.clone(),
            });
        }
        Ok(())
    }

    fn analyze_expr(&self, expr: &Expr) -> SemanticResult<Type> {
        match &expr.kind {
            ExprKind::Literal(lit_kind) => match lit_kind {
                LiteralKind::Int(_) => Ok(Type::Int),
                LiteralKind::Float(_) => Ok(Type::Float),
                LiteralKind::Bool(_) => Ok(Type::Bool),
                LiteralKind::String(_) => Ok(Type::String),
            },
            ExprKind::Var(var) => {
                if let Some(var) = self.variables.get(&var.value) {
                    Ok(var.clone())
                } else {
                    Err(SemanticError::VariableNotFound {
                        name: var.clone(),
                        expr: expr.clone(),
                    })
                }
            }
            ExprKind::BinOp(_) => self.analyze_binop(expr),
            ExprKind::UnaryOp { .. } => self.analyze_unaryop(expr),
            ExprKind::Call { name, args } => {
                if let Some(func) = self.functions.get(name) {
                    if func.args.len() != args.len() {
                        panic!("argument count mismatch")
                    }
                    for (arg, expected) in args.iter().zip(func.args.iter()) {
                        let arg_ty = self.analyze_expr(arg)?;
                        if arg_ty != expected.1 {
                            return Err(SemanticError::ArgumentTypeMismatch {
                                expected: expected.1.clone(),
                                found: arg_ty,
                                expr: arg.clone(),
                            });
                        }
                    }
                    Ok(func.ret_ty.clone())
                } else {
                    Err(SemanticError::FunctionNotFound {
                        name: name.clone(),
                        expr: expr.clone(),
                    })
                }
            }
        }
    }

    fn analyze_binop(&self, expr: &Expr) -> SemanticResult<Type> {
        let binop = match expr {
            Expr {
                kind: ExprKind::BinOp(binop),
                ..
            } => binop,
            _ => unreachable!("This function should only be called with a binary operation"),
        };
        let lhs_ty = self.analyze_expr(&binop.lhs)?;
        let rhs_ty = self.analyze_expr(&binop.rhs)?;
        if lhs_ty != rhs_ty {
            return Err(SemanticError::ExpressionTypeMismatch {
                expected: lhs_ty,
                found: rhs_ty,
                expr: *binop.rhs.clone(),
            });
        }

        if lhs_ty == Type::Float && (binop.kind == BinOpKind::And || binop.kind == BinOpKind::Or) {
            return Err(SemanticError::UnsupportedBinOp {
                ty: Type::Float,
                expr: expr.clone(),
            });
        }

        let return_ty = match binop.kind {
            parsing::BinOpKind::Add
            | parsing::BinOpKind::Sub
            | parsing::BinOpKind::Mul
            | parsing::BinOpKind::Div
            | parsing::BinOpKind::Mod => lhs_ty,
            parsing::BinOpKind::Eq
            | parsing::BinOpKind::Neq
            | parsing::BinOpKind::Lt
            | parsing::BinOpKind::Leq
            | parsing::BinOpKind::Gt
            | parsing::BinOpKind::Geq => Type::Bool,
            parsing::BinOpKind::And | parsing::BinOpKind::Or => Type::Bool,
        };
        Ok(return_ty)
    }

    fn analyze_unaryop(&self, expr: &Expr) -> SemanticResult<Type> {
        let (kind, rhs) = match expr {
            Expr {
                kind: ExprKind::UnaryOp { kind, rhs },
                ..
            } => (kind, rhs),
            _ => unreachable!("This function should only be called with a unary operation"),
        };
        let rhs_ty = self.analyze_expr(rhs)?;
        match kind {
            UnaryOpKind::Cast(ty) => Ok(ty.clone()),
            _ => Ok(rhs_ty),
        }
    }
}
