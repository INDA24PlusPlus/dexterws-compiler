use std::collections::HashMap;

use crate::{
    chainmap::ChainMap,
    parsing::{
        self, Assignment, Ast, BinOp, BinOpKind, Block, Expr, ExprKind, Function, FunctionSignature, Identifier, ItemKind, LiteralKind, NodeSpan, Statement, StatementKind, Type, UnaryOpKind
    }, tokenizing::TokenLocation,
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
    structs: (),
}

pub struct SemanticAnalyzer {
    variables: ChainMap<String, Type>,
    function_variables: Vec<Variable>,
    functions: HashMap<String, FunctionSignature>,
    loop_depth: usize,
    current_function: Option<String>,
}

const BUILTINS: &[(&str, Type, Type)] = &[
    ("iprint", Type::Int, Type::Int),
    ("fprint", Type::Float, Type::Int),
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
        expr: Expr,
    },
}

type SemanticResult<T> = Result<T, SemanticError>;

impl SemanticAnalyzer {
    pub fn new() -> SemanticAnalyzer {
        SemanticAnalyzer {
            variables: ChainMap::new(),
            function_variables: vec![],
            functions: HashMap::new(),
            loop_depth: 0,
            current_function: None,
        }
    }

    pub fn analyze_ast(&mut self, ast: Ast, name: &str) -> SemanticResult<Module> {
        let mut module = Module {
            name: name.to_string(),
            functions: vec![],
            structs: (),
        };
        
        self.declare_builtins();
        self.declare_functions(&ast)?;

        for item in ast.nodes {
            match item.kind {
                ItemKind::Function(func) => {
                    let extended_func = self.analyze_function(func)?;
                    module.functions.push(extended_func);
                }
            }
        }

        Ok(module)
    }

    fn declare_builtins(&mut self) {
        for (name, arg_ty, ret_ty) in BUILTINS {
            let sig = FunctionSignature {
                name: Identifier {
                    value: name.to_string(),
                    span: NodeSpan {
                        start: TokenLocation { line_span: (0, 0), col_span: (0, 0) },
                        end: TokenLocation { line_span: (0, 0), col_span: (0, 0) },
                    },
                },
                args: vec![("arg".to_string(), *arg_ty)],
                ret_ty: *ret_ty,
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
                ty: arg.1,
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

    fn analyze_statement(&mut self, stmt: &Statement) -> SemanticResult<()> {
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
            }
            StatementKind::Return(_) => self.analyze_return(stmt)?,
        }
        Ok(())
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
                expected: *expected,
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

        self.variables.insert(var.name.clone(), var.ty);
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
                    expected: func.ret_ty,
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
            },
            ExprKind::Var(var) => {
                if let Some(var) = self.variables.get(&var.value) {
                    Ok(*var)
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
                                expected: expected.1,
                                found: arg_ty,
                                expr: arg.clone(),
                            });
                        }
                    }
                    Ok(func.ret_ty)
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
            UnaryOpKind::Cast(ty) => Ok(*ty),
            _ => Ok(rhs_ty),
        }
    }
}
