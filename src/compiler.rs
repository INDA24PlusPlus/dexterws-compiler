use std::{collections::HashMap, path::{Path, PathBuf}};

use crate::{parsing::{self, Ast, ItemKind}, tokenizing};

pub struct Compiler {
    files: HashMap<PathBuf, String>,
    file_ids: Vec<PathBuf>,
}

impl Compiler {
    pub fn new() -> Compiler {
        Compiler {
            files: HashMap::new(),
            file_ids: Vec::new(),
        }
    }

    pub fn parse_file(&mut self, path: &Path) -> Result<Option<Ast>, String> {
        let file = std::fs::read_to_string(path);
        match file {
            Ok(file) => {
                let inserted = self.files.insert(path.to_path_buf(), file);
                if inserted.is_some() {
                    return Ok(None)
                }
                self.file_ids.push(path.to_path_buf());
            },
            Err(_) => return Err(format!("could not read file: {}", path.to_str().unwrap())),
        }
        let lexer = tokenizing::Lexer::new(self.files.get(path).unwrap());
        let tokens = lexer.collect::<Vec<_>>();
        let file_id = self.file_ids.len() - 1;
        let parser = parsing::Parser::new(tokens.into_iter(), file_id);
        let mut ast = match parser.parse() {
            Ok(ast) => ast,
            Err(err) => return Err(err.to_string(self.files.get(path).unwrap())),
        };
        let mut merge_ast = Ast { nodes: Vec::new() };
        for item in &ast.nodes {
            match &item.kind {
                ItemKind::Import(p) => {
                    let current_dir = path.parent().unwrap();
                    let new_path = current_dir.join(p);
                    let new_ast = self.parse_file(&new_path)?;
                    if let Some(new_ast) = new_ast {
                        merge_ast.nodes.extend(new_ast.nodes);
                    }
                }
                _ => (),
            }
        }
        ast.nodes.extend(merge_ast.nodes);
        Ok(Some(ast))
    }

    pub fn compile_file(&mut self, path: &Path) -> Result<(), String> {
        let ast = self.parse_file(path)?;
        if let Some(ast) = ast {
            let mut semantical = crate::semantical::SemanticAnalyzer::new();
            let file_name = path.file_name().unwrap().to_str().unwrap();
            let semantical_result = semantical.analyze_ast(ast, file_name);
            let new_ast = match semantical_result {
                Ok(ast) => ast,
                Err(err) => {
                    let file_id = err.file_id();
                    let file = self.files.get(&self.file_ids[file_id as usize]).unwrap();
                    return Err(err.to_string(file));
                }
            };
            let llvm_context = inkwell::context::Context::create();
            let llvm_module = llvm_context.create_module("main");
            let mut codegen = crate::codegen::CodeGen::new(&llvm_context, llvm_module);
            codegen.generate(new_ast);
            codegen.spit_out();
            codegen.verify();
            codegen.create_binary(file_name);
        }
        Ok(())
    }
}