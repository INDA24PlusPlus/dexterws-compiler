use dexterws_compiler::{parsing::StatementKind, tokenizing::Token};

fn unexpected_token(file: &str, token: Token) {
    let line_pos = token.location.line_span.0 - 1;
    let col_pos = token.location.col_span.0 - 1;
    let lines = file.lines().collect::<Vec<_>>();
    let line_num = line_pos + 1;
    let col_num = col_pos + 1;
    eprintln!(
        "\x1b[31merror\x1b[0m: unexpected token at {}:{}",
        line_num, col_num
    );
    let line_nums = [line_num - 1, line_num, line_num + 1]
        .iter()
        .map(|n| n.to_string())
        .collect::<Vec<_>>();
    let max_len = line_nums.iter().map(|s| s.len()).max().unwrap();
    let padding = max_len - line_nums[0].len();
    let idx = (line_pos as usize).checked_sub(1);
    if let Some(idx) = idx {
        eprint!(" \x1b[34m{}{} | \x1b[0m", " ".repeat(padding), line_nums[0]);
        eprintln!("{}", lines[idx]);
    }
    let padding = max_len - line_nums[1].len();
    eprint!(" \x1b[34m{}{} | \x1b[0m", " ".repeat(padding), line_nums[1]);
    for (i, c) in lines[line_pos as usize].chars().enumerate() {
        if i >= col_pos as usize && i < token.location.col_span.1 as usize - 1 {
            eprint!("\x1b[31m{}\x1b[0m", c);
        } else {
            eprint!("{}", c);
        }
    }
    eprintln!();
    let mut caret = String::new();
    for _ in 0..col_pos {
        caret.push(' ');
    }
    for _ in col_pos..token.location.col_span.1 - 1 {
        caret.push('^');
    }
    eprint!(" \x1b[34m{} | \x1b[0m", " ".repeat(max_len));
    eprintln!("\x1b[31m{}\x1b[0m", caret);
    let padding = max_len - line_nums[2].len();
    // Lol not really needed, who has a 4 billion line file?
    let idx = line_pos as usize + 1;
    if idx < lines.len() {
        eprint!(" \x1b[34m{}{} | \x1b[0m", " ".repeat(padding), line_nums[2]);
        eprintln!("{}", lines[idx]);
    }
}

fn main() {
    let input = std::env::args().nth(1).expect("no file name given");
    let path = std::path::Path::new(&input);
    let file = std::fs::read_to_string(path).expect("could not read file");
    let lexer = dexterws_compiler::tokenizing::Lexer::new(&file);
    let tokens = lexer.collect::<Vec<_>>();
    let parser = dexterws_compiler::parsing::Parser::new(tokens.into_iter());
    let ast = parser.parse().unwrap();
    let mut semantical = dexterws_compiler::semantical::SemanticAnalyzer::new();
    let file_name = path.file_name().unwrap().to_str().unwrap();
    let semantical_result = semantical.analyze_ast(ast, file_name);
    let new_ast = match semantical_result {
        Ok(ast) => ast,
        Err(err) => {
            println!("{}", err.to_string(&file));
            return;
        }
    };
    let llvm_context = inkwell::context::Context::create();
    let llvm_module = llvm_context.create_module("main");
    let mut codegen = dexterws_compiler::codegen::CodeGen::new(&llvm_context, llvm_module);
    codegen.generate(new_ast);
    codegen.spit_out();
    codegen.verify();
    codegen.spit_out_object(file_name);
}
