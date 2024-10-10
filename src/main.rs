fn main() {
    let file_name = std::env::args().nth(1).expect("no file name given");
    let file = std::fs::read_to_string(file_name).expect("could not read file");
    let lexer = dexterws_compiler::tokenizing::Lexer::new(&file);
    let tokens = lexer.collect::<Vec<_>>();
    let parser = dexterws_compiler::parsing::Parser::new(tokens.into_iter());
    for stmt in parser {
        println!("{:?}", stmt);
    }
}
