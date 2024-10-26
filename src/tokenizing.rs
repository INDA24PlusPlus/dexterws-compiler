use std::{iter::Peekable, str::Chars};

pub struct Lexer<'a> {
    data: Peekable<Chars<'a>>,
    line: u32,
    column: u32,
}

#[derive(Debug, Clone, PartialEq)]
pub enum SymbolKind {
    Plus,
    Minus,
    Star,
    Slash,
    Percent,
    At,
    Walrus,
    AssignEq,
    ColonColon,
    Semicolon,

    Equals,
    Neq,
    Lt,
    Gt,
    Leq,
    Geq,
    And,
    Or,
    Not,

    LParen,
    RParen,
    LBrace,
    RBrace,
    Comma,
    RArrow,
    Colon,
}

impl SymbolKind {
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "+" => Some(Self::Plus),
            "-" => Some(Self::Minus),
            "*" => Some(Self::Star),
            "/" => Some(Self::Slash),
            "%" => Some(Self::Percent),
            "@" => Some(Self::At),
            ":=" => Some(Self::Walrus),
            "::" => Some(Self::ColonColon),
            "=" => Some(Self::AssignEq),
            ";" => Some(Self::Semicolon),
            "==" => Some(Self::Equals),
            "!=" => Some(Self::Neq),
            "<" => Some(Self::Lt),
            ">" => Some(Self::Gt),
            "<=" => Some(Self::Leq),
            ">=" => Some(Self::Geq),
            "&&" => Some(Self::And),
            "||" => Some(Self::Or),
            "!" => Some(Self::Not),
            "(" => Some(Self::LParen),
            ")" => Some(Self::RParen),
            "{" => Some(Self::LBrace),
            "}" => Some(Self::RBrace),
            "," => Some(Self::Comma),
            "->" => Some(Self::RArrow),
            ":" => Some(Self::Colon),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum NumberKind {
    Int(i64),
    Float(f64),
}

#[derive(Debug, Clone, PartialEq)]
pub enum TokenKind {
    Number(NumberKind),
    Ident(String),
    Symbol(SymbolKind),
}

#[derive(Debug, Copy, Clone, PartialEq)]
pub struct TokenLocation {
    pub line_span: (u32, u32), 
    pub col_span: (u32, u32),
}

#[derive(Debug, Clone)]
pub struct Token {
    pub kind: TokenKind,
    pub location: TokenLocation,
}

impl<'a> Lexer<'a> {
    pub fn new(input: &'a str) -> Self {
        Self {
            data: input.chars().peekable(),
            line: 1,
            column: 1,
        }
    }

    pub fn peek(&mut self) -> Option<char> {
        self.data.peek().copied()
    }

    pub fn eat(&mut self) -> Option<char> {
        let c = self.data.next();
        if c == Some('\n') {
            self.line += 1;
            self.column = 1;
        } else {
            self.column += 1;
        }
        c
    }

    // Parses something until a predicate is true
    pub fn eat_until<F>(&mut self, mut predicate: F) -> (String, TokenLocation)
    where
        F: FnMut(char) -> bool,
    {
        let mut value = String::new();
        let start = (self.line, self.column);
        while let Some(c) = self.peek() {
            if predicate(c) {
                break;
            }
            value.push(c);
            self.eat();
        }
        let end = (self.line, self.column);
        (value, TokenLocation { line_span: (start.0, end.0), col_span: (start.1, end.1) })
    }

    pub fn eat_whitespace(&mut self) {
        self.eat_until(|c| !c.is_whitespace());
    }

    pub fn eat_number(&mut self) -> Token {
        let mut seen_dot = false;
        let (value, loc) = self.eat_until(|c| {
            if c == '.' {
                if seen_dot {
                    return true;
                }
                seen_dot = true;
                return false;
            }
            !c.is_ascii_digit()
        });
        let number = if seen_dot {
            NumberKind::Float(value.parse().unwrap())
        } else {
            NumberKind::Int(value.parse().unwrap())
        };
        Token {
            kind: TokenKind::Number(number),
            location: loc,
        }
    }

    pub fn eat_ident(&mut self) -> Token {
        let (value, loc) = self.eat_until(|c| !c.is_ascii_alphanumeric());
        Token {
            kind: TokenKind::Ident(value),
            location: loc,
        }
    }

    pub fn eat_symbol(&mut self) -> Token {
        let mut count = 0;
        let mut should_break = false;
        let (value, loc) = self.eat_until(|c| {
            count += 1;
            if should_break {
                return true;
            }
            if count > 2 {
                return true;
            }
            match c {
                ':' | '=' | '!' | '<' | '>' | '&' | '|' | '-' => (),
                '+' | '*' | '/' | '(' | ')' | ';' | '{' | '}' | ',' | '%' | '@' => {
                    should_break = true;
                }
                _ => return true,
            }
            false
        });
        let kind = SymbolKind::from_str(&value).unwrap();
        Token {
            kind: TokenKind::Symbol(kind),
            location: loc,
        }
    }
}

impl<'a> Iterator for Lexer<'a> {
    type Item = Token;

    fn next(&mut self) -> Option<Self::Item> {
        self.eat_whitespace();
        match self.peek() {
            Some(c) => {
                let token = match c {
                    '0'..='9' => self.eat_number(),
                    'a'..='z' | 'A'..='Z' => self.eat_ident(),
                    _ => self.eat_symbol(),
                };
                Some(token)
            }
            None => None,
        }
    }
}
