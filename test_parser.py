"""
Script de test pour le parser NORMiL
"""

import sys
sys.path.insert(0, '.')

from parser.lexer import Lexer
from parser.parser import Parser

print("=== Test Parser ===\n")

# Test 1: Variable
code1 = "let x: int = 42"
print(f"Code: {code1}")
lexer1 = Lexer(code1)
parser1 = Parser(lexer1.tokenize())
ast1 = parser1.parse()
print(f"AST: {ast1}")
print(f"Statement: {ast1.statements[0]}\n")

# Test 2: Fonction
code2 = """
fn add(a: int, b: int) -> int {
    return a + b
}
"""
print(f"Code: {code2}")
lexer2 = Lexer(code2)
parser2 = Parser(lexer2.tokenize())
ast2 = parser2.parse()
print(f"AST: {ast2}")
print(f"Statement: {ast2.statements[0]}\n")

# Test 3: Type déclaration
code3 = "type Vec = Vector<float, dim=256, q=8>"
print(f"Code: {code3}")
lexer3 = Lexer(code3)
parser3 = Parser(lexer3.tokenize())
ast3 = parser3.parse()
print(f"AST: {ast3}")
print(f"Statement: {ast3.statements[0]}\n")

# Test 4: Annotation + fonction
code4 = """
@plastic(rate: 0.001)
fn adapt(state: Vec, delta: Vec) -> Vec {
    return state
}
"""
print(f"Code: {code4}")
lexer4 = Lexer(code4)
parser4 = Parser(lexer4.tokenize())
ast4 = parser4.parse()
print(f"AST: {ast4}")
print(f"Statement: {ast4.statements[0]}\n")

# Test 5: Opérateurs vectoriels
code5 = "let v3 = v1 .+ v2"
print(f"Code: {code5}")
lexer5 = Lexer(code5)
parser5 = Parser(lexer5.tokenize())
ast5 = parser5.parse()
print(f"AST: {ast5}")
print(f"Statement: {ast5.statements[0]}\n")

print("=== Parser tests passed! ===")
