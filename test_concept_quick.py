#!/usr/bin/env python
"""Test rapide Concept"""

import sys
from pathlib import Path

# Ajouter le module au path
sys.path.insert(0, str(Path(__file__).parent))

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor

code = '''
c = Concept {
    concept_id: "test",
    centroid_vec: vec(2, [1.0, 2.0]),
    doc_count: 5,
    provenance_versions: [],
    trust_score: 0.5,
    labels: []
}
print(c.concept_id)
'''

try:
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    
    print("Tokens OK")
    
    parser = Parser(tokens)
    ast = parser.parse()
    
    print("Parse OK")
    print(f"AST: {ast}")
    
    executor = Executor()
    result = executor.execute(ast)
    
    print(f"Result: {result}")
except Exception as e:
    import traceback
    print(f"ERROR: {e}")
    traceback.print_exc()
