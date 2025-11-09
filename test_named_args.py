"""
Test des arguments nommés
"""
import sys
sys.path.insert(0, 'c:/Users/serveur/Documents/OpenRed/openredNetwork/modules/ia2/normil')

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor


def run_normil_code(code: str):
    """Helper pour exécuter du code NORMiL"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    executor = Executor()
    for stmt in ast.statements:
        executor.exec_statement(stmt)
    return executor


print("=== Test Arguments Nommés ===\n")

# Test 1: Arguments positionnels simples
print("1. Arguments positionnels (baseline):")
code1 = """
let v = random(128)
let nrm = norm(v)
print(nrm)
"""
run_normil_code(code1)
print()

# Test 2: Arguments nommés
print("2. Arguments nommés:")
code2 = """
let v = random(dim: 256)
let nrm = norm(v)
print(nrm)
"""
try:
    run_normil_code(code2)
    print()
except Exception as e:
    print(f"❌ Erreur: {e}\n")

# Test 3: Mélange positionnels et nommés
print("3. Mix positionnel + nommé:")
code3 = """
let v = random(256, mean: 0.5, std: 0.1)
let nrm = norm(v)
print(nrm)
"""
try:
    run_normil_code(code3)
    print()
except Exception as e:
    print(f"❌ Erreur: {e}\n")

# Test 4: Fonction utilisateur avec arguments nommés
print("4. Fonction utilisateur avec args nommés:")
code4 = """
fn greet(first: str, last: str) -> str {
    return first + " " + last
}

fn main() {
    let msg1 = greet("Diego", "Morales")
    print(msg1)
    
    let msg2 = greet(first: "John", last: "Doe")
    print(msg2)
    
    let msg3 = greet(last: "Smith", first: "Jane")
    print(msg3)
}
"""
try:
    exec4 = run_normil_code(code4)
    exec4.call_user_function(exec4.current_scope.get_function('main'), [], {})
except Exception as e:
    print(f"❌ Erreur: {e}\n")

print("=== Tests des arguments nommes termines! ===")
