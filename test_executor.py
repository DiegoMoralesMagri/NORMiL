"""
Script de test pour l'executor NORMiL
"""

import sys
sys.path.insert(0, '.')

from runtime.executor import run_normil_code

print("=== Test NORMiL Executor ===\n")

# Test 1: Variable simple
print("1. Variable:")
code1 = """
let x: int = 42
let y: int = 10
let z = x + y
print(z)
"""
exec1 = run_normil_code(code1)
print()

# Test 2: Fonction simple
print("2. Fonction:")
code2 = """
fn add(a: int, b: int) -> int {
    return a + b
}

let result = add(10, 32)
print(result)
"""
exec2 = run_normil_code(code2)
print()

# Test 3: Opérateurs vectoriels
print("3. Vecteurs:")
code3 = """
let v1 = zeros(128)
let v2 = ones(128)
let v3 = vec_add(v1, v2)
let nrm = norm(v2)
print(nrm)
"""
exec3 = run_normil_code(code3)
print()

# Test 4: Boucle
print("4. Boucle for:")
code4 = """
let numbers = [1, 2, 3, 4, 5]
for n in numbers {
    print(n)
}
"""
exec4 = run_normil_code(code4)
print()

# Test 5: Condition
print("5. If/Else:")
code5 = """
let x = 42
if x > 40 {
    print("Grand")
} else {
    print("Petit")
}
"""
exec5 = run_normil_code(code5)
print()

print("=== Executor tests passed! ===")
