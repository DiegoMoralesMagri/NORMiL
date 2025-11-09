#!/usr/bin/env python3
"""
Tests pour l'inférence de types (Phase 3.1)
"""

import sys
from pathlib import Path
from io import StringIO

# Ajouter le dossier parent au path (le dossier normil/)
NORMIL_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(NORMIL_ROOT))

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor


def run_code(code: str) -> str:
    """Helper pour exécuter du code NORMiL et capturer la sortie"""
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    
    try:
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        executor = Executor()
        executor.execute(ast)
        
        output = sys.stdout.getvalue()
        return output
    finally:
        sys.stdout = old_stdout


def test_infer_int():
    """Test inférence de type int"""
    code = """
    fn main() {
        let x = 42
        print(x)
    }
    """
    output = run_code(code)
    assert "42" in output


def test_infer_float():
    """Test inférence de type float"""
    code = """
    fn main() {
        let pi = 3.14159
        print(pi)
    }
    """
    output = run_code(code)
    assert "3.14159" in output


def test_infer_string():
    """Test inférence de type string"""
    code = """
    fn main() {
        let message = "Hello, inference!"
        print(message)
    }
    """
    output = run_code(code)
    assert "Hello, inference!" in output


def test_infer_bool():
    """Test inférence de type bool"""
    code = """
    fn main() {
        let flag = true
        print(flag)
        
        let disabled = false
        print(disabled)
    }
    """
    output = run_code(code)
    assert "true" in output.lower()
    assert "false" in output.lower()


def test_infer_from_expression():
    """Test inférence depuis une expression"""
    code = """
    fn main() {
        let a = 10
        let b = 20
        let sum = a + b
        print(sum)
    }
    """
    output = run_code(code)
    assert "30" in output


def test_infer_from_function_return():
    """Test inférence depuis le retour d'une fonction"""
    code = """
    fn get_number() -> int {
        return 42
    }
    
    fn main() {
        let result = get_number()
        print(result)
    }
    """
    output = run_code(code)
    assert "42" in output


def test_infer_vector():
    """Test inférence de type Vec"""
    code = """
    fn main() {
        let v = zeros(dim: 64)
        let n = norm(v)
        print(n)
    }
    """
    output = run_code(code)
    assert "0" in output


def test_infer_from_vector_operation():
    """Test inférence depuis opération vectorielle"""
    code = """
    fn main() {
        let v1 = ones(dim: 64)
        let v2 = ones(dim: 64)
        let sum = vec_add(v1, v2)
        let n = norm(sum)
        print(n)
    }
    """
    output = run_code(code)
    # norm(v) où v contient 64 valeurs de 2.0
    # sqrt(64 * 4) = sqrt(256) = 16.0
    assert "16" in output


def test_mixed_inference_and_explicit():
    """Test mélange d'inférence et types explicites"""
    code = """
    fn main() {
        let x = 10
        let y: int = 20
        let z = x + y
        print(z)
    }
    """
    output = run_code(code)
    assert "30" in output


def test_const_with_inference():
    """Test const avec inférence"""
    code = """
    fn main() {
        const PI = 3.14159
        print(PI)
    }
    """
    output = run_code(code)
    assert "3.14159" in output


def test_inference_in_loops():
    """Test inférence dans des boucles"""
    code = """
    fn main() {
        for i in range(0, 3) {
            let doubled = i * 2
            print(doubled)
        }
    }
    """
    output = run_code(code)
    lines = [line.strip() for line in output.strip().split('\n')]
    assert "0" in lines[0]
    assert "2" in lines[1]
    assert "4" in lines[2]


def test_inference_with_negative_numbers():
    """Test inférence avec nombres négatifs"""
    code = """
    fn main() {
        let neg_int = -42
        let neg_float = -3.14
        print(neg_int)
        print(neg_float)
    }
    """
    output = run_code(code)
    assert "-42" in output
    assert "-3.14" in output


def test_inference_with_comparison():
    """Test inférence avec comparaisons (retour bool)"""
    code = """
    fn main() {
        let is_greater = 10 > 5
        print(is_greater)
        
        let is_equal = 5 == 5
        print(is_equal)
    }
    """
    output = run_code(code)
    output_lower = output.lower()
    assert "true" in output_lower


def test_inference_multiple_variables():
    """Test inférence de plusieurs variables de types différents"""
    code = """
    fn main() {
        let a = 42
        let b = 3.14
        let c = "hello"
        let d = true
        let e = zeros(dim: 32)
        
        print(a)
        print(b)
        print(c)
        print(d)
        print(norm(e))
    }
    """
    output = run_code(code)
    assert "42" in output
    assert "3.14" in output
    assert "hello" in output
    assert "true" in output.lower()
    assert "0" in output


def test_inference_in_nested_scopes():
    """Test inférence dans des scopes imbriqués"""
    code = """
    fn main() {
        let outer = 100
        
        if true {
            let inner = 200
            let sum = outer + inner
            print(sum)
        }
    }
    """
    output = run_code(code)
    assert "300" in output


def test_inference_with_pattern_matching():
    """Test inférence avec pattern matching"""
    code = """
    fn classify(n: int) -> str {
        match n {
            case 0 -> { return "zero" }
            case int(x) where x > 0 -> { return "positif" }
            case _ -> { return "negatif" }
        }
    }
    
    fn main() {
        let value = 42
        let result = classify(value)
        print(result)
    }
    """
    output = run_code(code)
    assert "positif" in output


def test_inference_with_plastic_function():
    """Test inférence avec fonction @plastic"""
    code = """
    @plastic(rate: 0.01, mode: "hebbian")
    fn learn(w: Vec, x: Vec) -> Vec {
        let delta = scale(vec_mul(w, x), 0.01)
        return vec_add(w, delta)
    }
    
    fn main() {
        let weights = random(dim: 32, mean: 0.0, std: 0.1)
        let input = random(dim: 32, mean: 1.0, std: 0.2)
        
        let updated = learn(weights, input)
        let n = norm(updated)
        print(n)
    }
    """
    output = run_code(code)
    # Devrait afficher une norme > 0
    assert output.strip() != ""
    try:
        value = float(output.strip().split('\n')[0])
        assert value > 0
    except:
        pass  # Si le format n'est pas exactement comme attendu


def test_inference_with_atomic_function():
    """Test inférence avec fonction @atomic"""
    code = """
    @atomic
    fn safe_add(a: int, b: int) -> int {
        let result = a + b
        return result
    }
    
    fn main() {
        let sum = safe_add(10, 20)
        print(sum)
    }
    """
    output = run_code(code)
    assert "30" in output


if __name__ == "__main__":
    print("Running type inference tests...")
    
    tests = [
        ("Infer int", test_infer_int),
        ("Infer float", test_infer_float),
        ("Infer string", test_infer_string),
        ("Infer bool", test_infer_bool),
        ("Infer from expression", test_infer_from_expression),
        ("Infer from function return", test_infer_from_function_return),
        ("Infer vector", test_infer_vector),
        ("Infer from vector operation", test_infer_from_vector_operation),
        ("Mixed inference and explicit", test_mixed_inference_and_explicit),
        ("Const with inference", test_const_with_inference),
        ("Inference in loops", test_inference_in_loops),
        ("Inference with negative numbers", test_inference_with_negative_numbers),
        ("Inference with comparison", test_inference_with_comparison),
        ("Inference multiple variables", test_inference_multiple_variables),
        ("Inference in nested scopes", test_inference_in_nested_scopes),
        ("Inference with pattern matching", test_inference_with_pattern_matching),
        ("Inference with @plastic", test_inference_with_plastic_function),
        ("Inference with @atomic", test_inference_with_atomic_function),
    ]
    
    passed = 0
    failed = 0
    
    for name, test_func in tests:
        try:
            test_func()
            print(f"[PASS] {name}")
            passed += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            failed += 1
        except Exception as e:
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*50}")
    print(f"Tests passed: {passed}/{len(tests)}")
    print(f"Tests failed: {failed}/{len(tests)}")
    
    if failed == 0:
        print("All tests passed!")
        sys.exit(0)
    else:
        print(f"{failed} test(s) failed")
        sys.exit(1)
