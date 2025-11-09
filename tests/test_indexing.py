"""
Tests pour Phase 3.4 - Indexation de vecteurs
"""
import sys
import os

# Configuration du path
NORMIL_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, NORMIL_ROOT)

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor


def run_code(code: str):
    """Helper pour exécuter du code NORMiL"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    executor = Executor()
    return executor.execute(ast)


def test_basic_index_access():
    """Test accès index simple"""
    code = """
    fn main() {
        let v = fill(dim: 5, value: 7.0)
        let elem = v[2]
        print(elem)
    }
    """
    result = run_code(code)
    print("[PASS] test_basic_index_access")


def test_first_element():
    """Test accès premier élément"""
    code = """
    fn main() {
        let v = fill(dim: 10, value: 3.14)
        let first = v[0]
        print(first)
    }
    """
    result = run_code(code)
    print("[PASS] test_first_element")


def test_last_element():
    """Test accès dernier élément"""
    code = """
    fn main() {
        let v = fill(dim: 8, value: 2.5)
        let last = v[7]
        print(last)
    }
    """
    result = run_code(code)
    print("[PASS] test_last_element")


def test_index_in_expression():
    """Test index dans expression"""
    code = """
    fn main() {
        let v = fill(dim: 4, value: 5.0)
        let sum = v[0] + v[1] + v[2]
        print(sum)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_in_expression")


def test_index_with_ones():
    """Test indexation avec ones"""
    code = """
    fn main() {
        let v = ones(dim: 6)
        let elem = v[3]
        print(elem)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_with_ones")


def test_index_with_zeros():
    """Test indexation avec zeros"""
    code = """
    fn main() {
        let v = zeros(dim: 5)
        let elem = v[2]
        print(elem)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_with_zeros")


def test_index_with_random():
    """Test indexation avec random"""
    code = """
    fn main() {
        let v = random(dim: 10, mean: 0.0, std: 1.0)
        let elem = v[5]
        print(elem)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_with_random")


def test_index_after_operation():
    """Test indexation après opération vectorielle"""
    code = """
    fn main() {
        let v1 = fill(dim: 5, value: 2.0)
        let v2 = fill(dim: 5, value: 3.0)
        let sum = vec_add(v1, v2)
        let elem = sum[0]
        print(elem)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_after_operation")


def test_index_after_scale():
    """Test indexation après scale"""
    code = """
    fn main() {
        let v = ones(dim: 4)
        let scaled = scale(v, 10.0)
        let elem = scaled[2]
        print(elem)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_after_scale")


def test_index_after_normalize():
    """Test indexation après normalisation"""
    code = """
    fn main() {
        let v = fill(dim: 64, value: 1.0)
        let normalized = normalize(v)
        let elem = normalized[0]
        print(elem)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_after_normalize")


def test_index_multiple_access():
    """Test accès multiples"""
    code = """
    fn main() {
        let v = fill(dim: 6, value: 4.0)
        let a = v[0]
        let b = v[2]
        let c = v[4]
        let sum = a + b + c
        print(sum)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_multiple_access")


def test_index_in_condition():
    """Test index dans condition"""
    code = """
    fn main() {
        let v = fill(dim: 5, value: 10.0)
        let elem = v[2]
        
        if elem > 5.0 {
            print("OK")
        } else {
            print("NOK")
        }
    }
    """
    result = run_code(code)
    print("[PASS] test_index_in_condition")


def test_index_in_function():
    """Test index passé à fonction"""
    code = """
    fn get_element(v: Vec, i: int) -> float {
        return v[i]
    }
    
    fn main() {
        let v = fill(dim: 8, value: 7.5)
        let elem = get_element(v, 3)
        print(elem)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_in_function")


def test_index_comparison():
    """Test comparaison d'éléments"""
    code = """
    fn main() {
        let v1 = fill(dim: 4, value: 5.0)
        let v2 = fill(dim: 4, value: 3.0)
        
        if v1[0] > v2[0] {
            print("v1 plus grand")
        } else {
            print("v2 plus grand")
        }
    }
    """
    result = run_code(code)
    print("[PASS] test_index_comparison")


def test_index_with_type_inference():
    """Test indexation avec inférence de type"""
    code = """
    fn main() {
        let v = fill(dim: 5, value: 2.5)
        let elem = v[1]  // Type inféré: float
        let doubled = elem * 2.0
        print(doubled)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_with_type_inference")


def test_index_chained():
    """Test accès index enchaîné avec opérations"""
    code = """
    fn main() {
        let v = ones(dim: 10)
        let scaled = scale(v, 5.0)
        let elem1 = scaled[0]
        let elem2 = scaled[9]
        let avg = (elem1 + elem2) / 2.0
        print(avg)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_chained")


def test_index_result_in_vec_op():
    """Test utiliser résultat index dans opération vectorielle"""
    code = """
    fn main() {
        let v1 = fill(dim: 4, value: 3.0)
        let scalar = v1[0]
        let v2 = ones(dim: 4)
        let scaled = scale(v2, scalar)
        let result = scaled[0]
        print(result)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_result_in_vec_op")


def test_index_boundary_0():
    """Test limite inférieure (index 0)"""
    code = """
    fn main() {
        let v = random(dim: 100, mean: 0.0, std: 1.0)
        let first = v[0]
        print("OK")
    }
    """
    result = run_code(code)
    print("[PASS] test_index_boundary_0")


def test_index_boundary_max():
    """Test limite supérieure (index max)"""
    code = """
    fn main() {
        let v = random(dim: 50, mean: 0.0, std: 1.0)
        let last = v[49]
        print("OK")
    }
    """
    result = run_code(code)
    print("[PASS] test_index_boundary_max")


def test_index_with_const():
    """Test indexation avec constante"""
    code = """
    const INDEX = 3
    
    fn main() {
        let v = fill(dim: 10, value: 8.0)
        let elem = v[INDEX]
        print(elem)
    }
    """
    result = run_code(code)
    print("[PASS] test_index_with_const")


def run_all_tests():
    """Exécute tous les tests"""
    tests = [
        test_basic_index_access,
        test_first_element,
        test_last_element,
        test_index_in_expression,
        test_index_with_ones,
        test_index_with_zeros,
        test_index_with_random,
        test_index_after_operation,
        test_index_after_scale,
        test_index_after_normalize,
        test_index_multiple_access,
        test_index_in_condition,
        test_index_in_function,
        test_index_comparison,
        test_index_with_type_inference,
        test_index_chained,
        test_index_result_in_vec_op,
        test_index_boundary_0,
        test_index_boundary_max,
        test_index_with_const,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"[FAIL] {test.__name__}: {e}")
            failed += 1
    
    print(f"\n{'='*60}")
    print(f"Tests indexation: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
