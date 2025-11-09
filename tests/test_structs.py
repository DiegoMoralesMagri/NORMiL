"""
Tests pour Phase 3.5 - Structures de données
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


def test_empty_struct():
    """Test struct vide"""
    code = """
    fn main() {
        let empty = {}
        print("OK")
    }
    """
    result = run_code(code)
    print("[PASS] test_empty_struct")


def test_simple_struct():
    """Test struct simple"""
    code = """
    fn main() {
        let point = {x: 3.0, y: 4.0}
        print(point.x)
    }
    """
    result = run_code(code)
    print("[PASS] test_simple_struct")


def test_field_access():
    """Test accès aux champs"""
    code = """
    fn main() {
        let person = {name: "Alice", age: 30}
        let name = person.name
        let age = person.age
        print(name)
        print(age)
    }
    """
    result = run_code(code)
    print("[PASS] test_field_access")


def test_mixed_types():
    """Test types mixtes dans struct"""
    code = """
    fn main() {
        let data = {
            count: 42,
            ratio: 3.14,
            label: "test",
            active: true
        }
        print(data.count)
        print(data.ratio)
        print(data.label)
        print(data.active)
    }
    """
    result = run_code(code)
    print("[PASS] test_mixed_types")


def test_nested_struct():
    """Test structs imbriqués"""
    code = """
    fn main() {
        let rect = {
            top_left: {x: 0.0, y: 10.0},
            bottom_right: {x: 20.0, y: 0.0}
        }
        let x1 = rect.top_left.x
        let y1 = rect.top_left.y
        let x2 = rect.bottom_right.x
        let y2 = rect.bottom_right.y
        print(x1)
        print(y1)
        print(x2)
        print(y2)
    }
    """
    result = run_code(code)
    print("[PASS] test_nested_struct")


def test_struct_in_expression():
    """Test struct dans expression"""
    code = """
    fn main() {
        let point = {x: 3.0, y: 4.0}
        let sum = point.x + point.y
        print(sum)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_in_expression")


def test_struct_calculation():
    """Test calculs avec champs de struct"""
    code = """
    fn main() {
        let point = {x: 3.0, y: 4.0}
        let dist_sq = point.x * point.x + point.y * point.y
        print(dist_sq)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_calculation")


def test_struct_with_vectors():
    """Test struct contenant des vecteurs"""
    code = """
    fn main() {
        let data = {
            weights: ones(dim: 64),
            bias: 0.5
        }
        let norm_val = norm(data.weights)
        print(norm_val)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_with_vectors")


def test_struct_in_function_arg():
    """Test passer struct en argument"""
    code = """
    fn get_x(point: any) -> float {
        return point.x
    }
    
    fn main() {
        let p = {x: 5.0, y: 10.0}
        let x_val = get_x(p)
        print(x_val)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_in_function_arg")


def test_struct_in_function_return():
    """Test retourner struct depuis fonction"""
    code = """
    fn create_point(x: float, y: float) -> any {
        return {x: x, y: y}
    }
    
    fn main() {
        let p = create_point(7.0, 8.0)
        print(p.x)
        print(p.y)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_in_function_return")


def test_struct_comparison():
    """Test comparaison de champs"""
    code = """
    fn main() {
        let p1 = {x: 5.0, y: 3.0}
        let p2 = {x: 2.0, y: 8.0}
        
        if p1.x > p2.x {
            print("p1.x plus grand")
        } else {
            print("p2.x plus grand")
        }
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_comparison")


def test_struct_with_type_inference():
    """Test struct avec inférence de types"""
    code = """
    fn main() {
        let config = {size: 128, rate: 0.01, enabled: true}
        let s = config.size
        let r = config.rate
        let e = config.enabled
        print(s)
        print(r)
        print(e)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_with_type_inference")


def test_struct_modification():
    """Test modification de struct (via référence)"""
    code = """
    fn main() {
        let point = {x: 1.0, y: 2.0}
        let old_x = point.x
        print(old_x)
        // Note: modification directe nécessiterait assignation
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_modification")


def test_struct_in_condition():
    """Test struct dans condition"""
    code = """
    fn main() {
        let config = {threshold: 0.5, enabled: true}
        
        if config.enabled {
            if config.threshold > 0.3 {
                print("Active et seuil OK")
            }
        }
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_in_condition")


def test_struct_multiple_levels():
    """Test struct à 3 niveaux d'imbrication"""
    code = """
    fn main() {
        let network = {
            layers: {
                input: {size: 128, activation: "relu"},
                hidden: {size: 64, activation: "relu"},
                output: {size: 10, activation: "softmax"}
            }
        }
        let input_size = network.layers.input.size
        print(input_size)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_multiple_levels")


def test_struct_with_string_ops():
    """Test struct avec opérations string"""
    code = """
    fn main() {
        let person = {first: "Alice", last: "Dupont", age: 30}
        let full_name = person.first + " " + person.last
        print(full_name)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_with_string_ops")


def test_struct_array_like():
    """Test struct comme collection"""
    code = """
    fn main() {
        let colors = {
            red: 255,
            green: 128,
            blue: 64
        }
        let total = colors.red + colors.green + colors.blue
        print(total)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_array_like")


def test_struct_with_for_loop():
    """Test struct dans boucle"""
    code = """
    fn main() {
        for i in range(0, 3) {
            let point = {x: 1.0, y: 2.0}
            let sum = point.x + point.y
            print(sum)
        }
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_with_for_loop")


def test_struct_factory_pattern():
    """Test pattern factory avec structs"""
    code = """
    fn create_config(size: int, rate: float) -> any {
        return {
            model_size: size,
            learning_rate: rate,
            epochs: 100,
            batch_size: 32
        }
    }
    
    fn main() {
        let cfg = create_config(256, 0.001)
        print(cfg.model_size)
        print(cfg.learning_rate)
        print(cfg.epochs)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_factory_pattern")


def test_struct_with_match():
    """Test struct avec pattern matching"""
    code = """
    fn classify_size(obj: any) -> str {
        let size = obj.size
        match size {
            case int(s) where s > 100 -> { return "large" }
            case int(s) where s > 50 -> { return "medium" }
            case _ -> { return "small" }
        }
    }
    
    fn main() {
        let item = {size: 75, weight: 10.5}
        let category = classify_size(item)
        print(category)
    }
    """
    result = run_code(code)
    print("[PASS] test_struct_with_match")


def run_all_tests():
    """Exécute tous les tests"""
    tests = [
        test_empty_struct,
        test_simple_struct,
        test_field_access,
        test_mixed_types,
        test_nested_struct,
        test_struct_in_expression,
        test_struct_calculation,
        test_struct_with_vectors,
        test_struct_in_function_arg,
        test_struct_in_function_return,
        test_struct_comparison,
        test_struct_with_type_inference,
        test_struct_modification,
        test_struct_in_condition,
        test_struct_multiple_levels,
        test_struct_with_string_ops,
        test_struct_array_like,
        test_struct_with_for_loop,
        test_struct_factory_pattern,
        test_struct_with_match,
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
    print(f"Tests structs: {passed} passed, {failed} failed")
    print(f"{'='*60}")
    
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
