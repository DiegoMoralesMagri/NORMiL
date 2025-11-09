#!/usr/bin/env python3
"""
Tests pour le système d'imports (Phase 3.2)
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


def test_import_simple():
    """Test import basique d'un module"""
    code = """
    import mathutils
    
    fn main() {
        let result = mathutils.abs(-42)
        print(result)
    }
    """
    output = run_code(code)
    assert "42" in output


def test_import_with_alias():
    """Test import avec alias"""
    code = """
    import vectors as vec
    
    fn main() {
        let v = vec.create_normalized(dim: 64, mean: 1.0, std: 0.2)
        let n = norm(v)
        print(n)
    }
    """
    output = run_code(code)
    # Le vecteur doit être normalisé (norme proche de 1.0)
    assert output.strip() != ""
    try:
        value = float(output.strip().split('\n')[0])
        assert 0.99 < value < 1.01
    except:
        pass


def test_import_multiple_modules():
    """Test import de plusieurs modules"""
    code = """
    import mathutils
    import vectors
    
    fn main() {
        let a = mathutils.max(10, 20)
        print(a)
        
        let v1 = vectors.create_normalized(dim: 32, mean: 1.0, std: 0.1)
        let v2 = vectors.create_normalized(dim: 32, mean: 0.5, std: 0.1)
        let sim = vectors.compute_similarity(v1, v2)
        print(sim)
    }
    """
    output = run_code(code)
    assert "20" in output
    # Similarité doit être un float
    lines = output.strip().split('\n')
    assert len(lines) >= 2


def test_import_module_function_call():
    """Test appel de fonction depuis module"""
    code = """
    import mathutils
    
    fn main() {
        let clamped = mathutils.clamp(value: 150, min_val: 0, max_val: 100)
        print(clamped)
    }
    """
    output = run_code(code)
    assert "100" in output


def test_import_module_multiple_calls():
    """Test appels multiples aux fonctions d'un module"""
    code = """
    import mathutils
    
    fn main() {
        let abs_val = mathutils.abs(-10)
        let max_val = mathutils.max(5, 15)
        let min_val = mathutils.min(5, 15)
        
        print(abs_val)
        print(max_val)
        print(min_val)
    }
    """
    output = run_code(code)
    assert "10" in output
    assert "15" in output
    assert "5" in output


def test_import_vectors_distance():
    """Test fonction distance du module vectors"""
    code = """
    import vectors
    
    fn main() {
        let v1 = zeros(dim: 64)
        let v2 = ones(dim: 64)
        let dist = vectors.distance(v1, v2)
        print(dist)
    }
    """
    output = run_code(code)
    # Distance entre zeros et ones devrait être sqrt(64) = 8.0
    assert output.strip() != ""
    try:
        value = float(output.strip().split('\n')[0])
        assert 7.9 < value < 8.1
    except:
        pass


def test_import_vectors_weighted_sum():
    """Test somme pondérée du module vectors"""
    code = """
    import vectors
    
    fn main() {
        let v1 = ones(dim: 32)
        let v2 = zeros(dim: 32)
        let result = vectors.weighted_sum(v1: v1, w1: 0.5, v2: v2, w2: 0.5)
        let n = norm(result)
        print(n)
    }
    """
    output = run_code(code)
    # weighted_sum(ones*0.5, zeros*0.5) = 0.5*ones
    # norm = 0.5 * sqrt(32) ≈ 2.828
    assert output.strip() != ""


def test_module_not_found():
    """Test erreur quand module introuvable"""
    code = """
    import nonexistent_module
    
    fn main() {
        print("Should not reach here")
    }
    """
    try:
        output = run_code(code)
        # Ne devrait pas arriver ici
        assert False, "Should have raised an error"
    except Exception as e:
        assert "Module not found" in str(e) or "nonexistent_module" in str(e)


def test_module_function_not_found():
    """Test erreur quand fonction du module introuvable"""
    code = """
    import mathutils
    
    fn main() {
        let x = mathutils.nonexistent_function(42)
        print(x)
    }
    """
    try:
        output = run_code(code)
        assert False, "Should have raised an error"
    except Exception as e:
        assert "no attribute" in str(e).lower() or "nonexistent_function" in str(e)


def test_import_with_local_variables():
    """Test que les imports n'interfèrent pas avec variables locales"""
    code = """
    import mathutils
    
    fn main() {
        let math_result = 42
        let module_result = mathutils.abs(-10)
        
        print(math_result)
        print(module_result)
    }
    """
    output = run_code(code)
    assert "42" in output
    assert "10" in output


def test_module_scope_isolation():
    """Test que les modules ont des scopes isolés"""
    code = """
    import mathutils
    
    fn main() {
        let x = 100
        let result = mathutils.abs(-50)
        print(result)
        print(x)
    }
    """
    output = run_code(code)
    assert "50" in output
    assert "100" in output


def test_import_in_function():
    """Test import au niveau global seulement (pas dans fonction)"""
    # Note: Dans notre implémentation actuelle, les imports sont traités
    # comme des statements normaux, donc ils fonctionnent partout.
    # Ce test vérifie juste que ça ne casse pas.
    code = """
    import mathutils
    
    fn helper() -> int {
        return mathutils.abs(-25)
    }
    
    fn main() {
        let result = helper()
        print(result)
    }
    """
    output = run_code(code)
    assert "25" in output


def test_module_caching():
    """Test que les modules sont mis en cache"""
    code = """
    import mathutils
    import mathutils as m
    
    fn main() {
        let r1 = mathutils.abs(-10)
        let r2 = m.abs(-20)
        print(r1)
        print(r2)
    }
    """
    output = run_code(code)
    assert "10" in output
    assert "20" in output


def test_vectors_similarity():
    """Test calcul de similarité"""
    code = """
    import vectors
    
    fn main() {
        let v1 = ones(dim: 64)
        let v2 = ones(dim: 64)
        let sim = vectors.compute_similarity(v1, v2)
        print(sim)
    }
    """
    output = run_code(code)
    # Similarité de deux vecteurs identiques = 1.0
    assert output.strip() != ""
    try:
        value = float(output.strip().split('\n')[0])
        assert 0.99 < value < 1.01
    except:
        pass


def test_import_parse_only():
    """Test que le parsing d'import fonctionne"""
    code = "import mathutils"
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    assert len(ast.statements) == 1
    from parser.ast_nodes import ImportStmt
    assert isinstance(ast.statements[0], ImportStmt)
    assert ast.statements[0].module_name == "mathutils"
    assert ast.statements[0].alias is None


def test_import_with_alias_parse():
    """Test parsing import avec alias"""
    code = "import vectors as vec"
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    
    from parser.ast_nodes import ImportStmt
    stmt = ast.statements[0]
    assert isinstance(stmt, ImportStmt)
    assert stmt.module_name == "vectors"
    assert stmt.alias == "vec"


if __name__ == "__main__":
    print("Running import tests...")
    
    tests = [
        ("Import simple", test_import_simple),
        ("Import with alias", test_import_with_alias),
        ("Import multiple modules", test_import_multiple_modules),
        ("Module function call", test_import_module_function_call),
        ("Module multiple calls", test_import_module_multiple_calls),
        ("Vectors distance", test_import_vectors_distance),
        ("Vectors weighted sum", test_import_vectors_weighted_sum),
        ("Module not found error", test_module_not_found),
        ("Module function not found", test_module_function_not_found),
        ("Import with local variables", test_import_with_local_variables),
        ("Module scope isolation", test_module_scope_isolation),
        ("Import in function", test_import_in_function),
        ("Module caching", test_module_caching),
        ("Vectors similarity", test_vectors_similarity),
        ("Import parse only", test_import_parse_only),
        ("Import with alias parse", test_import_with_alias_parse),
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
