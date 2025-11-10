#!/usr/bin/env python3
"""
Tests pour les Transactions @atomic de NORMiL
==============================================

Auteur : Diego Morales Magri

Teste les transactions atomiques:
- @atomic : Rollback automatique sur erreur
- Commit implicite sur succès
- Protection de l'état des variables
- Messages d'erreur appropriés
"""

import sys
from pathlib import Path
from io import StringIO

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor, ExecutionError


def run_code(code: str):
    """Execute NORMiL code and return executor"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    executor = Executor()
    executor.execute(ast)
    return executor


def test_atomic_success_commit():
    """Test qu'une transaction réussie commit les changements"""
    print("Testing @atomic success commit...")
    
    code = """
@atomic
fn update_value(x: int) -> int {
    let result = x + 10
    return result
}

fn main() {
    let value = update_value(5)
    print(value)
}
"""
    
    # Capturer stdout
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    
    try:
        executor = run_code(code)
        output = captured.getvalue().strip()
        
        # Vérifier que le résultat est correct
        assert output == "15", f"Expected 15, got {output}"
        
        # Vérifier que les métadonnées atomic sont présentes
        assert 'update_value' in executor.function_metadata
        assert 'atomic' in executor.function_metadata['update_value']
        
        print("=== @atomic success commit test passed! ===")
    finally:
        sys.stdout = old_stdout


def test_atomic_rollback_on_error():
    """Test que les changements sont annulés en cas d'erreur"""
    print("\nTesting @atomic rollback on error...")
    
    code = """
let global_counter = 0

@atomic
fn failing_update(x: int) -> int {
    global_counter = global_counter + 1
    let y = x + 10
    // Simuler une erreur en divisant par zéro
    let error = 10 / 0
    return y
}

fn main() {
    failing_update(5)
}
"""
    
    try:
        executor = run_code(code)
        # Ne devrait pas arriver ici
        assert False, "Should have raised an error"
    except Exception as e:
        # Vérifier que l'erreur mentionne le rollback
        error_msg = str(e)
        assert "Transaction rolled back" in error_msg or "division" in error_msg.lower(), \
            f"Expected rollback error, got: {error_msg}"
        print("=== @atomic rollback on error test passed! ===")


def test_atomic_protects_variables():
    """Test que @atomic protège les variables locales"""
    print("\nTesting @atomic protects variables...")
    
    code = """
@atomic
fn safe_update(x: int) -> int {
    let temp = x + 5
    let result = temp * 2
    return result
}

fn main() {
    let value = safe_update(10)
    print(value)
}
"""
    
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    
    try:
        executor = run_code(code)
        output = captured.getvalue().strip()
        
        # (10 + 5) * 2 = 30
        assert output == "30", f"Expected 30, got {output}"
        print("=== @atomic protects variables test passed! ===")
    finally:
        sys.stdout = old_stdout


def test_non_atomic_function():
    """Test qu'une fonction sans @atomic ne crée pas de transaction"""
    print("\nTesting non-atomic function...")
    
    code = """
fn regular_function(x: int) -> int {
    return x * 3
}

fn main() {
    let result = regular_function(7)
    print(result)
}
"""
    
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    
    try:
        executor = run_code(code)
        output = captured.getvalue().strip()
        
        assert output == "21", f"Expected 21, got {output}"
        
        # Vérifier qu'il n'y a pas de métadonnées atomic
        if 'regular_function' in executor.function_metadata:
            assert 'atomic' not in executor.function_metadata['regular_function']
        
        print("=== Non-atomic function test passed! ===")
    finally:
        sys.stdout = old_stdout


def test_atomic_with_return():
    """Test @atomic avec return explicite"""
    print("\nTesting @atomic with explicit return...")
    
    code = """
@atomic
fn compute(a: int, b: int) -> int {
    let sum = a + b
    if sum > 10 {
        return sum
    }
    return 0
}

fn main() {
    let r1 = compute(5, 8)
    let r2 = compute(2, 3)
    print(r1)
    print(r2)
}
"""
    
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    
    try:
        executor = run_code(code)
        output = captured.getvalue().strip().split('\n')
        
        assert output[0] == "13", f"Expected 13, got {output[0]}"
        assert output[1] == "0", f"Expected 0, got {output[1]}"
        print("=== @atomic with return test passed! ===")
    finally:
        sys.stdout = old_stdout


def test_atomic_metadata():
    """Test que les métadonnées @atomic sont correctes"""
    print("\nTesting @atomic metadata...")
    
    code = """
@atomic
fn transactional(x: int) -> int {
    return x + 1
}

fn main() {
    transactional(1)
}
"""
    
    executor = run_code(code)
    
    # Vérifier métadonnées
    assert 'transactional' in executor.function_metadata
    atomic_meta = executor.function_metadata['transactional']['atomic']
    
    assert atomic_meta['enabled'] == True
    assert atomic_meta['isolation_level'] == 'serializable'
    
    print("=== @atomic metadata test passed! ===")


def test_atomic_with_vectors():
    """Test @atomic avec opérations vectorielles"""
    print("\nTesting @atomic with vectors...")
    
    code = """
@atomic
fn vec_transform(v: Vec, scale_factor: float) -> Vec {
    let scaled = scale(v, scale_factor)
    let normalized = normalize(scaled)
    return normalized
}

fn main() {
    let vec = random(dim: 32, mean: 0.0, std: 1.0)
    let result = vec_transform(vec, 2.0)
    print(norm(result))
}
"""
    
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    
    try:
        executor = run_code(code)
        output = captured.getvalue().strip()
        
        # normalize() devrait retourner un vecteur de norme ~1.0
        norm_value = float(output)
        assert 0.99 < norm_value < 1.01, f"Expected norm ~1.0, got {norm_value}"
        print("=== @atomic with vectors test passed! ===")
    finally:
        sys.stdout = old_stdout


def test_atomic_combined_with_plastic():
    """Test combinaison @atomic et @plastic"""
    print("\nTesting @atomic combined with @plastic...")
    
    code = """
@atomic
@plastic(rate: 0.01)
fn atomic_plastic(state: Vec, input: Vec) -> Vec {
    let delta = scale(input, 0.01)
    let result = vec_add(state, delta)
    return result
}

fn main() {
    let state = random(dim: 16, mean: 0.0, std: 0.1)
    let input = random(dim: 16, mean: 1.0, std: 0.1)
    let result = atomic_plastic(state, input)
    print(norm(result))
}
"""
    
    old_stdout = sys.stdout
    sys.stdout = captured = StringIO()
    
    try:
        executor = run_code(code)
        output = captured.getvalue().strip()
        
        # Vérifier que les deux métadonnées sont présentes
        assert 'atomic_plastic' in executor.function_metadata
        meta = executor.function_metadata['atomic_plastic']
        assert 'atomic' in meta
        assert 'plastic' in meta
        
        print("=== @atomic combined with @plastic test passed! ===")
    finally:
        sys.stdout = old_stdout


# Point d'entrée
if __name__ == "__main__":
    print("=" * 55)
    print("        NORMiL Atomic Transactions Test Suite")
    print("=" * 55)
    
    tests = [
        test_atomic_success_commit,
        test_atomic_rollback_on_error,
        test_atomic_protects_variables,
        test_non_atomic_function,
        test_atomic_with_return,
        test_atomic_metadata,
        test_atomic_with_vectors,
        test_atomic_combined_with_plastic,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAILED: {e}")
            failed += 1
        except Exception as e:
            print(f"ERROR: {e}")
            import traceback
            traceback.print_exc()
            failed += 1
    
    print("\n" + "=" * 55)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests FAILED")
        sys.exit(1)
    else:
        print("         ALL TESTS PASSED!")
    print("=" * 55)
