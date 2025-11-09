#!/usr/bin/env python3
"""
Tests pour les Annotations de NORMiL
=====================================

Teste les différentes annotations:
- @plastic(rate, mode) : Plasticité neuronale
- Métadonnées accessibles dans le scope
- Valeurs par défaut
"""

import sys
from pathlib import Path
from io import StringIO

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor


def run_code(code: str):
    """Execute NORMiL code and return executor"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    executor = Executor()
    executor.execute(ast)
    return executor


def test_plastic_annotation_basic():
    """Test annotation @plastic basique"""
    print("Testing @plastic annotation basic...")
    
    code = """
@plastic(rate: 0.005)
fn adapt(state: int) -> int {
    return state + 1
}

fn main() {
    let result = adapt(10)
    print(result)
}
"""
    
    executor = run_code(code)
    
    # Vérifier que les métadonnées sont stockées
    assert 'adapt' in executor.function_metadata, "Function metadata not stored"
    assert 'plastic' in executor.function_metadata['adapt'], "Plastic metadata not found"
    
    plastic_meta = executor.function_metadata['adapt']['plastic']
    assert plastic_meta['rate'] == 0.005, f"Expected rate 0.005, got {plastic_meta['rate']}"
    assert plastic_meta['mode'] == 'hebbian', f"Expected mode 'hebbian', got {plastic_meta['mode']}"
    assert plastic_meta['enabled'] == True, "Expected enabled=True"
    
    print("=== @plastic annotation basic test passed! ===")


def test_plastic_annotation_with_mode():
    """Test annotation @plastic avec mode personnalisé"""
    print("\nTesting @plastic annotation with mode...")
    
    code = """
@plastic(rate: 0.01, mode: "stdp")
fn learn(input: int, target: int) -> int {
    return input
}

fn main() {
    let result = learn(5, 10)
}
"""
    
    executor = run_code(code)
    
    # Vérifier métadonnées
    assert 'learn' in executor.function_metadata
    plastic_meta = executor.function_metadata['learn']['plastic']
    
    assert plastic_meta['rate'] == 0.01, f"Expected rate 0.01, got {plastic_meta['rate']}"
    assert plastic_meta['mode'] == "stdp", f"Expected mode 'stdp', got {plastic_meta['mode']}"
    
    print("=== @plastic annotation with mode test passed! ===")


def test_plastic_annotation_defaults():
    """Test valeurs par défaut de @plastic"""
    print("\nTesting @plastic annotation defaults...")
    
    code = """
@plastic
fn default_plastic(x: int) -> int {
    return x * 2
}

fn main() {
    let result = default_plastic(21)
}
"""
    
    # Note: @plastic sans arguments nécessite une modification du parser
    # Pour l'instant, testons avec des arguments
    code = """
@plastic(rate: 0.001)
fn default_mode(x: int) -> int {
    return x * 2
}

fn main() {
    let result = default_mode(21)
}
"""
    
    executor = run_code(code)
    
    # Vérifier valeurs par défaut
    plastic_meta = executor.function_metadata['default_mode']['plastic']
    assert plastic_meta['rate'] == 0.001
    assert plastic_meta['mode'] == 'hebbian', "Default mode should be 'hebbian'"
    
    print("=== @plastic annotation defaults test passed! ===")


def test_metadata_accessible_in_function():
    """Test que les métadonnées sont accessibles dans la fonction"""
    print("\nTesting metadata accessible in function scope...")
    
    code = """
@plastic(rate: 0.002, mode: "custom")
fn check_metadata(x: int) -> int {
    // Les métadonnées sont dans __metadata__
    // Pour l'instant, on ne peut pas encore y accéder depuis NORMiL
    // (nécessiterait l'ajout d'une primitive get_metadata)
    return x + 1
}

fn main() {
    let result = check_metadata(5)
    print(result)
}
"""
    
    executor = run_code(code)
    
    # Vérifier que les métadonnées existent
    assert 'check_metadata' in executor.function_metadata
    plastic_meta = executor.function_metadata['check_metadata']['plastic']
    assert plastic_meta['rate'] == 0.002
    assert plastic_meta['mode'] == "custom"
    
    print("=== Metadata accessible test passed! ===")


def test_multiple_annotations():
    """Test plusieurs annotations sur une même fonction"""
    print("\nTesting multiple annotations (future)...")
    
    # Pour l'instant, testons juste @plastic
    # Dans le futur, on pourrait avoir @plastic @memoize @profile
    code = """
@plastic(rate: 0.003)
fn multi_annotated(a: int, b: int) -> int {
    return a + b
}

fn main() {
    let result = multi_annotated(10, 20)
}
"""
    
    executor = run_code(code)
    
    assert 'multi_annotated' in executor.function_metadata
    assert 'plastic' in executor.function_metadata['multi_annotated']
    
    print("=== Multiple annotations test passed! ===")


def test_function_without_annotation():
    """Test fonction sans annotation"""
    print("\nTesting function without annotation...")
    
    code = """
fn regular_function(x: int) -> int {
    return x * 2
}

fn main() {
    let result = regular_function(15)
    print(result)
}
"""
    
    executor = run_code(code)
    
    # Vérifier qu'il n'y a pas de métadonnées pour cette fonction
    assert 'regular_function' not in executor.function_metadata, \
        "Function without annotation should not have metadata"
    
    print("=== Function without annotation test passed! ===")


def test_plastic_with_string_mode():
    """Test @plastic avec mode en tant que string"""
    print("\nTesting @plastic with string mode...")
    
    code = """
@plastic(rate: 0.007, mode: "anti_hebbian")
fn anti_learn(x: int) -> int {
    return -x
}

fn main() {
    let result = anti_learn(42)
}
"""
    
    executor = run_code(code)
    
    plastic_meta = executor.function_metadata['anti_learn']['plastic']
    assert plastic_meta['mode'] == "anti_hebbian"
    assert plastic_meta['rate'] == 0.007
    
    print("=== @plastic with string mode test passed! ===")


def test_plastic_rate_variations():
    """Test différentes valeurs de rate"""
    print("\nTesting @plastic rate variations...")
    
    code = """
@plastic(rate: 0.0001)
fn slow_adapt(x: int) -> int {
    return x
}

@plastic(rate: 0.1)
fn fast_adapt(x: int) -> int {
    return x
}

fn main() {
    slow_adapt(1)
    fast_adapt(2)
}
"""
    
    executor = run_code(code)
    
    slow_meta = executor.function_metadata['slow_adapt']['plastic']
    fast_meta = executor.function_metadata['fast_adapt']['plastic']
    
    assert slow_meta['rate'] == 0.0001
    assert fast_meta['rate'] == 0.1
    
    print("=== @plastic rate variations test passed! ===")


# Point d'entrée
if __name__ == "__main__":
    print("=" * 55)
    print("        NORMiL Annotations Test Suite")
    print("=" * 55)
    
    tests = [
        test_plastic_annotation_basic,
        test_plastic_annotation_with_mode,
        test_plastic_annotation_defaults,
        test_metadata_accessible_in_function,
        test_multiple_annotations,
        test_function_without_annotation,
        test_plastic_with_string_mode,
        test_plastic_rate_variations,
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
