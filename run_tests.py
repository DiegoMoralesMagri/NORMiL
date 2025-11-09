#!/usr/bin/env python3
"""
NORMiL Test Runner
==================

Lance tous les tests unitaires et affiche un rapport.

Usage:
    python run_tests.py
"""
import sys
import subprocess
from pathlib import Path

# Tests à exécuter
TESTS = [
    'test_parser.py',
    'test_primitives.py',
    'test_executor.py',
    'test_named_args.py',
    'tests/test_pattern_matching.py',
    'tests/test_annotations.py',
    'tests/test_atomic.py',
    'tests/test_type_inference.py',
    'tests/test_imports.py',
    'tests/test_string_ops.py',
    'tests/test_indexing.py',
    'tests/test_structs.py',
    'tests/test_python_interop.py',
    'tests/test_python_advanced.py',
    'tests/test_python_conversions.py',
    'tests/test_python_objects.py',
]

def run_test(test_file: str) -> tuple[bool, str]:
    """
    Exécute un fichier de test.
    
    Returns:
        (success: bool, output: str)
    """
    print(f"Running {test_file}...", end=' ')
    
    try:
        result = subprocess.run(
            [sys.executable, test_file],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            print("✅ PASSED")
            return True, result.stdout
        else:
            print("❌ FAILED")
            return False, result.stdout + result.stderr
    
    except subprocess.TimeoutExpired:
        print("⏱️ TIMEOUT")
        return False, "Test timed out after 30 seconds"
    
    except Exception as e:
        print(f"💥 ERROR: {e}")
        return False, str(e)


def main():
    """Point d'entrée principal"""
    print("╔═══════════════════════════════════════════════════════╗")
    print("║         NORMiL Test Suite                             ║")
    print("╚═══════════════════════════════════════════════════════╝")
    print()
    
    results = {}
    
    for test in TESTS:
        if not Path(test).exists():
            print(f"⚠️  {test} not found, skipping")
            results[test] = (False, "File not found")
            continue
        
        success, output = run_test(test)
        results[test] = (success, output)
    
    # Rapport final
    print()
    print("═" * 60)
    print("SUMMARY")
    print("═" * 60)
    
    passed = sum(1 for success, _ in results.values() if success)
    total = len(results)
    
    for test, (success, output) in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status:10s} {test}")
    
    print()
    print(f"Results: {passed}/{total} tests passed")
    
    if passed == total:
        print()
        print("🎉 All tests passed!")
        return 0
    else:
        print()
        print("❌ Some tests failed. See output above for details.")
        
        # Afficher les détails des échecs
        print()
        print("═" * 60)
        print("FAILURE DETAILS")
        print("═" * 60)
        for test, (success, output) in results.items():
            if not success:
                print()
                print(f"--- {test} ---")
                print(output)
        
        return 1


if __name__ == '__main__':
    sys.exit(main())
