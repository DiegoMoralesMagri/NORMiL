#!/usr/bin/env python3
"""
Tests pour le Pattern Matching de NORMiL
=========================================

Teste les différents types de patterns:
- Literal patterns (42, "hello", true)
- Wildcard pattern (_)
- Type extraction (int(x), float(f), str(s), bool(b))
- Conditions where
- Struct patterns (EpisodicRecord(e))
- Binding simple (case x)
"""

import sys
from pathlib import Path
from io import StringIO

# Ajouter le dossier parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor
from runtime.normil_types import EpisodicRecord, Vec
import numpy as np


def run_code(code: str, capture_print: bool = True):
    """Execute NORMiL code and optionally capture print output"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    executor = Executor()
    
    if capture_print:
        # Capturer stdout
        old_stdout = sys.stdout
        sys.stdout = captured_output = StringIO()
        
        try:
            executor.execute(ast)
            output = captured_output.getvalue()
            lines = [line.strip() for line in output.strip().split('\n') if line.strip()]
            return lines
        finally:
            sys.stdout = old_stdout
    else:
        executor.execute(ast)
        return executor


def test_literal_patterns():
    """Test pattern matching sur littéraux"""
    print("Testing literal patterns...")
    
    code = """
fn classify_number(n: int) -> str {
    match n {
        case 0 -> {
            return "zero"
        }
        case 1 -> {
            return "one"
        }
        case 42 -> {
            return "answer"
        }
        case _ -> {
            return "other"
        }
    }
}

fn main() {
    print(classify_number(0))
    print(classify_number(1))
    print(classify_number(42))
    print(classify_number(99))
}
"""
    
    result = run_code(code)
    assert result == ["zero", "one", "answer", "other"], f"Expected ['zero', 'one', 'answer', 'other'], got {result}"
    print("=== Literal patterns test passed! ===")


def test_wildcard_pattern():
    """Test pattern wildcard (_)"""
    print("\nTesting wildcard pattern...")
    
    code = """
fn is_special(n: int) -> str {
    match n {
        case 7 -> {
            return "lucky"
        }
        case 13 -> {
            return "unlucky"
        }
        case _ -> {
            return "normal"
        }
    }
}

fn main() {
    print(is_special(7))
    print(is_special(13))
    print(is_special(5))
}
"""
    
    result = run_code(code)
    assert result == ["lucky", "unlucky", "normal"], f"Expected ['lucky', 'unlucky', 'normal'], got {result}"
    print("=== Wildcard pattern test passed! ===")


def test_type_extraction():
    """Test extraction de type: int(x), float(f), str(s), bool(b)"""
    print("\nTesting type extraction patterns...")
    
    # Note: NORMiL doesn't have 'any' type yet, so we'll test with explicit type
    # For real polymorphism, we'd need 'any' type support
    code = """
fn describe_int(value: int) -> str {
    match value {
        case int(x) -> {
            if x > 0 {
                return "positive"
            }
            if x < 0 {
                return "negative"
            }
            return "zero"
        }
        case _ -> {
            return "unknown"
        }
    }
}

fn main() {
    print(describe_int(42))
    print(describe_int(-5))
    print(describe_int(0))
}
"""
    
    result = run_code(code)
    assert result == ["positive", "negative", "zero"], f"Expected ['positive', 'negative', 'zero'], got {result}"
    print("=== Type extraction test passed! ===")


def test_where_conditions():
    """Test conditions where dans les patterns"""
    print("\nTesting where conditions...")
    
    code = """
fn classify_score(score: float) -> str {
    match score {
        case float(s) where s >= 0.9 -> {
            return "excellent"
        }
        case float(s) where s >= 0.7 -> {
            return "good"
        }
        case float(s) where s >= 0.5 -> {
            return "average"
        }
        case _ -> {
            return "poor"
        }
    }
}

fn main() {
    print(classify_score(0.95))
    print(classify_score(0.8))
    print(classify_score(0.6))
    print(classify_score(0.3))
}
"""
    
    result = run_code(code)
    assert result == ["excellent", "good", "average", "poor"], f"Expected ['excellent', 'good', 'average', 'poor'], got {result}"
    print("=== Where conditions test passed! ===")


def test_simple_binding():
    """Test binding simple: case x"""
    print("\nTesting simple binding...")
    
    code = """
fn double_value(n: int) -> int {
    match n {
        case x -> {
            return x + x
        }
    }
}

fn main() {
    print(double_value(5))
    print(double_value(10))
    print(double_value(21))
}
"""
    
    result = run_code(code)
    assert result == ["10", "20", "42"], f"Expected ['10', '20', '42'], got {result}"
    print("=== Simple binding test passed! ===")


def test_combined_patterns():
    """Test combinaison de plusieurs types de patterns"""
    print("\nTesting combined patterns...")
    
    code = """
fn analyze_number(n: int) -> str {
    match n {
        case 0 -> {
            return "zero literal"
        }
        case int(x) where x > 100 -> {
            return "big number"
        }
        case int(x) where x > 0 -> {
            return "positive number"
        }
        case int(x) -> {
            return "negative number"
        }
    }
}

fn main() {
    print(analyze_number(0))
    print(analyze_number(150))
    print(analyze_number(50))
    print(analyze_number(-10))
}
"""
    
    result = run_code(code)
    assert result == ["zero literal", "big number", "positive number", "negative number"], \
        f"Expected ['zero literal', 'big number', 'positive number', 'negative number'], got {result}"
    print("=== Combined patterns test passed! ===")


def test_string_patterns():
    """Test pattern matching sur strings"""
    print("\nTesting string patterns...")
    
    code = """
fn greet_language(lang: str) -> str {
    match lang {
        case "fr" -> {
            return "Bonjour"
        }
        case "en" -> {
            return "Hello"
        }
        case "es" -> {
            return "Hola"
        }
        case str(s) -> {
            return "Unknown: " + s
        }
    }
}

fn main() {
    print(greet_language("fr"))
    print(greet_language("en"))
    print(greet_language("de"))
}
"""
    
    result = run_code(code)
    assert result == ["Bonjour", "Hello", "Unknown: de"], f"Expected ['Bonjour', 'Hello', 'Unknown: de'], got {result}"
    print("=== String patterns test passed! ===")


def test_boolean_patterns():
    """Test pattern matching sur booléens"""
    print("\nTesting boolean patterns...")
    
    code = """
fn describe_bool(value: bool) -> str {
    match value {
        case true -> {
            return "yes"
        }
        case false -> {
            return "no"
        }
    }
}

fn main() {
    print(describe_bool(true))
    print(describe_bool(false))
}
"""
    
    result = run_code(code)
    assert result == ["yes", "no"], f"Expected ['yes', 'no'], got {result}"
    print("=== Boolean patterns test passed! ===")


# Point d'entrée
if __name__ == "__main__":
    print("=" * 55)
    print("        NORMiL Pattern Matching Test Suite")
    print("=" * 55)
    
    tests = [
        test_literal_patterns,
        test_wildcard_pattern,
        test_type_extraction,
        test_where_conditions,
        test_simple_binding,
        test_combined_patterns,
        test_string_patterns,
        test_boolean_patterns,
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
            failed += 1
    
    print("\n" + "=" * 55)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed} tests FAILED")
        sys.exit(1)
    else:
        print("         ALL TESTS PASSED!")
    print("=" * 55)
