#!/usr/bin/env python3
"""
Tests pour les opérations sur strings (Phase 3.3)
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


def test_string_concatenation():
    """Test concaténation de strings avec +"""
    code = """
    fn main() {
        let result = "Hello" + " " + "World"
        print(result)
    }
    """
    output = run_code(code)
    assert "Hello World" in output


def test_string_concat_with_inference():
    """Test concaténation avec inférence de types"""
    code = """
    fn main() {
        let part1 = "Hello"
        let part2 = "World"
        let combined = part1 + ", " + part2 + "!"
        print(combined)
    }
    """
    output = run_code(code)
    assert "Hello, World!" in output


def test_string_to_string_conversion():
    """Test conversion to_string()"""
    code = """
    fn main() {
        let num = 42
        let text = "Number: " + to_string(num)
        print(text)
    }
    """
    output = run_code(code)
    assert "Number: 42" in output


def test_string_length():
    """Test string_length()"""
    code = """
    fn main() {
        let word = "NORMiL"
        let len_val = string_length(word)
        print(len_val)
    }
    """
    output = run_code(code)
    assert "6" in output


def test_string_upper_lower():
    """Test string_upper() et string_lower()"""
    code = """
    fn main() {
        let lower = string_lower("HELLO")
        let upper = string_upper("world")
        print(lower)
        print(upper)
    }
    """
    output = run_code(code)
    assert "hello" in output
    assert "WORLD" in output


def test_string_substring():
    """Test string_substring()"""
    code = """
    fn main() {
        let full = "Hello, World!"
        let sub = string_substring(full, start: 0, end: 5)
        print(sub)
    }
    """
    output = run_code(code)
    assert "Hello" in output


def test_string_split():
    """Test string_split()"""
    code = """
    fn main() {
        let sentence = "one two three"
        let words = string_split(sentence, delimiter: " ")
        print(string_length(to_string(words)))
    }
    """
    output = run_code(code)
    # Liste convertie en string
    assert output.strip() != ""


def test_string_join():
    """Test string_join()"""
    code = """
    fn main() {
        let parts = ["Hello", "NORMiL", "World"]
        let joined = string_join(parts, separator: " ")
        print(joined)
    }
    """
    output = run_code(code)
    assert "Hello NORMiL World" in output


def test_string_replace():
    """Test string_replace()"""
    code = """
    fn main() {
        let original = "Hello Python"
        let replaced = string_replace(original, old: "Python", new: "NORMiL")
        print(replaced)
    }
    """
    output = run_code(code)
    assert "Hello NORMiL" in output


def test_string_contains():
    """Test string_contains()"""
    code = """
    fn main() {
        let has_world = string_contains("Hello World", substring: "World")
        let has_foo = string_contains("Hello World", substring: "Foo")
        print(has_world)
        print(has_foo)
    }
    """
    output = run_code(code)
    assert "True" in output
    assert "False" in output


def test_string_startswith():
    """Test string_startswith()"""
    code = """
    fn main() {
        let starts = string_startswith("Hello World", prefix: "Hello")
        print(starts)
    }
    """
    output = run_code(code)
    assert "True" in output


def test_string_endswith():
    """Test string_endswith()"""
    code = """
    fn main() {
        let ends = string_endswith("Hello World", suffix: "World")
        print(ends)
    }
    """
    output = run_code(code)
    assert "True" in output


def test_string_trim():
    """Test string_trim()"""
    code = """
    fn main() {
        let padded = "  space  "
        let trimmed = string_trim(padded)
        print(string_length(trimmed))
    }
    """
    output = run_code(code)
    assert "5" in output


def test_string_repeat():
    """Test string_repeat()"""
    code = """
    fn main() {
        let repeated = string_repeat("Ha", count: 3)
        print(repeated)
    }
    """
    output = run_code(code)
    assert "HaHaHa" in output


def test_string_char_at():
    """Test string_char_at()"""
    code = """
    fn main() {
        let word = "Hello"
        let first = string_char_at(word, index: 0)
        let last = string_char_at(word, index: 4)
        print(first)
        print(last)
    }
    """
    output = run_code(code)
    assert "H" in output
    assert "o" in output


def test_string_index_of():
    """Test string_index_of()"""
    code = """
    fn main() {
        let text = "Hello World"
        let index = string_index_of(text, substring: "World")
        let not_found = string_index_of(text, substring: "XYZ")
        print(index)
        print(not_found)
    }
    """
    output = run_code(code)
    assert "6" in output
    assert "-1" in output


def test_string_concat_numbers():
    """Test concaténation strings et nombres convertis"""
    code = """
    fn main() {
        let age = 25
        let height = 180
        let info = "Age: " + to_string(age) + ", Height: " + to_string(height)
        print(info)
    }
    """
    output = run_code(code)
    assert "Age: 25" in output
    assert "Height: 180" in output


def test_string_multiple_operations():
    """Test combinaison de plusieurs opérations"""
    code = """
    fn main() {
        let text = "  hello world  "
        let trimmed = string_trim(text)
        let upper = string_upper(trimmed)
        let replaced = string_replace(upper, old: "WORLD", new: "NORMIL")
        print(replaced)
    }
    """
    output = run_code(code)
    assert "HELLO NORMIL" in output


def test_string_concat_empty():
    """Test concaténation avec chaînes vides"""
    code = """
    fn main() {
        let empty = ""
        let result = empty + "Hello" + empty
        print(result)
    }
    """
    output = run_code(code)
    assert "Hello" in output


def test_string_operations_in_function():
    """Test opérations string dans une fonction"""
    code = """
    fn format_name(first: str, last: str) -> str {
        let full = first + " " + last
        return string_upper(full)
    }
    
    fn main() {
        let result = format_name(first: "john", last: "doe")
        print(result)
    }
    """
    output = run_code(code)
    assert "JOHN DOE" in output


if __name__ == "__main__":
    print("Running string operations tests...")
    
    tests = [
        ("String concatenation", test_string_concatenation),
        ("Concat with inference", test_string_concat_with_inference),
        ("to_string conversion", test_string_to_string_conversion),
        ("String length", test_string_length),
        ("Upper/Lower", test_string_upper_lower),
        ("Substring", test_string_substring),
        ("Split", test_string_split),
        ("Join", test_string_join),
        ("Replace", test_string_replace),
        ("Contains", test_string_contains),
        ("Starts with", test_string_startswith),
        ("Ends with", test_string_endswith),
        ("Trim", test_string_trim),
        ("Repeat", test_string_repeat),
        ("Char at", test_string_char_at),
        ("Index of", test_string_index_of),
        ("Concat numbers", test_string_concat_numbers),
        ("Multiple operations", test_string_multiple_operations),
        ("Concat empty", test_string_concat_empty),
        ("Operations in function", test_string_operations_in_function),
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
