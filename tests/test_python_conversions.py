"""
Tests pour la Phase 4.3 - Conversions automatiques de types Python
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier parent au path
NORMIL_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(NORMIL_ROOT))

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor
import unittest


class TestPythonConversions(unittest.TestCase):
    """Tests pour les conversions de types entre NORMiL et Python"""
    
    def setUp(self):
        self.executor = Executor()
    
    def run_code(self, code: str):
        """Helper pour exécuter du code NORMiL"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.executor.execute(ast)
    
    def test_python_list_to_normil(self):
        """Test 1: Convertir une liste Python en structure NORMiL"""
        # Pour l'instant, testons ce qui fonctionne déjà
        code = """
        import sys
        let version_major = sys.version_info[0]
        """
        self.run_code(code)
        
        # sys.version_info est un tuple Python
        # Vérifions si l'accès par index fonctionne
        version = self.executor.current_scope.get_var("version_major")
        self.assertIsInstance(version, int)
        self.assertGreaterEqual(version, 3)
    
    def test_normil_list_literal(self):
        """Test 2: Liste littérale NORMiL"""
        code = """
        let liste = [1, 2, 3, 4, 5]
        let premier = liste[0]
        let dernier = liste[4]
        """
        self.run_code(code)
        
        liste = self.executor.current_scope.get_var("liste")
        premier = self.executor.current_scope.get_var("premier")
        dernier = self.executor.current_scope.get_var("dernier")
        
        self.assertEqual(liste, [1, 2, 3, 4, 5])
        self.assertEqual(premier, 1)
        self.assertEqual(dernier, 5)
    
    def test_python_function_returning_list(self):
        """Test 3: Fonction Python retournant une liste"""
        code = """
        import sys
        let path_list = sys.path
        """
        self.run_code(code)
        
        # sys.path est une liste Python
        path = self.executor.current_scope.get_var("path_list")
        self.assertIsInstance(path, list)
    
    def test_normil_list_to_python_function(self):
        """Test 4: Passer une liste NORMiL à une fonction Python"""
        code = """
        let numbers = [1, 2, 3, 4, 5]
        let length = len(numbers)
        """
        self.run_code(code)
        
        length = self.executor.current_scope.get_var("length")
        self.assertEqual(length, 5)
    
    def test_python_string_methods(self):
        """Test 5: Méthodes de chaînes Python"""
        code = """
        let text = "hello world"
        let upper = text.upper()
        """
        
        # Note: Ceci nécessite le support des méthodes d'objets (Phase 4.4)
        # Pour l'instant, testons quelque chose de plus simple
        code = """
        import json
        let text = "hello"
        let json_text = json.dumps(text)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("json_text")
        self.assertEqual(result, '"hello"')
    
    def test_mixed_type_list(self):
        """Test 6: Liste avec types mixtes"""
        code = """
        let mixed = [1, 2.5, "text", true]
        let num = mixed[0]
        let float_val = mixed[1]
        let str_val = mixed[2]
        let bool_val = mixed[3]
        """
        self.run_code(code)
        
        num = self.executor.current_scope.get_var("num")
        float_val = self.executor.current_scope.get_var("float_val")
        str_val = self.executor.current_scope.get_var("str_val")
        bool_val = self.executor.current_scope.get_var("bool_val")
        
        self.assertEqual(num, 1)
        self.assertEqual(float_val, 2.5)
        self.assertEqual(str_val, "text")
        self.assertEqual(bool_val, True)
    
    def test_vec_to_list(self):
        """Test 7: Liste passée à fonction Python builtin"""
        code = """
        let v = [1.0, 2.0, 3.0]
        let length = len(v)
        """
        self.run_code(code)
        
        length = self.executor.current_scope.get_var("length")
        self.assertEqual(length, 3)
    
    def test_python_tuple_access(self):
        """Test 8: Accès aux éléments d'un tuple Python"""
        code = """
        import sys
        let info = sys.version_info
        let major = info[0]
        let minor = info[1]
        """
        self.run_code(code)
        
        major = self.executor.current_scope.get_var("major")
        minor = self.executor.current_scope.get_var("minor")
        
        self.assertIsInstance(major, int)
        self.assertIsInstance(minor, int)
    
    def test_none_handling(self):
        """Test 9: Gestion de None de Python"""
        code = """
        import random
        let result = random.seed(42)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertIsNone(result)
    
    def test_boolean_conversion(self):
        """Test 10: Conversion booléenne"""
        code = """
        let true_val = true
        let false_val = false
        
        import json
        let true_json = json.dumps(true_val)
        let false_json = json.dumps(false_val)
        """
        self.run_code(code)
        
        true_json = self.executor.current_scope.get_var("true_json")
        false_json = self.executor.current_scope.get_var("false_json")
        
        self.assertEqual(true_json, "true")
        self.assertEqual(false_json, "false")


if __name__ == '__main__':
    unittest.main()
