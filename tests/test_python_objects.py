"""
Tests pour la Phase 4.4 - Accès aux objets Python
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


class TestPythonObjects(unittest.TestCase):
    """Tests pour l'accès aux objets et classes Python"""
    
    def setUp(self):
        self.executor = Executor()
    
    def run_code(self, code: str):
        """Helper pour exécuter du code NORMiL"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.executor.execute(ast)
    
    def test_string_method_upper(self):
        """Test 1: Méthode upper() sur une chaîne"""
        code = """
        let text = "hello"
        let upper_text = text.upper()
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("upper_text")
        self.assertEqual(result, "HELLO")
    
    def test_string_method_lower(self):
        """Test 2: Méthode lower() sur une chaîne"""
        code = """
        let text = "WORLD"
        let lower_text = text.lower()
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("lower_text")
        self.assertEqual(result, "world")
    
    def test_string_method_replace(self):
        """Test 3: Méthode replace() avec arguments"""
        code = """
        let text = "hello world"
        let new_text = text.replace("world", "NORMiL")
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("new_text")
        self.assertEqual(result, "hello NORMiL")
    
    def test_list_method_append(self):
        """Test 4: Méthode append() sur une liste"""
        code = """
        let items = [1, 2, 3]
        items.append(4)
        """
        self.run_code(code)
        
        items = self.executor.current_scope.get_var("items")
        self.assertEqual(items, [1, 2, 3, 4])
    
    def test_string_method_split(self):
        """Test 5: Méthode split() retournant une liste"""
        code = """
        let text = "one,two,three"
        let parts = text.split(",")
        """
        self.run_code(code)
        
        parts = self.executor.current_scope.get_var("parts")
        self.assertEqual(parts, ["one", "two", "three"])
    
    def test_string_method_startswith(self):
        """Test 6: Méthode startswith() retournant un booléen"""
        code = """
        let text = "hello world"
        let starts = text.startswith("hello")
        let not_starts = text.startswith("goodbye")
        """
        self.run_code(code)
        
        starts = self.executor.current_scope.get_var("starts")
        not_starts = self.executor.current_scope.get_var("not_starts")
        
        self.assertTrue(starts)
        self.assertFalse(not_starts)
    
    def test_chained_methods(self):
        """Test 7: Chaînage de méthodes"""
        code = """
        let text = "  hello world  "
        let result = text.strip().upper()
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertEqual(result, "HELLO WORLD")
    
    def test_datetime_class_instantiation(self):
        """Test 8: Instantiation d'une classe Python"""
        code = """
        import datetime
        let date = datetime.datetime(2024, 1, 1)
        """
        self.run_code(code)
        
        # Vérifier que l'objet est créé
        date = self.executor.current_scope.get_var("date")
        self.assertIsNotNone(date)
        # C'est un objet datetime.datetime Python
        import datetime
        self.assertIsInstance(date, datetime.datetime)
    
    def test_datetime_method_call(self):
        """Test 9: Appel de méthode sur un objet datetime"""
        code = """
        import datetime
        let date = datetime.datetime(2024, 6, 15)
        let weekday = date.weekday()
        """
        self.run_code(code)
        
        # weekday() retourne 0-6 (Lundi-Dimanche)
        weekday = self.executor.current_scope.get_var("weekday")
        self.assertIsInstance(weekday, int)
        self.assertIn(weekday, range(7))
    
    def test_object_attribute_access(self):
        """Test 10: Accès à un attribut d'objet"""
        code = """
        import datetime
        let date = datetime.datetime(2024, 3, 15)
        let year = date.year
        let month = date.month
        let day = date.day
        """
        self.run_code(code)
        
        year = self.executor.current_scope.get_var("year")
        month = self.executor.current_scope.get_var("month")
        day = self.executor.current_scope.get_var("day")
        
        self.assertEqual(year, 2024)
        self.assertEqual(month, 3)
        self.assertEqual(day, 15)


if __name__ == '__main__':
    unittest.main()
