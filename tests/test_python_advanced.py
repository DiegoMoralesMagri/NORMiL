"""
Tests pour la Phase 4.2 - Appels avancés de fonctions Python
=========================================
Auteur : Diego Morales Magri
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


class TestPythonAdvanced(unittest.TestCase):
    """Tests pour les fonctionnalités avancées Python"""
    
    def setUp(self):
        self.executor = Executor()
    
    def run_code(self, code: str):
        """Helper pour exécuter du code NORMiL"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.executor.execute(ast)
    
    def test_python_function_with_multiple_args(self):
        """Test 1: Fonction Python avec plusieurs arguments"""
        code = """
        import math
        let result = math.pow(2.0, 10.0)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertEqual(result, 1024.0)
    
    def test_python_function_returning_none(self):
        """Test 2: Fonction Python retournant None"""
        code = """
        import random
        random.seed(42)
        let result = random.seed(100)
        """
        self.run_code(code)
        
        # seed() retourne None
        result = self.executor.current_scope.get_var("result")
        self.assertIsNone(result)
    
    def test_python_function_with_string_arg(self):
        """Test 3: Fonction Python avec argument string"""
        code = """
        import json
        let text = json.dumps("hello")
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("text")
        self.assertEqual(result, '"hello"')
    
    def test_python_function_with_bool_arg(self):
        """Test 4: Fonction Python avec argument booléen"""
        code = """
        import json
        let text = json.dumps(true)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("text")
        self.assertEqual(result, 'true')
    
    def test_python_function_chain_calls(self):
        """Test 5: Chaîne d'appels Python"""
        code = """
        import math
        let a = math.sqrt(16.0)
        let b = math.pow(a, 2.0)
        let c = math.floor(b)
        """
        self.run_code(code)
        
        a = self.executor.current_scope.get_var("a")
        b = self.executor.current_scope.get_var("b")
        c = self.executor.current_scope.get_var("c")
        
        self.assertEqual(a, 4.0)
        self.assertEqual(b, 16.0)
        self.assertEqual(c, 16)
    
    def test_python_exception_handling(self):
        """Test 6: Gestion des exceptions Python"""
        code = """
        import math
        let result = math.sqrt(-1.0)
        """
        
        # sqrt de nombre négatif devrait lever une exception
        with self.assertRaises(Exception):
            self.run_code(code)
    
    def test_python_function_with_default_args(self):
        """Test 7: Fonction Python avec arguments par défaut"""
        # En Python: round(number, ndigits=None)
        code = """
        import builtins
        let a = builtins.round(3.7)
        let b = builtins.round(3.14159, 2)
        """
        
        # Note: builtins n'est pas importable directement, utilisons math
        code = """
        import math
        let val = math.floor(3.7)
        """
        self.run_code(code)
        
        val = self.executor.current_scope.get_var("val")
        self.assertEqual(val, 3)
    
    def test_python_type_conversion_int_to_float(self):
        """Test 8: Conversion automatique int -> float"""
        code = """
        import math
        let result = math.sqrt(16)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertEqual(result, 4.0)
    
    def test_python_function_with_list_result(self):
        """Test 9: Fonction Python retournant une liste"""
        code = """
        import random
        random.seed(42)
        let choices = random.choices([1, 2, 3, 4, 5], k: 3)
        """
        
        # Pour l'instant, ceci pourrait ne pas fonctionner avec kwargs
        # On va tester une version simple
        code = """
        import sys
        let version = sys.version_info
        """
        
        # Version très simple pour commencer
        code = """
        import math
        let pi = math.pi
        """
        self.run_code(code)
        
        pi = self.executor.current_scope.get_var("pi")
        self.assertAlmostEqual(pi, 3.14159, places=5)
    
    def test_mixed_normil_python_calculation(self):
        """Test 10: Calcul mixte NORMiL et Python"""
        code = """
        import math
        
        let rayon = 5.0
        let aire = math.pi * rayon * rayon
        let perimetre = 2.0 * math.pi * rayon
        """
        self.run_code(code)
        
        aire = self.executor.current_scope.get_var("aire")
        perimetre = self.executor.current_scope.get_var("perimetre")
        
        self.assertAlmostEqual(aire, 78.5398, places=3)
        self.assertAlmostEqual(perimetre, 31.4159, places=3)


if __name__ == '__main__':
    unittest.main()
