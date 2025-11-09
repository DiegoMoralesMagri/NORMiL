"""
Tests pour la Phase 4.1 - Interopérabilité Python (Importation de modules)
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


class TestPythonInterop(unittest.TestCase):
    """Tests pour l'interopérabilité Python"""
    
    def setUp(self):
        self.executor = Executor()
    
    def run_code(self, code: str):
        """Helper pour exécuter du code NORMiL"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.executor.execute(ast)
    
    def test_import_math_module(self):
        """Test 1: Import du module math de Python"""
        code = """
        import math
        let pi = math.pi
        """
        self.run_code(code)
        
        # Vérifier que pi a été importé
        pi_value = self.executor.current_scope.get_var("pi")
        self.assertAlmostEqual(pi_value, 3.14159265, places=5)
    
    def test_import_math_with_alias(self):
        """Test 2: Import math avec alias"""
        code = """
        import math as m
        let e = m.e
        """
        self.run_code(code)
        
        e_value = self.executor.current_scope.get_var("e")
        self.assertAlmostEqual(e_value, 2.71828, places=4)
    
    def test_python_function_call_sqrt(self):
        """Test 3: Appel d'une fonction Python (sqrt)"""
        code = """
        import math
        let result = math.sqrt(16.0)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertAlmostEqual(result, 4.0)
    
    def test_python_function_call_pow(self):
        """Test 4: Appel de fonction Python avec deux arguments"""
        code = """
        import math
        let result = math.pow(2.0, 3.0)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertAlmostEqual(result, 8.0)
    
    def test_python_function_in_expression(self):
        """Test 5: Fonction Python dans une expression"""
        code = """
        import math
        let result = math.sqrt(25.0) + math.sqrt(9.0)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertAlmostEqual(result, 8.0)  # 5.0 + 3.0
    
    def test_multiple_python_modules(self):
        """Test 6: Import de plusieurs modules Python"""
        code = """
        import math
        import random
        let pi = math.pi
        """
        self.run_code(code)
        
        # Vérifier que les deux modules sont chargés
        self.assertIsNotNone(self.executor.current_scope.get_module("math"))
        self.assertIsNotNone(self.executor.current_scope.get_module("random"))
    
    def test_python_module_constant_in_calculation(self):
        """Test 7: Utiliser une constante Python dans un calcul"""
        code = """
        import math
        let circumference = 2.0 * math.pi * 5.0
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("circumference")
        self.assertAlmostEqual(result, 31.4159, places=3)
    
    def test_python_function_nested_calls(self):
        """Test 8: Appels imbriqués de fonctions Python"""
        code = """
        import math
        let result = math.sqrt(math.pow(3.0, 2.0) + math.pow(4.0, 2.0))
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertAlmostEqual(result, 5.0)  # sqrt(9 + 16) = sqrt(25) = 5
    
    def test_python_function_in_condition(self):
        """Test 9: Fonction Python dans une condition"""
        code = """
        import math
        let is_positive = math.sqrt(16.0) > 0.0
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("is_positive")
        self.assertTrue(result)
    
    def test_python_module_in_function(self):
        """Test 10: Utiliser module Python dans fonction NORMiL"""
        code = """
        import math
        
        fn circle_area(radius: Float) -> Float {
            return math.pi * radius * radius
        }
        
        let area = circle_area(3.0)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("area")
        self.assertAlmostEqual(result, 28.274, places=2)
    
    def test_import_random_module(self):
        """Test 11: Import du module random"""
        code = """
        import random
        random.seed(42)
        let val = random.random()
        """
        self.run_code(code)
        
        val = self.executor.current_scope.get_var("val")
        # Vérifier que c'est un float entre 0 et 1
        self.assertIsInstance(val, float)
        self.assertGreaterEqual(val, 0.0)
        self.assertLessEqual(val, 1.0)
    
    def test_python_module_caching(self):
        """Test 12: Les modules Python sont cachés"""
        code = """
        import math
        let a = math.pi
        import math
        let b = math.pi
        """
        self.run_code(code)
        
        # Les deux variables doivent avoir la même valeur
        a = self.executor.current_scope.get_var("a")
        b = self.executor.current_scope.get_var("b")
        self.assertEqual(a, b)
    
    def test_invalid_python_module(self):
        """Test 13: Erreur si module Python inexistant"""
        code = """
        import nonexistent_module_xyz
        """
        
        with self.assertRaises(Exception) as context:
            self.run_code(code)
        
        # Vérifier que l'erreur mentionne le module
        self.assertIn("nonexistent_module_xyz", str(context.exception))
    
    def test_invalid_python_attribute(self):
        """Test 14: Erreur si attribut Python inexistant"""
        code = """
        import math
        let x = math.nonexistent_attr
        """
        
        with self.assertRaises(Exception) as context:
            self.run_code(code)
        
        self.assertIn("no attribute", str(context.exception).lower())
    
    def test_python_and_normil_modules_together(self):
        """Test 15: Mélanger modules Python et NORMiL"""
        # Créer un module NORMiL
        normil_module = """
        fn double(x: Float) -> Float {
            return x * 2.0
        }
        """
        
        with open("tests/temp_module.nor", "w") as f:
            f.write(normil_module)
        
        try:
            code = """
            import math
            import temp_module as tm
            let result = tm.double(math.pi)
            """
            
            # Ajouter le dossier tests au module_paths
            self.executor.module_paths.insert(0, "tests")
            
            self.run_code(code)
            
            result = self.executor.current_scope.get_var("result")
            self.assertAlmostEqual(result, 6.283, places=2)  # 2 * pi
        finally:
            # Nettoyer
            if os.path.exists("tests/temp_module.nor"):
                os.remove("tests/temp_module.nor")
    
    def test_python_function_with_integer(self):
        """Test 16: Appel fonction Python avec entier"""
        code = """
        import math
        let result = math.factorial(5)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertEqual(result, 120)
    
    def test_python_floor_function(self):
        """Test 17: Fonction Python floor"""
        code = """
        import math
        let result = math.floor(3.7)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertEqual(result, 3)
    
    def test_python_ceil_function(self):
        """Test 18: Fonction Python ceil"""
        code = """
        import math
        let result = math.ceil(3.2)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertEqual(result, 4)
    
    def test_python_trigonometric_function(self):
        """Test 19: Fonctions trigonométriques Python"""
        code = """
        import math
        let result = math.sin(math.pi / 2.0)
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("result")
        self.assertAlmostEqual(result, 1.0, places=5)
    
    def test_python_module_in_loop(self):
        """Test 20: Utiliser module Python dans une boucle"""
        code = """
        import math
        let sum = 0.0
        
        for i in [1.0, 2.0, 3.0, 4.0] {
            sum = sum + math.sqrt(i)
        }
        """
        self.run_code(code)
        
        result = self.executor.current_scope.get_var("sum")
        # sqrt(1) + sqrt(2) + sqrt(3) + sqrt(4) ≈ 1 + 1.414 + 1.732 + 2 = 6.146
        self.assertAlmostEqual(result, 6.146, places=2)


if __name__ == '__main__':
    unittest.main()
