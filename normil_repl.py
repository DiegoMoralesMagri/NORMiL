#!/usr/bin/env python3
"""
NORMiL REPL (Read-Eval-Print Loop)
===================================

Interface interactive pour exécuter du code NORMiL ligne par ligne.

Usage:
    python normil_repl.py
    
Commandes spéciales:
    help        - Afficher l'aide
    history     - Afficher l'historique des commandes
    clear       - Effacer l'écran
    reset       - Réinitialiser l'environnement
    exit/quit   - Quitter le REPL
"""
import sys
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
NORMIL_ROOT = Path(__file__).parent
sys.path.insert(0, str(NORMIL_ROOT))

from parser.lexer import Lexer
from parser.parser import Parser, ParseError
from runtime.executor import Executor, ExecutionError


class NORMiLREPL:
    """REPL interactif pour NORMiL"""
    
    def __init__(self):
        self.executor = Executor()
        self.history = []
        self.multiline_buffer = []
        self.in_multiline = False
        
    def print_banner(self):
        """Affiche le banner de bienvenue"""
        print("╔═══════════════════════════════════════════════════════╗")
        print("║         NORMiL REPL v0.1.0                            ║")
        print("║   Langage pour le contrôle de l'IA O-RedMind          ║")
        print("╚═══════════════════════════════════════════════════════╝")
        print()
        print("Tapez 'help' pour l'aide, 'exit' pour quitter")
        print()
    
    def print_help(self):
        """Affiche l'aide"""
        print()
        print("=== Aide NORMiL REPL ===")
        print()
        print("Commandes spéciales:")
        print("  help        - Afficher cette aide")
        print("  history     - Afficher l'historique des commandes")
        print("  clear       - Effacer l'historique (environnement conservé)")
        print("  reset       - Réinitialiser complètement l'environnement")
        print("  exit, quit  - Quitter le REPL")
        print()
        print("Exemples:")
        print("  >>> let x = 42")
        print("  >>> let y = 10")
        print("  >>> print(x + y)")
        print("  52")
        print()
        print("  >>> let v = random(256, mean: 0.5, std: 0.1)")
        print("  >>> let magnitude = norm(v)")
        print("  >>> print(magnitude)")
        print()
        print("Multi-lignes (avec { } ou fn):")
        print("  >>> fn double(x: int) -> int {")
        print("  ...     return x * 2")
        print("  ... }")
        print("  >>> print(double(21))")
        print("  42")
        print()
    
    def show_history(self):
        """Affiche l'historique"""
        print()
        print("=== Historique ===")
        for i, cmd in enumerate(self.history, 1):
            print(f"{i:3d}. {cmd}")
        print()
    
    def reset_environment(self):
        """Réinitialise l'environnement"""
        self.executor = Executor()
        self.history = []
        print("✅ Environnement réinitialisé")
    
    def clear_history(self):
        """Efface l'historique mais conserve l'environnement"""
        self.history = []
        print("✅ Historique effacé")
    
    def execute_code(self, code: str):
        """Exécute du code NORMiL"""
        try:
            # Lexer
            lexer = Lexer(code)
            tokens = lexer.tokenize()
            
            # Parser
            parser = Parser(tokens)
            ast = parser.parse()
            
            # Executor
            for stmt in ast.statements:
                result = self.executor.exec_statement(stmt)
                # Si c'est une expression statement, afficher le résultat
                # (sauf None)
                if result is not None:
                    print(f"=> {result}")
            
            return True
            
        except ParseError as e:
            print(f"❌ Erreur de syntaxe: {e}")
            return False
        except ExecutionError as e:
            print(f"❌ Erreur d'exécution: {e}")
            return False
        except Exception as e:
            print(f"❌ Erreur: {type(e).__name__}: {e}")
            return False
    
    def is_complete_statement(self, code: str) -> bool:
        """Vérifie si le code est une instruction complète"""
        # Simple heuristique: compter les { }
        open_braces = code.count('{')
        close_braces = code.count('}')
        
        if open_braces > close_braces:
            return False
        
        # Vérifier si ça commence par fn, type, transaction
        stripped = code.strip()
        if stripped.startswith(('fn ', 'type ', 'transaction ', 'if ', 'for ', 'while ')):
            # Doit avoir des braces
            if '{' in code and '}' in code:
                return open_braces == close_braces
            elif '{' in code:
                return False
        
        return True
    
    def run(self):
        """Lance le REPL"""
        self.print_banner()
        
        while True:
            try:
                # Prompt
                if self.in_multiline:
                    prompt = "... "
                else:
                    prompt = ">>> "
                
                # Lire l'entrée
                try:
                    line = input(prompt)
                except EOFError:
                    print()
                    break
                
                # Commande vide
                if not line.strip():
                    continue
                
                # Commandes spéciales
                cmd = line.strip().lower()
                
                if cmd in ('exit', 'quit'):
                    print("Au revoir ! 👋")
                    break
                
                if cmd == 'help':
                    self.print_help()
                    continue
                
                if cmd == 'history':
                    self.show_history()
                    continue
                
                if cmd == 'clear':
                    self.clear_history()
                    continue
                
                if cmd == 'reset':
                    self.reset_environment()
                    continue
                
                # Code normal
                self.history.append(line)
                
                # Gérer le mode multi-lignes
                if self.in_multiline:
                    self.multiline_buffer.append(line)
                    full_code = '\n'.join(self.multiline_buffer)
                    
                    if self.is_complete_statement(full_code):
                        # Exécuter le buffer complet
                        self.execute_code(full_code)
                        self.multiline_buffer = []
                        self.in_multiline = False
                else:
                    # Vérifier si c'est multi-lignes
                    if not self.is_complete_statement(line):
                        self.in_multiline = True
                        self.multiline_buffer = [line]
                    else:
                        # Exécuter directement
                        self.execute_code(line)
                
            except KeyboardInterrupt:
                print("\n(KeyboardInterrupt - Tapez 'exit' pour quitter)")
                self.multiline_buffer = []
                self.in_multiline = False
                continue


def main():
    """Point d'entrée du REPL"""
    repl = NORMiLREPL()
    repl.run()


if __name__ == '__main__':
    main()
