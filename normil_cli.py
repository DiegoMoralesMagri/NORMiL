#!/usr/bin/env python3
"""
NORMiL CLI - Interface en ligne de commande pour exécuter des scripts NORMiL
Usage:
    python normil_cli.py run <script.nor>
    python normil_cli.py parse <script.nor>
    python normil_cli.py tokenize <script.nor>
"""
import sys
import argparse
from pathlib import Path

# Ajouter le répertoire parent au PYTHONPATH
NORMIL_ROOT = Path(__file__).parent
sys.path.insert(0, str(NORMIL_ROOT))

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor


def command_run(script_path: str):
    """Exécuter un script NORMiL"""
    path = Path(script_path)
    if not path.exists():
        print(f"❌ Erreur: Fichier non trouvé '{script_path}'", file=sys.stderr)
        sys.exit(1)
    
    if not path.suffix == '.nor':
        print(f"⚠️  Attention: Le fichier n'a pas l'extension .nor", file=sys.stderr)
    
    try:
        # Lire le code source
        source_code = path.read_text(encoding='utf-8')
        
        # Lexer
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Parser
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Executor
        executor = Executor()
        for stmt in ast.statements:
            executor.exec_statement(stmt)
        
        # Si une fonction main() existe, l'appeler
        main_func = executor.current_scope.get_function('main')
        if main_func:
            # Appeler main() directement
            executor.call_user_function(main_func, [])
        
    except FileNotFoundError:
        print(f"❌ Erreur: Fichier non trouvé '{script_path}'", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        print(f"❌ Erreur d'exécution: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def command_parse(script_path: str):
    """Parser un script NORMiL et afficher l'AST"""
    path = Path(script_path)
    if not path.exists():
        print(f"❌ Erreur: Fichier non trouvé '{script_path}'", file=sys.stderr)
        sys.exit(1)
    
    try:
        source_code = path.read_text(encoding='utf-8')
        
        # Lexer
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Parser
        parser = Parser(tokens)
        ast = parser.parse()
        
        # Afficher l'AST
        print("=== AST ===")
        for i, stmt in enumerate(ast.statements, 1):
            print(f"{i}. {stmt.__class__.__name__}")
            print(f"   {stmt}")
            print()
        
        print(f"✅ Parsing réussi: {len(ast.statements)} statements")
        
    except Exception as e:
        print(f"❌ Erreur de parsing: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def command_tokenize(script_path: str):
    """Tokenizer un script NORMiL et afficher les tokens"""
    path = Path(script_path)
    if not path.exists():
        print(f"❌ Erreur: Fichier non trouvé '{script_path}'", file=sys.stderr)
        sys.exit(1)
    
    try:
        source_code = path.read_text(encoding='utf-8')
        
        # Lexer
        lexer = Lexer(source_code)
        tokens = lexer.tokenize()
        
        # Afficher les tokens
        print("=== TOKENS ===")
        for i, token in enumerate(tokens, 1):
            print(f"{i:4d}. {token.type.name:20s} {repr(token.value):30s} (line {token.line}, col {token.column})")
        
        print(f"\n✅ Tokenization réussie: {len(tokens)} tokens")
        
    except Exception as e:
        print(f"❌ Erreur de tokenization: {type(e).__name__}: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description='NORMiL - Langage pour le contrôle de l\'IA O-RedMind',
        epilog='Exemples:\n'
               '  python normil_cli.py run examples/hello.nor\n'
               '  python normil_cli.py parse examples/memory_operations.nor\n'
               '  python normil_cli.py tokenize examples/pattern_matching.nor',
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Commande à exécuter')
    
    # Commande 'run'
    run_parser = subparsers.add_parser('run', help='Exécuter un script NORMiL')
    run_parser.add_argument('script', help='Chemin vers le fichier .nor')
    
    # Commande 'parse'
    parse_parser = subparsers.add_parser('parse', help='Parser un script et afficher l\'AST')
    parse_parser.add_argument('script', help='Chemin vers le fichier .nor')
    
    # Commande 'tokenize'
    tokenize_parser = subparsers.add_parser('tokenize', help='Tokenizer un script et afficher les tokens')
    tokenize_parser.add_argument('script', help='Chemin vers le fichier .nor')
    
    args = parser.parse_args()
    
    if args.command == 'run':
        command_run(args.script)
    elif args.command == 'parse':
        command_parse(args.script)
    elif args.command == 'tokenize':
        command_tokenize(args.script)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == '__main__':
    main()
