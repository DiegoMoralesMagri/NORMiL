"""
NORMiL Parser Package
=====================

Analyseur lexical et syntaxique pour le langage NORMiL.

Modules:
- lexer: Tokenisation du code source
- parser: Construction de l'AST (Abstract Syntax Tree)
- ast_nodes: Définition des nœuds de l'AST
"""

from .lexer import Lexer, Token, TokenType
from .parser import Parser
from .ast_nodes import *

__all__ = [
    'Lexer',
    'Token',
    'TokenType',
    'Parser',
]
