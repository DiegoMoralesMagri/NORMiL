"""
NORMiL Runtime Package
======================

Runtime d'exécution pour NORMiL.

Modules:
- normil_types: Types natifs NORMiL (Vec, EpisodicRecord, etc.)
- primitives: Primitives natives implémentées en Python
- executor: Exécuteur d'AST
- sandbox: Environnement d'exécution sécurisé
"""

from .normil_types import *
from .primitives import *
# from .executor import Executor  # TODO: À activer quand executor créé

__all__ = [
    'Vec',
    'EpisodicRecord',
    'Concept',
    # 'Executor',
]
