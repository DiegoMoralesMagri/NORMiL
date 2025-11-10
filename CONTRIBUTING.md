# 🤝 Contributing to NORMiL


**Date** : Novembre 2025
**Auteur** : Diego Morales Magri

---

Merci de votre intérêt pour contribuer à NORMiL ! Ce document vous guide dans le processus de contribution.

---

## 📋 Table des Matières

1. [Code of Conduct](#code-of-conduct)
2. [Comment Contribuer](#comment-contribuer)
3. [Setup Développement](#setup-développement)
4. [Architecture](#architecture)
5. [Workflow Git](#workflow-git)
6. [Tests](#tests)
7. [Documentation](#documentation)
8. [Pull Request Process](#pull-request-process)

---

## 🌟 Code of Conduct

- Soyez respectueux et constructif
- Accueillez les nouveaux contributeurs
- Focalisez sur le problème, pas la personne
- Documentez vos changements

---

## 🚀 Comment Contribuer

### Types de Contributions Bienvenues

1. **Bugs** : Signaler et corriger des bugs
2. **Features** : Proposer et implémenter de nouvelles fonctionnalités
3. **Documentation** : Améliorer README, guides, commentaires
4. **Tests** : Ajouter des tests unitaires/intégration
5. **Exemples** : Créer des exemples `.nor` utiles
6. **Performance** : Optimiser le code existant

### Trouver un Sujet

- Consultez les [Issues](https://github.com/DiegoMoralesMagri/OpenRed/issues)
- Regardez les TODOs dans le code
- Consultez la [Roadmap](README.md#roadmap)
- Proposez vos idées !

---

## 🛠️ Setup Développement

### Prérequis

```bash
Python 3.9+
NumPy
Git
```

### Installation

```bash
# 1. Fork le repo
git clone https://github.com/<votre-username>/OpenRed.git

# 2. Aller dans le dossier NORMiL
cd OpenRed/openredNetwork/modules/ia2/normil

# 3. Créer un environnement virtuel
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# ou
.venv\Scripts\activate  # Windows

# 4. Installer les dépendances
pip install numpy

# 5. Vérifier que tout fonctionne
python test_executor.py
```

### Structure du Projet

```
normil/
├── parser/          # Analyse lexicale et syntaxique
│   ├── lexer.py
│   ├── parser.py
│   └── ast_nodes.py
├── runtime/         # Environnement d'exécution
│   ├── executor.py
│   ├── normil_types.py
│   └── primitives.py
├── examples/        # Exemples de code NORMiL
├── tests/           # Tests unitaires
└── docs/            # Documentation
```

---

## 🏗️ Architecture

### Pipeline d'Exécution

```
Code NORMiL (.nor)
    ↓
Lexer (tokens)
    ↓
Parser (AST)
    ↓
Executor (Python runtime)
    ↓
Résultat
```

### Composants Clés

| Composant            | Fichier                     | Responsabilité    |
| -------------------- | --------------------------- | ------------------ |
| **Lexer**      | `parser/lexer.py`         | Tokenization       |
| **Parser**     | `parser/parser.py`        | Construction AST   |
| **AST**        | `parser/ast_nodes.py`     | Nœuds syntaxiques |
| **Executor**   | `runtime/executor.py`     | Interprétation    |
| **Types**      | `runtime/normil_types.py` | Types natifs       |
| **Primitives** | `runtime/primitives.py`   | Fonctions built-in |

### Ajouter une Feature

#### 1. Nouveau Token (Lexer)

```python
# parser/lexer.py
class TokenType(Enum):
    # ... existing tokens
    MY_NEW_TOKEN = "MY_NEW_TOKEN"

# Dans tokenize():
elif self.current_char == '@':
    return Token(TokenType.MY_NEW_TOKEN, '@', ...)
```

#### 2. Nouveau Nœud AST

```python
# parser/ast_nodes.py
@dataclass
class MyNewNode(Statement):
    """Description du nœud"""
    field1: str
    field2: Expression
  
    def __repr__(self) -> str:
        return f"MyNewNode({self.field1}, {self.field2})"
```

#### 3. Parser le Nœud

```python
# parser/parser.py
def parse_my_new_statement(self) -> MyNewNode:
    """Parse MY_NEW_TOKEN statement"""
    self.expect(TokenType.MY_NEW_TOKEN)
    field1 = self.expect(TokenType.IDENTIFIER).value
    field2 = self.parse_expression()
    return MyNewNode(field1, field2)
```

#### 4. Exécuter le Nœud

```python
# runtime/executor.py
def exec_statement(self, stmt: Statement) -> Any:
    # ... existing cases
    elif isinstance(stmt, MyNewNode):
        return self.exec_my_new_node(stmt)

def exec_my_new_node(self, stmt: MyNewNode) -> Any:
    """Exécute MyNewNode"""
    # Votre logique ici
    pass
```

#### 5. Tester

```python
# test_my_feature.py
def test_my_new_feature():
    code = "@mynew identifier expression"
    ast = parse_code(code)
    result = execute_ast(ast)
    assert result == expected_value
```

---

## 🔀 Workflow Git

### Créer une Branche

```bash
git checkout -b feature/ma-nouvelle-feature
# ou
git checkout -b fix/correction-bug-123
```

### Commits

Utilisez des messages descriptifs :

```bash
# ✅ Bon
git commit -m "feat: Ajoute support pour les tuples"
git commit -m "fix: Corrige crash avec arguments nommés vides"
git commit -m "docs: Améliore documentation du REPL"
git commit -m "test: Ajoute tests pour pattern matching"

# ❌ Mauvais
git commit -m "update"
git commit -m "fix bug"
git commit -m "wip"
```

### Convention de Messages

Préfixes recommandés :

- `feat:` Nouvelle fonctionnalité
- `fix:` Correction de bug
- `docs:` Documentation
- `test:` Tests
- `refactor:` Refactoring
- `perf:` Optimisation performance
- `style:` Formatage, style

---

## 🧪 Tests

### Lancer les Tests

```bash
# Tous les tests
python test_lexer.py
python test_parser.py
python test_primitives.py
python test_executor.py
python test_named_args.py

# Ou via script (à créer)
./run_tests.sh
```

### Écrire des Tests

```python
# test_my_feature.py
import sys
sys.path.insert(0, 'path/to/normil')

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor

def test_ma_feature():
    """Test de ma nouvelle feature"""
    code = """
    let x = my_new_feature(42)
    print(x)
    """
  
    # Setup
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    executor = Executor()
  
    # Execute
    for stmt in ast.statements:
        executor.exec_statement(stmt)
  
    # Assert
    assert executor.current_scope.get_var('x') == expected_value
  
if __name__ == '__main__':
    test_ma_feature()
    print("✅ Tests passed!")
```

### Couverture de Tests

Visez au minimum :

- **Nouveaux tokens** : 1 test
- **Nouveaux nœuds AST** : 2-3 tests (parsing + execution)
- **Nouvelles primitives** : 3-5 tests (cas nominal, edge cases)
- **Bug fixes** : 1 test de non-régression

---

## 📚 Documentation

### Docstrings

```python
def my_function(param1: int, param2: str) -> bool:
    """
    Description courte de la fonction.
  
    Description plus détaillée si nécessaire.
    Peut tenir sur plusieurs lignes.
  
    Args:
        param1: Description du paramètre 1
        param2: Description du paramètre 2
  
    Returns:
        Description du retour
  
    Raises:
        ValueError: Si param1 < 0
  
    Example:
        >>> my_function(42, "hello")
        True
    """
    if param1 < 0:
        raise ValueError("param1 must be >= 0")
    return param1 > 0 and len(param2) > 0
```

### Exemples `.nor`

Créez des exemples clairs et commentés :

```normil
// ============================================
// my_example.nor
// Démonstration de ma nouvelle feature
// ============================================

// Fonction exemple
fn example_function(x: int) -> int {
    // Utilise la nouvelle feature
    let result = my_new_feature(x)
    return result * 2
}

// Point d'entrée
fn main() {
    let value = example_function(21)
    print(value)  // Devrait afficher 42
}
```

---

## 🔄 Pull Request Process

### Avant de Soumettre

Checklist :

- [ ] Code suit le [Style Guide](STYLE_GUIDE.md)
- [ ] Tests passent tous
- [ ] Documentation mise à jour
- [ ] Exemples ajoutés si nécessaire
- [ ] Pas de code commenté inutile
- [ ] Pas de `print()` de debug

### Créer la PR

1. **Push votre branche**

   ```bash
   git push origin feature/ma-feature
   ```
2. **Créer la Pull Request** sur GitHub

   - Titre clair et descriptif
   - Description détaillée :
     - Quoi : Qu'est-ce qui change ?
     - Pourquoi : Pourquoi ce changement ?
     - Comment : Comment avez-vous implémenté ?
   - Screenshots/exemples si applicable
   - Référencer les issues liées
3. **Template de PR**

```markdown
## Description

Ajoute le support pour [feature X] qui permet [objectif Y].

## Motivation

Actuellement, NORMiL ne supporte pas [X]. Cette PR résout ce problème
en implémentant [solution Z].

## Changements

- Ajout de `TokenType.NEW_TOKEN` dans lexer
- Nouveau nœud AST `NewFeatureNode`
- Implémentation dans executor
- Tests complets ajoutés

## Tests

- [x] test_lexer.py passe
- [x] test_parser.py passe
- [x] test_executor.py passe
- [x] test_new_feature.py ajouté (3 tests)

## Checklist

- [x] Code suit le style guide
- [x] Documentation mise à jour
- [x] Exemples ajoutés
- [ ] Review par au moins 1 personne
```

### Review Process

- Soyez patient et ouvert aux commentaires
- Répondez aux questions de review
- Appliquez les suggestions pertinentes
- Re-demandez une review après modifications

---

## 🎯 Priorités Actuelles

Consultez la [Roadmap](README.md#roadmap) pour les priorités.

### Contributions Faciles pour Débuter

- 🟢 **Ajouter des exemples** `.nor`
- 🟢 **Améliorer la documentation**
- 🟢 **Ajouter des tests** unitaires
- 🟡 **Corriger des bugs** simples
- 🔴 **Implémenter de nouvelles features**

---

## 📞 Besoin d'Aide ?

- Consultez la [documentation](README.md)
- Lisez le [QUICKSTART](QUICKSTART.md)
- Posez vos questions dans les Issues
- Contactez les mainteneurs

---

## 🏆 Contributeurs

Merci à tous ceux qui ont contribué à NORMiL !

<!-- Liste mise à jour automatiquement -->

---

## 📜 Licence

En contribuant à NORMiL, vous acceptez que vos contributions soient
sous la même licence que le projet (voir LICENSE).

---

**Merci de contribuer à NORMiL !** 🚀

Ensemble, créons le meilleur langage pour contrôler l'IA O-RedMind. 🧠❤️
