# 🚀 NORMiL - Quick Start Guide

Bienvenue dans **NORMiL**, le langage dédié pour contrôler l'IA O-RedMind !

---

## ⚡ Installation Express (30 secondes)

```bash
# 1. Installer numpy
pip install numpy

# 2. Aller dans le dossier NORMiL
cd openredNetwork/modules/ia2/normil

# 3. Tester !
python normil_cli.py run examples/hello.nor
```

**Résultat attendu** :
```
Bonjour, O-RedMind !
```

✅ **Si vous voyez ce message, NORMiL fonctionne !**

---

## 📖 Votre Premier Programme NORMiL

Créez un fichier `mon_script.nor` :

```normil
// Mon premier script NORMiL
fn main() {
    print("Hello from NORMiL!")
    
    let x = 42
    let y = 10
    let z = x + y
    
    print("42 + 10 =")
    print(z)
}
```

Exécutez :
```bash
python normil_cli.py run mon_script.nor
```

---

## 🎯 Exemples de Base

### 1. Variables et Types

```normil
fn main() {
    let age: int = 25
    let pi: float = 3.14159
    let name: str = "OpenRed"
    let active: bool = true
    
    print(name)
    print(age)
}
```

### 2. Fonctions

```normil
fn double(x: int) -> int {
    return x * 2
}

fn greet(name: str) -> str {
    return "Bonjour, " + name + " !"
}

fn main() {
    let result = double(21)
    print(result)  // 42
    
    let msg = greet("Diego")
    print(msg)  // Bonjour, Diego !
}
```

### 3. Conditions

```normil
fn check_age(age: int) {
    if age >= 18 {
        print("Majeur")
    } else {
        print("Mineur")
    }
}

fn main() {
    check_age(25)  // Majeur
    check_age(15)  // Mineur
}
```

### 4. Boucles

```normil
fn main() {
    // Boucle for
    for i in range(1, 6) {
        print(i)
    }
    // Output: 1 2 3 4 5
    
    // Boucle while
    let count = 0
    while count < 3 {
        print(count)
        count = count + 1
    }
    // Output: 0 1 2
}
```

### 5. Vecteurs (Spécialité NORMiL!)

```normil
fn main() {
    // Créer des vecteurs
    let v1 = zeros(256)      // Vecteur de 256 zéros
    let v2 = ones(256)       // Vecteur de 256 uns
    let v3 = random(256)     // Vecteur aléatoire
    
    // Opérations vectorielles
    let sum = vec_add(v1, v2)
    let diff = vec_sub(v2, v1)
    let prod = vec_mul(v2, v2)
    
    // Produit scalaire
    let similarity = dot(v2, v3)
    print("Similarité:")
    print(similarity)
    
    // Norme
    let magnitude = norm(v2)
    print("Norme:")
    print(magnitude)
}
```

---

## 🧠 Mémoire (Features Avancées)

### Mémoire Épisodique

```normil
fn main() {
    // Créer un vecteur pour l'épisode
    let vec = random(256)
    
    // Ajouter un épisode (simulation)
    // Note: EpisodicRecord nécessite plus de setup
    
    print("Épisode ajouté en mémoire")
}
```

### Consolidation

```normil
fn main() {
    // Convertir plusieurs épisodes en concept
    // (exemple simplifié)
    
    print("Consolidation mémoire en cours...")
}
```

---

## 🛠️ Commandes CLI Utiles

### Exécuter un script
```bash
python normil_cli.py run mon_script.nor
```

### Voir l'AST (arbre syntaxique)
```bash
python normil_cli.py parse mon_script.nor
```

### Voir les tokens
```bash
python normil_cli.py tokenize mon_script.nor
```

---

## 🧪 Tester les Exemples Fournis

```bash
# Hello World
python normil_cli.py run examples/hello.nor

# Opérations vectorielles avancées
# (nécessite arguments nommés - Phase 2)
# python normil_cli.py run examples/memory_operations.nor
```

---

## 📚 Primitives Disponibles (Top 10)

| Primitive | Description | Exemple |
|-----------|-------------|---------|
| `print(x)` | Afficher une valeur | `print("Hello")` |
| `zeros(dim)` | Vecteur de zéros | `let v = zeros(256)` |
| `ones(dim)` | Vecteur de uns | `let v = ones(128)` |
| `random(dim)` | Vecteur aléatoire | `let v = random(512)` |
| `vec_add(v1, v2)` | Addition vectorielle | `let sum = vec_add(v1, v2)` |
| `dot(v1, v2)` | Produit scalaire | `let sim = dot(v1, v2)` |
| `norm(v)` | Norme L2 | `let mag = norm(v)` |
| `normalize(v)` | Normaliser | `let u = normalize(v)` |
| `range(n)` | Séquence 0..n-1 | `for i in range(10)` |
| `str(x)` | Convertir en string | `let s = str(42)` |

**Voir la liste complète** : 45+ primitives dans `runtime/primitives.py`

---

## 🐛 Debugging

### Erreur de syntaxe ?
```bash
python normil_cli.py parse mon_script.nor
```
→ Affichera l'erreur de parsing avec ligne/colonne

### Erreur d'exécution ?
```bash
python normil_cli.py run mon_script.nor
```
→ Traceback Python complet

### Voir les tokens ?
```bash
python normil_cli.py tokenize mon_script.nor
```
→ Liste tous les tokens détectés

---

## 💡 Tips & Tricks

### 1. Toujours définir `main()`
```normil
fn main() {
    // Votre code ici
}
```
Le CLI appelle automatiquement `main()` si elle existe.

### 2. Types explicites recommandés
```normil
// Bon ✅
let x: int = 42

// Fonctionne mais moins clair
let x = 42
```

### 3. Les vecteurs sont typés
```normil
let v1 = zeros(256)  // Vec de dimension 256
let v2 = zeros(128)  // Vec de dimension 128
// vec_add(v1, v2)   // ❌ Erreur: dimensions incompatibles
```

### 4. Attention aux noms de primitives
```normil
// ❌ Éviter de nommer vos fonctions comme les primitives
fn print(x: int) {  // Conflit avec primitive print()
    // ...
}

// ✅ Utilisez des noms différents
fn afficher(x: int) {
    print(x)
}
```

---

## 📖 Ressources

- **README complet** : `README.md`
- **Spécifications** : `SPECIFICATION.md`
- **Rapport MVP** : `MVP_ACHIEVEMENT.md`
- **Tests** : `test_*.py` (4 fichiers)
- **Exemples** : `examples/*.nor`

---

## 🆘 Aide

### Erreurs courantes

**1. `ModuleNotFoundError: No module named 'numpy'`**
```bash
pip install numpy
```

**2. `ParseError: Expected X, got Y`**
→ Vérifier la syntaxe avec `parse` :
```bash
python normil_cli.py parse mon_script.nor
```

**3. `ExecutionError: Undefined function: xxx`**
→ Vérifier que la fonction est définie ou existe dans les primitives

**4. `ValueError: Dimension mismatch`**
→ Les vecteurs doivent avoir la même dimension pour les opérations

---

## 🎓 Tutoriel Complet : Calculatrice

```normil
// calculatrice.nor - Exemple complet

fn add(a: int, b: int) -> int {
    return a + b
}

fn sub(a: int, b: int) -> int {
    return a - b
}

fn mul(a: int, b: int) -> int {
    return a * b
}

fn div(a: int, b: int) -> int {
    return a / b
}

fn main() {
    print("=== Calculatrice NORMiL ===")
    
    let x = 20
    let y = 5
    
    print("Nombres:")
    print(x)
    print(y)
    
    print("Addition:")
    print(add(x, y))
    
    print("Soustraction:")
    print(sub(x, y))
    
    print("Multiplication:")
    print(mul(x, y))
    
    print("Division:")
    print(div(x, y))
}
```

Exécutez :
```bash
python normil_cli.py run calculatrice.nor
```

Résultat :
```
=== Calculatrice NORMiL ===
Nombres:
20
5
Addition:
25
Soustraction:
15
Multiplication:
100
Division:
4
```

---

## 🎉 Félicitations !

Vous maîtrisez maintenant les bases de NORMiL !

**Prochaines étapes** :
1. Créez vos propres scripts
2. Explorez les exemples avancés
3. Expérimentez avec les vecteurs
4. Testez la mémoire épisodique/sémantique

---

**Amusez-vous bien avec NORMiL !** 🚀🧠

