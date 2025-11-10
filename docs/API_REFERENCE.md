# Référence API NORMiL v0.6.0


**Date** : Novembre 2025
**Auteur** : Diego Morales Magri

---

## Table des Matières

1. [Primitives Vectorielles](#primitives-vectorielles)
2. [Opérations Vectorielles](#opérations-vectorielles)
3. [Opérations sur Chaînes](#opérations-sur-chaînes) ✨ **Phase 3.3**
4. [Métriques](#métriques)
5. [Annotations](#annotations)
6. [Pattern Matching](#pattern-matching)
7. [Système de Modules](#système-de-modules) ✨ **Phase 3.2**
8. [Interopérabilité Python](#interopérabilité-python) 🚀 **Phase 4 Complète**
   - Phase 4.1: Import de modules
   - Phase 4.2: Appels de fonctions
   - Phase 4.3: Conversions de types
   - Phase 4.4: Objets et classes
9. [Inférence de Types](#inférence-de-types) ✨ **Phase 3.1**
10. [Fonctions Built-in](#fonctions-built-in)
11. [Types](#types)

---

## Primitives Vectorielles

### `zeros(dim: int) -> Vec`

Crée un vecteur de zéros.

**Paramètres:**

- `dim`: Dimension du vecteur (entier positif)

**Retour:**

- Vecteur de dimension `dim` avec toutes les valeurs à 0.0

**Exemple:**

```normil
let v = zeros(dim: 128)
// v = [0.0, 0.0, ..., 0.0]  (128 éléments)
```

**Erreurs possibles:**

- `dim` doit être > 0
- `dim` doit être un entier

---

### `ones(dim: int) -> Vec`

Crée un vecteur de uns.

**Paramètres:**

- `dim`: Dimension du vecteur

**Retour:**

- Vecteur de dimension `dim` avec toutes les valeurs à 1.0

**Exemple:**

```normil
let v = ones(dim: 64)
// v = [1.0, 1.0, ..., 1.0]  (64 éléments)
// norm(v) ≈ 8.0
```

---

### `fill(dim: int, value: float) -> Vec`

Crée un vecteur rempli avec une valeur spécifique.

**Paramètres:**

- `dim`: Dimension du vecteur
- `value`: Valeur de remplissage

**Retour:**

- Vecteur de dimension `dim` avec toutes les valeurs égales à `value`

**Exemple:**

```normil
let v = fill(dim: 32, value: 0.5)
// v = [0.5, 0.5, ..., 0.5]  (32 éléments)
```

---

### `random(dim: int, mean: float, std: float) -> Vec`

Crée un vecteur avec des valeurs aléatoires suivant une distribution normale.

**Paramètres:**

- `dim`: Dimension du vecteur
- `mean`: Moyenne de la distribution
- `std`: Écart-type de la distribution

**Retour:**

- Vecteur de dimension `dim` avec valeurs ~ N(mean, std²)

**Exemple:**

```normil
let v = random(dim: 256, mean: 0.0, std: 1.0)
// Vecteur de 256 valeurs ~ N(0, 1)
```

**Notes:**

- Utilise la distribution normale (Gaussian)
- Implémentation: NumPy `np.random.normal(mean, std, dim)`

---

## Opérations Vectorielles

### `vec_add(v1: Vec, v2: Vec) -> Vec`

Addition élément par élément de deux vecteurs.

**Paramètres:**

- `v1`: Premier vecteur
- `v2`: Second vecteur

**Retour:**

- Nouveau vecteur où result[i] = v1[i] + v2[i]

**Exemple:**

```normil
let v1 = fill(dim: 64, value: 1.0)
let v2 = fill(dim: 64, value: 2.0)
let sum = vec_add(v1, v2)
// sum = [3.0, 3.0, ..., 3.0]
```

**Erreurs:**

- Les vecteurs doivent avoir la même dimension

---

### `vec_sub(v1: Vec, v2: Vec) -> Vec`

Soustraction élément par élément.

**Paramètres:**

- `v1`: Premier vecteur
- `v2`: Second vecteur

**Retour:**

- Nouveau vecteur où result[i] = v1[i] - v2[i]

**Exemple:**

```normil
let v1 = fill(dim: 64, value: 5.0)
let v2 = fill(dim: 64, value: 2.0)
let diff = vec_sub(v1, v2)
// diff = [3.0, 3.0, ..., 3.0]
```

---

### `vec_mul(v1: Vec, v2: Vec) -> Vec`

Multiplication élément par élément (produit de Hadamard).

**Paramètres:**

- `v1`: Premier vecteur
- `v2`: Second vecteur

**Retour:**

- Nouveau vecteur où result[i] = v1[i] * v2[i]

**Exemple:**

```normil
let v1 = fill(dim: 64, value: 3.0)
let v2 = fill(dim: 64, value: 2.0)
let prod = vec_mul(v1, v2)
// prod = [6.0, 6.0, ..., 6.0]
```

**Note:**

- ⚠️ Ce n'est PAS le produit scalaire (voir `dot`)

---

### `scale(v: Vec, scalar: float) -> Vec`

Multiplie un vecteur par un scalaire.

**Paramètres:**

- `v`: Vecteur à multiplier
- `scalar`: Facteur de multiplication

**Retour:**

- Nouveau vecteur où result[i] = v[i] * scalar

**Exemple:**

```normil
let v = ones(dim: 64)
let doubled = scale(v, 2.0)
// doubled = [2.0, 2.0, ..., 2.0]
// norm(doubled) ≈ 16.0
```

---

## Opérations sur Chaînes

✨ **Nouvelle fonctionnalité Phase 3.3**

### Opérateur de Concaténation `+`

Concatène des chaînes de caractères avec l'opérateur `+`.

**Syntaxe:**

```normil
let result = string1 + string2
```

**Comportement:**

- Si au moins un opérande est `str`, l'autre est automatiquement converti
- Fonctionne avec int, float, bool convertis en string

**Exemples:**

```normil
let greeting = "Hello" + " " + "World"
// greeting = "Hello World"

let age = 25
let message = "J'ai " + to_string(age) + " ans"
// message = "J'ai 25 ans"

let score = 98.5
let text = "Score: " + to_string(score)
// text = "Score: 98.5"
```

---

### `to_string(value: any) -> str`

Convertit une valeur en chaîne de caractères.

**Paramètres:**

- `value`: Valeur à convertir (int, float, bool, str)

**Retour:**

- Représentation en chaîne de la valeur

**Exemples:**

```normil
let n = 42
let s = to_string(n)  // "42"

let pi = 3.14159
let s2 = to_string(pi)  // "3.14159"

let flag = true
let s3 = to_string(flag)  // "true"
```

**Note:** Alias de `str()` pour éviter confusion avec le type `str`.

---

### `string_length(s: str) -> int`

Retourne la longueur d'une chaîne.

**Paramètres:**

- `s`: Chaîne de caractères

**Retour:**

- Nombre de caractères dans la chaîne

**Exemple:**

```normil
let text = "NORMiL"
let len = string_length(text)  // 6
```

---

### `string_upper(s: str) -> str`

Convertit une chaîne en majuscules.

**Paramètres:**

- `s`: Chaîne à convertir

**Retour:**

- Nouvelle chaîne en majuscules

**Exemple:**

```normil
let text = "hello"
let upper = string_upper(text)  // "HELLO"
```

---

### `string_lower(s: str) -> str`

Convertit une chaîne en minuscules.

**Paramètres:**

- `s`: Chaîne à convertir

**Retour:**

- Nouvelle chaîne en minuscules

**Exemple:**

```normil
let text = "WORLD"
let lower = string_lower(text)  // "world"
```

---

### `string_substring(s: str, start: int, end: int) -> str`

Extrait une sous-chaîne.

**Paramètres:**

- `s`: Chaîne source
- `start`: Index de départ (inclusif)
- `end`: Index de fin (exclusif)

**Retour:**

- Sous-chaîne de `start` à `end-1`

**Exemple:**

```normil
let text = "Hello World"
let sub = string_substring(text, 0, 5)  // "Hello"
let sub2 = string_substring(text, 6, 11)  // "World"
```

---

### `string_split(s: str, separator: str) -> str`

Découpe une chaîne selon un séparateur.

**Paramètres:**

- `s`: Chaîne à découper
- `separator`: Séparateur

**Retour:**

- Premier élément de la division (limitation temporaire)

**Exemple:**

```normil
let text = "a,b,c"
let first = string_split(text, ",")  // "a"
```

**Note:** Version future retournera une liste.

---

### `string_join(items: str, separator: str) -> str`

Joint des éléments avec un séparateur.

**Paramètres:**

- `items`: Chaîne à joindre (temporairement simple string)
- `separator`: Séparateur

**Retour:**

- Items joints par le séparateur

**Exemple:**

```normil
let parts = "Hello World NORMiL"
let joined = string_join(parts, " ")  // "Hello World NORMiL"
```

---

### `string_replace(s: str, old: str, new: str) -> str`

Remplace toutes les occurrences d'une sous-chaîne.

**Paramètres:**

- `s`: Chaîne source
- `old`: Sous-chaîne à remplacer
- `new`: Chaîne de remplacement

**Retour:**

- Nouvelle chaîne avec remplacements effectués

**Exemple:**

```normil
let text = "Hello World"
let replaced = string_replace(text, "World", "NORMiL")
// replaced = "Hello NORMiL"
```

---

### `string_contains(s: str, substring: str) -> bool`

Vérifie si une chaîne contient une sous-chaîne.

**Paramètres:**

- `s`: Chaîne à vérifier
- `substring`: Sous-chaîne recherchée

**Retour:**

- `true` si trouvée, `false` sinon

**Exemple:**

```normil
let text = "Hello World"
let has_world = string_contains(text, "World")  // true
let has_foo = string_contains(text, "foo")  // false
```

---

### `string_startswith(s: str, prefix: str) -> bool`

Vérifie si une chaîne commence par un préfixe.

**Paramètres:**

- `s`: Chaîne à vérifier
- `prefix`: Préfixe recherché

**Retour:**

- `true` si commence par le préfixe, `false` sinon

**Exemple:**

```normil
let text = "Hello World"
let starts = string_startswith(text, "Hello")  // true
let not_starts = string_startswith(text, "World")  // false
```

---

### `string_endswith(s: str, suffix: str) -> bool`

Vérifie si une chaîne finit par un suffixe.

**Paramètres:**

- `s`: Chaîne à vérifier
- `suffix`: Suffixe recherché

**Retour:**

- `true` si finit par le suffixe, `false` sinon

**Exemple:**

```normil
let text = "Hello World"
let ends = string_endswith(text, "World")  // true
let not_ends = string_endswith(text, "Hello")  // false
```

---

### `string_trim(s: str) -> str`

Enlève les espaces en début et fin de chaîne.

**Paramètres:**

- `s`: Chaîne à nettoyer

**Retour:**

- Chaîne sans espaces de début/fin

**Exemple:**

```normil
let text = "  Hello World  "
let trimmed = string_trim(text)  // "Hello World"
```

---

### `string_repeat(s: str, n: int) -> str`

Répète une chaîne n fois.

**Paramètres:**

- `s`: Chaîne à répéter
- `n`: Nombre de répétitions

**Retour:**

- Chaîne répétée n fois

**Exemple:**

```normil
let pattern = "Ha"
let repeated = string_repeat(pattern, 3)  // "HaHaHa"
```

---

### `string_char_at(s: str, index: int) -> str`

Retourne le caractère à un index donné.

**Paramètres:**

- `s`: Chaîne source
- `index`: Position du caractère (0-indexed)

**Retour:**

- Caractère à l'index (string d'un caractère)

**Exemple:**

```normil
let text = "NORMiL"
let char = string_char_at(text, 0)  // "N"
let char2 = string_char_at(text, 4)  // "i"
```

**Erreurs:**

- Index hors limites retourne chaîne vide

---

### `string_index_of(s: str, substring: str) -> int`

Trouve la position d'une sous-chaîne.

**Paramètres:**

- `s`: Chaîne à rechercher
- `substring`: Sous-chaîne recherchée

**Retour:**

- Index de la première occurrence, ou -1 si non trouvé

**Exemple:**

```normil
let text = "Hello World"
let pos = string_index_of(text, "World")  // 6
let not_found = string_index_of(text, "foo")  // -1
```

---

## Métriques

### `dot(v1: Vec, v2: Vec) -> float`

Calcule le produit scalaire (dot product).

**Paramètres:**

- `v1`: Premier vecteur
- `v2`: Second vecteur

**Retour:**

- Somme de v1[i] * v2[i] pour tout i

**Exemple:**

```normil
let v1 = fill(dim: 64, value: 2.0)
let v2 = fill(dim: 64, value: 3.0)
let d = dot(v1, v2)
// d = 2.0 * 3.0 * 64 = 384.0
```

**Applications:**

- Similarité entre vecteurs
- Projections
- Calcul d'activation dans les réseaux neuronaux

---

### `norm(v: Vec) -> float`

Calcule la norme L2 (euclidienne) d'un vecteur.

**Paramètres:**

- `v`: Vecteur

**Retour:**

- sqrt(sum(v[i]²)) pour tout i

**Exemple:**

```normil
let v = ones(dim: 64)
let n = norm(v)
// n = sqrt(64) = 8.0
```

**Formule:**

```
norm(v) = √(v₁² + v₂² + ... + vₙ²)
```

---

### `normalize(v: Vec) -> Vec`

Normalise un vecteur à la norme unitaire (norme = 1.0).

**Paramètres:**

- `v`: Vecteur à normaliser

**Retour:**

- Vecteur dans la même direction avec norm = 1.0

**Exemple:**

```normil
let v = random(dim: 128, mean: 0.0, std: 1.0)
let v_norm = normalize(v)
// norm(v_norm) ≈ 1.0
```

**Formule:**

```
normalize(v) = v / norm(v)
```

**Erreurs:**

- Éviter de normaliser un vecteur de norme ~0 (division par zéro)

---

## Annotations

### `@plastic`

Annotation pour la plasticité neuronale.

**Syntaxe:**

```normil
@plastic(rate: float, mode: str)
fn fonction(...) -> ... {
    // corps
}
```

**Paramètres:**

- `rate`: Taux d'apprentissage (default: 0.001)

  - Valeurs typiques: 0.0001 à 0.1
  - Plus élevé = apprentissage plus rapide mais moins stable
- `mode`: Mode de plasticité (default: "hebbian")

  - `"hebbian"`: Apprentissage Hebbien classique
  - `"anti_hebbian"`: Désapprentissage/oubli
  - `"stdp"`: Spike-Timing Dependent Plasticity
  - `"competitive"`: Apprentissage compétitif
  - `"backprop"`: Rétropropagation (backpropagation)

**Exemple complet:**

```normil
@plastic(rate: 0.01, mode: "hebbian")
fn learn(weights: Vec, input: Vec) -> Vec {
    let delta = scale(vec_mul(weights, input), 0.01)
    return vec_add(weights, delta)
}
```

**Métadonnées accessibles:**

```normil
// Dans la fonction annotée, les métadonnées sont stockées
// dans executor.function_metadata[function_name]
// {
//     'plastic': True,
//     'rate': 0.01,
//     'mode': 'hebbian'
// }
```

**Modes détaillés:**

#### Hebbian

```normil
@plastic(rate: 0.01, mode: "hebbian")
fn hebbian_learning(w: Vec, x: Vec) -> Vec {
    // Δw = rate * w ⊙ x
    // "Neurons that fire together, wire together"
    let delta = scale(vec_mul(w, x), 0.01)
    return vec_add(w, delta)
}
```

#### Anti-Hebbian

```normil
@plastic(rate: 0.005, mode: "anti_hebbian")
fn forgetting(w: Vec, x: Vec) -> Vec {
    // Δw = -rate * w ⊙ x
    // Oubli progressif
    let delta = scale(vec_mul(w, x), 0.005)
    return vec_sub(w, delta)
}
```

#### STDP

```normil
@plastic(rate: 0.002, mode: "stdp")
fn stdp_update(w: Vec, pre: Vec, post: Vec) -> Vec {
    // Timing-dependent plasticity
    // Δw dépend du timing relatif
    let timing = scale(post, 0.8)  // Simule timing
    let delta = scale(vec_mul(w, timing), 0.002)
    return vec_add(w, delta)
}
```

---

### `@atomic`

Annotation pour les transactions avec rollback automatique.

**Syntaxe:**

```normil
@atomic
fn fonction(...) -> ... {
    // corps avec garanties ACID-like
}
```

**Garanties:**

- ✅ **Atomicité**: Tout ou rien
- ✅ **Isolation**: Changements temporaires isolés
- ✅ **Rollback**: État restauré sur erreur
- ✅ **Commit**: Changements appliqués sur succès

**Exemple:**

```normil
@atomic
fn safe_update(value: int) -> int {
    let temp = value + 10
    // Si erreur ici, rollback à l'état initial
    let result = temp * 2
    return result  // Commit implicite
}
```

**Comportement:**

1. **Avant l'exécution:**

   - Snapshot de toutes les variables locales
   - État sauvegardé avec `deepcopy`
2. **Pendant l'exécution:**

   - Modifications dans un scope isolé
   - Pas d'effet sur les variables externes
3. **Si succès:**

   - Commit implicite
   - Changements appliqués
4. **Si erreur:**

   - Rollback automatique
   - État restauré au snapshot
   - Exception propagée avec contexte

**Exemple avec rollback:**

```normil
@atomic
fn risky_operation(v: Vec) -> Vec {
    let n = norm(v)
  
    if n < 0.001 {
        // Division par zéro évitée
        // Rollback automatique
        return zeros(dim: 64)
    }
  
    return normalize(v)  // Commit si ok
}
```

---

### Combinaison d'Annotations

**Ordre recommandé:** `@atomic` puis `@plastic`

```normil
@atomic
@plastic(rate: 0.005, mode: "hebbian")
fn safe_learning(w: Vec, x: Vec) -> Vec {
    let delta = scale(vec_mul(w, x), 0.005)
    let new_w = vec_add(w, delta)
  
    let n = norm(new_w)
  
    if n > 5.0 {
        // Trop instable - rollback
        return w
    }
  
    return normalize(new_w)  // Commit si stable
}
```

**Avantages:**

- Apprentissage avec vérification de stabilité
- Protection contre les explosions de gradient
- Garantie de cohérence

---

## Système de Modules

✨ **Nouvelle fonctionnalité Phase 3.2**

NORMiL supporte maintenant un système de modules pour organiser et réutiliser le code.

### Import de Modules

**Syntaxe de base:**

```normil
import module_name
```

**Import avec alias:**

```normil
import module_name as alias
```

**Utilisation:**

```normil
import math

fn main() {
    let result = math.abs(-42.0)
    print(result)  // 42.0
}
```

---

### Organisation des Modules

**Structure:**

```
project/
├── main.nor           # Votre programme principal
└── modules/           # Dossier des modules
    ├── math.nor       # Module mathématiques
    ├── vectors.nor    # Module vecteurs avancés
    └── custom.nor     # Vos modules personnalisés
```

**Chemin de recherche:**

1. Dossier `modules/` relatif au fichier courant
2. Dossier `modules/` dans le répertoire de travail

---

### Création d'un Module

**Fichier:** `modules/math.nor`

```normil
fn abs(x: float) -> float {
    if x < 0.0 {
        return -x
    } else {
        return x
    }
}

fn max(a: float, b: float) -> float {
    if a > b {
        return a
    } else {
        return b
    }
}

fn min(a: float, b: float) -> float {
    if a < b {
        return a
    } else {
        return b
    }
}

fn clamp(value: float, min_val: float, max_val: float) -> float {
    if value < min_val {
        return min_val
    } else if value > max_val {
        return max_val
    } else {
        return value
    }
}
```

---

### Utilisation des Modules

**Import simple:**

```normil
import math

fn main() {
    let x = math.abs(-10.0)
    let y = math.max(5.0, 10.0)
    let z = math.clamp(15.0, 0.0, 10.0)
  
    print(x)  // 10.0
    print(y)  // 10.0
    print(z)  // 10.0
}
```

**Import avec alias:**

```normil
import vectors as vec

fn main() {
    let v1 = zeros(dim: 64)
    let v2 = ones(dim: 64)
  
    let normalized = vec.create_normalized(v2)
    let sim = vec.compute_similarity(v1, v2)
  
    print(sim)
}
```

**Imports multiples:**

```normil
import math
import vectors as vec

fn main() {
    // Utilisation de plusieurs modules
    let x = math.abs(-5.0)
  
    let v = fill(dim: 64, value: x)
    let norm_v = vec.create_normalized(v)
  
    print(norm(norm_v))  // ~1.0
}
```

---

### Modules Pré-définis

#### Module `math`

**Fonctions disponibles:**

- `abs(x: float) -> float` - Valeur absolue
- `max(a: float, b: float) -> float` - Maximum de deux valeurs
- `min(a: float, b: float) -> float` - Minimum de deux valeurs
- `clamp(value: float, min_val: float, max_val: float) -> float` - Limite entre min et max

**Exemple:**

```normil
import math

fn main() {
    print(math.abs(-42.0))              // 42.0
    print(math.max(10.0, 25.0))         // 25.0
    print(math.min(10.0, 25.0))         // 10.0
    print(math.clamp(15.0, 0.0, 10.0))  // 10.0
}
```

---

#### Module `vectors`

**Fonctions disponibles:**

- `create_normalized(v: Vec) -> Vec` - Crée un vecteur normalisé
- `compute_similarity(v1: Vec, v2: Vec) -> float` - Similarité cosinus
- `weighted_sum(v1: Vec, w1: float, v2: Vec, w2: float) -> Vec` - Somme pondérée
- `distance(v1: Vec, v2: Vec) -> float` - Distance euclidienne

**Exemple:**

```normil
import vectors as vec

fn main() {
    let v1 = random(dim: 64, mean: 1.0, std: 0.2)
    let v2 = random(dim: 64, mean: 0.5, std: 0.1)
  
    // Normalisation
    let n1 = vec.create_normalized(v1)
    let n2 = vec.create_normalized(v2)
  
    // Similarité
    let sim = vec.compute_similarity(n1, n2)
    print(sim)
  
    // Somme pondérée
    let combined = vec.weighted_sum(
        v1: n1, w1: 0.7,
        v2: n2, w2: 0.3
    )
    print(norm(combined))
  
    // Distance
    let dist = vec.distance(v1, v2)
    print(dist)
}
```

---

## Interopérabilité Python

🚀 **Nouvelle fonctionnalité Phase 4.1 & 4.2**

NORMiL peut importer et utiliser **n'importe quel module Python**, vous donnant accès à tout l'écosystème Python : NumPy, SciPy, pandas, scikit-learn, et bien plus.

### Import de Modules Python

**Syntaxe (identique aux modules NORMiL):**

```normil
import python_module
import python_module as alias
```

**Détection automatique:**

- NORMiL cherche d'abord un fichier `.nor` dans le dossier `modules/`
- Si non trouvé, essaie d'importer comme module Python
- Gestion d'erreur si le module n'existe ni en NORMiL ni en Python

**Exemple:**

```normil
import math        // Module Python (math.py n'existe pas dans modules/)
import mathutils   // Module NORMiL (mathutils.nor existe)
```

---

### Modules Python Standards

#### Module `math`

**Constantes:**

- `math.pi` - π (3.141592...)
- `math.e` - e (2.718281...)
- `math.tau` - τ = 2π
- `math.inf` - Infini
- `math.nan` - Not a Number

**Fonctions de base:**

```normil
import math

fn main() {
    // Racine carrée
    let sqrt_val = math.sqrt(16.0)  // 4.0
  
    // Puissance
    let pow_val = math.pow(2.0, 10.0)  // 1024.0
  
    // Arrondi
    let ceil_val = math.ceil(3.2)    // 4
    let floor_val = math.floor(3.8)  // 3
  
    // Valeur absolue
    let abs_val = math.fabs(-5.5)    // 5.5
}
```

**Fonctions trigonométriques:**

```normil
import math

fn main() {
    let angle = math.pi / 4.0  // 45 degrés
  
    let sin_val = math.sin(angle)      // 0.7071...
    let cos_val = math.cos(angle)      // 0.7071...
    let tan_val = math.tan(angle)      // 1.0
  
    // Inverses
    let asin_val = math.asin(0.5)      // π/6
    let acos_val = math.acos(0.5)      // π/3
    let atan_val = math.atan(1.0)      // π/4
}
```

**Fonctions exponentielles et logarithmiques:**

```normil
import math

fn main() {
    // Exponentielle
    let exp_val = math.exp(1.0)        // e ≈ 2.718
  
    // Logarithmes
    let log_val = math.log(math.e)     // 1.0 (ln)
    let log10_val = math.log10(100.0)  // 2.0
    let log2_val = math.log2(8.0)      // 3.0
}
```

---

#### Module `random`

**Génération aléatoire:**

```normil
import random

fn main() {
    // Fixer la seed pour reproductibilité
    random.seed(42)
  
    // Nombre aléatoire [0.0, 1.0)
    let rand_float = random.random()
  
    // Entier aléatoire [a, b] (inclusif)
    let rand_int = random.randint(1, 100)
  
    // Flottant aléatoire [a, b)
    let rand_uniform = random.uniform(0.0, 10.0)
}
```

**Distribution normale:**

```normil
import random

fn main() {
    random.seed(123)
  
    // Gaussienne standard (μ=0, σ=1)
    let gauss = random.gauss(0.0, 1.0)
  
    // Distribution normale
    let normal = random.normalvariate(5.0, 2.0)
}
```

---

#### Module `json`

**Sérialisation:**

```normil
import json

fn main() {
    // Convertir en JSON
    let json_str = json.dumps("hello")     // "\"hello\""
    let json_num = json.dumps(42)          // "42"
    let json_bool = json.dumps(true)       // "true"
  
    print(json_str)
    print(json_num)
    print(json_bool)
}
```

---

### Appels de Fonctions Python

**Arguments multiples:**

```normil
import math

fn main() {
    // Fonction à 2 arguments
    let pow_result = math.pow(2.0, 10.0)  // 1024.0
  
    // Fonction à 3 arguments
    let atan2_result = math.atan2(1.0, 1.0)  // π/4
}
```

**Appels imbriqués:**

```normil
import math

fn main() {
    // Les appels Python peuvent être imbriqués
    let hypotenuse = math.sqrt(
        math.pow(3.0, 2.0) + math.pow(4.0, 2.0)
    )  // 5.0
  
    // Dans des expressions complexes
    let aire_cercle = math.pi * math.pow(5.0, 2.0)  // 78.539...
}
```

**Chaîne d'appels:**

```normil
import math

fn main() {
    let a = math.sqrt(16.0)      // 4.0
    let b = math.pow(a, 2.0)     // 16.0
    let c = math.floor(b)        // 16
  
    print(c)
}
```

**Calculs mixtes NORMiL/Python:**

```normil
import math

fn main() {
    // Variables NORMiL
    let rayon = 5.0
  
    // Calculs avec fonctions Python
    let aire = math.pi * rayon * rayon
    let perimetre = 2.0 * math.pi * rayon
    let volume = (4.0 / 3.0) * math.pi * math.pow(rayon, 3.0)
  
    print(aire)       // 78.539...
    print(perimetre)  // 31.415...
    print(volume)     // 523.598...
}
```

---

### Gestion des Types

**Conversions automatiques:**

```normil
import math

fn main() {
    // int → float automatique
    let result1 = math.sqrt(16)     // 4.0 (16 converti en 16.0)
  
    // bool converti en int/float selon contexte
    let result2 = math.pow(2.0, 3)  // 8.0 (3 utilisé comme 3.0)
}
```

**Types supportés:**

- `int` NORMiL → `int` Python
- `float` NORMiL → `float` Python
- `str` NORMiL → `str` Python
- `bool` NORMiL → `bool` Python
- `None` Python → `None` en NORMiL

**Retours de fonctions:**

```normil
import random

fn main() {
    // Fonction retournant None
    let result = random.seed(42)  // None
  
    // Fonction retournant float
    let value = random.random()   // float
  
    // Fonction retournant int
    let dice = random.randint(1, 6)  // int
}
```

---

### Gestion des Exceptions

**Exceptions Python propagées:**

```normil
import math

fn main() {
    // Ceci lève une exception Python (sqrt de négatif)
    // let error = math.sqrt(-1.0)  // ValueError!
  
    // Utilisez des valeurs valides
    let valid = math.sqrt(1.0)  // 1.0
}
```

**Erreurs d'import:**

```normil
// Module inexistant
// import nonexistent_module  // ImportError!

// Attribut inexistant
import math
// let x = math.nonexistent_function()  // AttributeError!
```

---

### Exemples Pratiques

**Calcul scientifique:**

```normil
import math

fn distance_euclidienne(x1: float, y1: float, x2: float, y2: float) -> float {
    let dx = x2 - x1
    let dy = y2 - y1
    return math.sqrt(dx * dx + dy * dy)
}

fn aire_triangle(a: float, b: float, c: float) -> float {
    // Formule de Héron
    let s = (a + b + c) / 2.0
    let area_sq = s * (s - a) * (s - b) * (s - c)
    return math.sqrt(area_sq)
}

fn main() {
    let dist = distance_euclidienne(0.0, 0.0, 3.0, 4.0)
    print(dist)  // 5.0
  
    let area = aire_triangle(3.0, 4.0, 5.0)
    print(area)  // 6.0
}
```

**Simulation Monte Carlo:**

```normil
import random
import math

fn estimer_pi(iterations: int) -> float {
    random.seed(42)
    let inside = 0
  
    for i in range(0, iterations) {
        let x = random.random()
        let y = random.random()
        let distance = math.sqrt(x * x + y * y)
      
        if distance <= 1.0 {
            inside = inside + 1
        }
    }
  
    let pi_estimate = 4.0 * inside / iterations
    return pi_estimate
}

fn main() {
    let pi_approx = estimer_pi(10000)
    print(pi_approx)  // ~3.14...
}
```

**Conversions d'angles:**

```normil
import math

fn degres_vers_radians(degres: float) -> float {
    return degres * math.pi / 180.0
}

fn radians_vers_degres(radians: float) -> float {
    return radians * 180.0 / math.pi
}

fn main() {
    let rad_90 = degres_vers_radians(90.0)
    print(rad_90)  // π/2 ≈ 1.5707...
  
    let deg_pi = radians_vers_degres(math.pi)
    print(deg_pi)  // 180.0
}
```

---

### Limites Actuelles

**Pas encore supporté:**

- ❌ Arguments nommés Python (kwargs) : `func(x=1, y=2)`
- ❌ Conversion automatique Vec ↔ numpy.ndarray
- ❌ Certains types Python complexes

**Fonctionnel (Phases 4.1-4.4 complètes):**

- ✅ Import de modules Python (Phase 4.1)
- ✅ Accès aux constantes (Phase 4.1)
- ✅ Appels de fonctions (Phase 4.2)
- ✅ Types primitifs (int, float, str, bool, None) (Phase 4.3)
- ✅ Conversions automatiques (Phase 4.3)
- ✅ Listes et tuples Python (Phase 4.3)
- ✅ Instantiation de classes Python (Phase 4.4)
- ✅ Méthodes d'objets (Phase 4.4)
- ✅ Attributs d'objets (Phase 4.4)
- ✅ Chaînage de méthodes (Phase 4.4)
- ✅ Méthodes sur types natifs (str, list) (Phase 4.4)
- ✅ Exceptions Python

---

### Objets et Classes Python (Phase 4.4)

**Méthodes sur les types natifs:**

Les chaînes, listes et autres types Python supportent leurs méthodes natives :

```normil
fn manipuler_strings() {
    let texte = "hello world"
  
    // Méthodes de transformation
    let upper = texte.upper()        // "HELLO WORLD"
    let lower = upper.lower()        // "hello world"
    let replaced = texte.replace("world", "NORMiL")  // "hello NORMiL"
  
    // Méthodes de parsing
    let mots = texte.split(" ")      // ["hello", "world"]
  
    // Méthodes de test
    let starts = texte.startswith("hello")  // true
    let ends = texte.endswith("world")      // true
}
```

**Méthodes sur les listes:**

```normil
fn manipuler_listes() {
    let nombres = [1, 2, 3]
  
    // Modification en place
    nombres.append(4)
    nombres.append(5)
    print(nombres)  // [1, 2, 3, 4, 5]
}
```

**Chaînage de méthodes:**

Les méthodes peuvent être chaînées pour des transformations complexes :

```normil
fn chainer() {
    let texte = "  hello world  "
  
    // Chaîner strip() puis upper()
    let resultat = texte.strip().upper()
    print(resultat)  // "HELLO WORLD"
  
    // Chaînage complexe
    let complexe = "  python rocks  "
        .strip()
        .replace("python", "NORMiL")
        .upper()
    print(complexe)  // "NORMIL ROCKS"
}
```

**Instantiation de classes Python:**

```normil
import datetime

fn creer_dates() {
    // Créer un objet datetime
    let noel = datetime.datetime(2024, 12, 25)
    let nouvel_an = datetime.datetime(2024, 1, 1)
  
    print(noel)        // Objet datetime
    print(nouvel_an)   // Objet datetime
}
```

**Accès aux attributs d'objets:**

```normil
import datetime

fn explorer_date() {
    let date = datetime.datetime(2024, 6, 15)
  
    // Accès aux attributs
    let annee = date.year     // 2024
    let mois = date.month     // 6
    let jour = date.day       // 15
  
    print(annee)
    print(mois)
    print(jour)
}
```

**Appel de méthodes sur objets:**

```normil
import datetime

fn utiliser_methodes() {
    let date = datetime.datetime(2024, 12, 25)
  
    // Appeler des méthodes
    let jour_semaine = date.weekday()  // 0=Lundi, 6=Dimanche
    print(jour_semaine)  // 2 (Mercredi)
}
```

**Exemples pratiques:**

Validation d'email avec méthodes :

```normil
fn valider_email(email: str) -> bool {
    let parties = email.split("@")
  
    if parties.length == 2 {
        let user_ok = parties[0].length > 0
        let domain_ok = parties[1].length > 0
        return user_ok && domain_ok
    }
  
    return false
}

fn main() {
    print(valider_email("user@example.com"))  // true
    print(valider_email("@example.com"))      // false
}
```

Parser CSV simple :

```normil
fn parser_csv(ligne: str) -> [str] {
    return ligne.split(",")
}

fn formater_entetes(entetes: [str]) -> [str] {
    let resultat = []
    for entete in entetes {
        let maj = entete.upper()
        resultat.append(maj)
    }
    return resultat
}

fn main() {
    let ligne = "nom,prenom,age"
    let colonnes = parser_csv(ligne)
    let titres = formater_entetes(colonnes)
  
    print(titres)  // ["NOM", "PRENOM", "AGE"]
}
```

Manipulation de dates :

```normil
import datetime

fn analyser_annee(annee: int) {
    let debut = datetime.datetime(annee, 1, 1)
    let milieu = datetime.datetime(annee, 6, 15)
    let fin = datetime.datetime(annee, 12, 31)
  
    print(debut.weekday())
    print(milieu.weekday())
    print(fin.weekday())
}

fn main() {
    analyser_annee(2024)
}
```

---

### Caractéristiques du Système de Modules

**Caching:**

- Modules chargés une seule fois en mémoire
- Performances optimales avec multiples imports

**Scopes isolés:**

- Chaque module a son propre scope
- Pas de pollution de namespace
- Variables de module non accessibles

**Accès aux fonctions:**

- Syntaxe `module.fonction()`
- Ou `alias.fonction()` avec import aliasé

**Gestion des erreurs:**

```normil
import non_existent  // Erreur: Module 'non_existent' not found

import math
let x = math.fonction_inexistante()  // Erreur: function not found
```

---

## Inférence de Types

✨ **Nouvelle fonctionnalité Phase 3.1**

NORMiL peut maintenant déduire automatiquement le type des variables.

### Syntaxe

**Avec annotation explicite (classique):**

```normil
let x: int = 42
let y: float = 3.14
let s: str = "hello"
```

**Avec inférence (nouveau):**

```normil
let x = 42          // Type inféré: int
let y = 3.14        // Type inféré: float
let s = "hello"     // Type inféré: str
```

---

### Types Inférés

**Depuis des literals:**

```normil
let a = 42               // int
let b = 3.14             // float
let c = "NORMiL"         // str
let d = true             // bool
let e = zeros(dim: 64)   // Vec
```

**Depuis des expressions:**

```normil
let sum = 10 + 20        // int (10 et 20 sont int)
let avg = 10.0 / 2.0     // float (opération sur floats)
let msg = "Hello"        // str
let flag = 5 > 3         // bool (résultat de comparaison)
```

**Depuis des retours de fonction:**

```normil
fn get_number() -> int {
    return 42
}

fn main() {
    let n = get_number()  // Type inféré: int
    print(n)
}
```

**Avec des vecteurs:**

```normil
let v1 = zeros(dim: 64)           // Vec
let v2 = random(dim: 128, mean: 0.0, std: 1.0)  // Vec
let v3 = vec_add(v1, v2)          // Type invalide (dimensions différentes)
```

---

### Ordre de Déduction

L'inférence suit cet ordre de priorité:

1. **bool** - Si la valeur est `true` ou `false`
2. **int** - Si la valeur est un entier sans point décimal
3. **float** - Si la valeur est un nombre avec point décimal
4. **str** - Si la valeur est entre guillemets
5. **Vec** - Si la valeur est un vecteur NumPy

**Exemples:**

```normil
let a = true         // bool (priorité 1)
let b = 42           // int (priorité 2)
let c = 42.0         // float (priorité 3, a un point)
let d = "42"         // str (priorité 4)
let e = zeros(dim: 5) // Vec (priorité 5)
```

---

### Compatibilité

**Avec @plastic:**

```normil
@plastic(rate: 0.01, mode: "hebbian")
fn learn(weights: Vec, input: Vec) -> Vec {
    let delta = vec_mul(weights, input)  // Inféré: Vec
    let scaled = scale(delta, 0.01)      // Inféré: Vec
    return vec_add(weights, scaled)
}
```

**Avec @atomic:**

```normil
@atomic
fn safe_operation(value: int) -> int {
    let temp = value + 10    // Inféré: int
    let result = temp * 2    // Inféré: int
    return result
}
```

**Avec const:**

```normil
const PI = 3.14159          // Inféré: float
const MAX_ITER = 1000       // Inféré: int
const GREETING = "Hello"    // Inféré: str
```

---

### Limites

**Type explicite recommandé dans certains cas:**

1. **Paramètres de fonction** (obligatoire):

```normil
fn add(a: int, b: int) -> int {  // Types requis
    return a + b
}
```

2. **Pour la clarté**:

```normil
// Moins clair
let x = 42

// Plus clair pour les grands projets
let max_iterations: int = 42
```

3. **Conversion de type intentionnelle**:

```normil
let x: float = 42  // Force int vers float
// vs
let y = 42         // Inféré comme int
```

---

## Pattern Matching

### Syntaxe Générale

```normil
match expression {
    case pattern1 -> { block1 }
    case pattern2 -> { block2 }
    case _ -> { default_block }
}
```

### Types de Patterns

#### 1. Literal Pattern

```normil
match x {
    case 0 -> { return "zero" }
    case 1 -> { return "un" }
    case 42 -> { return "reponse" }
}
```

#### 2. Wildcard Pattern

```normil
match x {
    case _ -> { return "n'importe quoi" }
}
```

#### 3. Type Extraction Pattern

```normil
match value {
    case int(x) -> { /* x est extrait */ }
    case float(f) -> { /* f est extrait */ }
    case str(s) -> { /* s est extrait */ }
    case bool(b) -> { /* b est extrait */ }
}
```

#### 4. Pattern avec Condition (where)

```normil
match value {
    case int(x) where x > 0 -> { return "positif" }
    case int(x) where x < 0 -> { return "negatif" }
    case int(x) where x == 0 -> { return "zero" }
}
```

### Exemples Complets

#### Classification par Plages

```normil
fn classifier(score: float) -> str {
    match score {
        case float(s) where s >= 0.9 -> { return "A" }
        case float(s) where s >= 0.8 -> { return "B" }
        case float(s) where s >= 0.7 -> { return "C" }
        case float(s) where s >= 0.6 -> { return "D" }
        case _ -> { return "F" }
    }
}
```

#### Validation avec Patterns

```normil
fn valider_entree(n: int) -> bool {
    match n {
        case int(x) where x >= 1 and x <= 100 -> {
            return true
        }
        case _ -> {
            return false
        }
    }
}
```

#### Patterns sur Strings

```normil
fn detecter_langue(mot: str) -> str {
    match mot {
        case "bonjour" -> { return "francais" }
        case "hello" -> { return "anglais" }
        case "hola" -> { return "espagnol" }
        case _ -> { return "inconnue" }
    }
}
```

---

## Fonctions Built-in

### `print(value: any)`

Affiche une valeur sur la sortie standard.

**Paramètres:**

- `value`: Valeur à afficher (int, float, str, bool, Vec)

**Comportement:**

- Int/Float/Bool: Affiche la valeur
- String: Affiche sans quotes
- Vec: Affiche un résumé (premiers/derniers éléments si > 10)

**Exemple:**

```normil
print("Hello")        // Hello
print(42)             // 42
print(3.14)           // 3.14
print(true)           // true

let v = ones(dim: 64)
print(v)              // [1.0, 1.0, 1.0, ...]
```

---

### `range(start: int, end: int) -> iterable`

Génère une séquence d'entiers.

**Paramètres:**

- `start`: Valeur de départ (incluse)
- `end`: Valeur de fin (exclue)

**Retour:**

- Itérable pour boucles for

**Exemple:**

```normil
for i in range(0, 5) {
    print(i)  // 0, 1, 2, 3, 4
}

for i in range(10, 15) {
    print(i)  // 10, 11, 12, 13, 14
}
```

---

## Types

### Types Primitifs

| Type      | Description | Valeurs            | Exemple                         |
| --------- | ----------- | ------------------ | ------------------------------- |
| `int`   | Entier      | ..., -1, 0, 1, ... | `let x: int = 42`             |
| `float` | Flottant    | 3.14, -0.5, 1e-3   | `let y: float = 3.14`         |
| `str`   | Chaîne     | "hello", "world"   | `let s: str = "hi"`           |
| `bool`  | Booléen    | true, false        | `let b: bool = true`          |
| `Vec`   | Vecteur     | NumPy array        | `let v: Vec = zeros(dim: 64)` |

### Type Vec

**Représentation interne:**

- NumPy array (np.ndarray)
- dtype: float64
- 1-dimension

**Propriétés:**

- Dimension fixe à la création
- Opérations vectorisées (rapides)
- Compatible avec toutes les primitives vectorielles

**Exemple d'utilisation:**

```normil
let v1: Vec = random(dim: 128, mean: 0.0, std: 1.0)
let v2: Vec = ones(dim: 128)
let v3: Vec = vec_add(v1, v2)

let n: float = norm(v3)
let d: float = dot(v1, v2)
```

---

### Types O-RedMind (✨ Phase 5)

NORMiL fournit des types spécialisés pour l'architecture O-RedMind :

#### EpisodicRecord - Mémoire Épisodique

Enregistrement d'événement brut horodaté avec vecteurs multimodaux.

**Structure:**

```normil
EpisodicRecord {
    id: str,                    // Identifiant unique
    timestamp: float,           // Horodatage (Unix timestamp)
    sources: [str],             // Sources d'information
    vecs: {str: Vec},          // Vecteurs multimodaux (dict)
    summary: str,               // Résumé textuel
    labels: [Label],            // Labels avec scores
    trust: float,               // Score de confiance (0.0-1.0)
    provenance: Provenance,     // Traçabilité
    outcome: str                // Résultat/statut
}
```

**Exemple:**

```normil
let event = EpisodicRecord {
    id: "evt_001",
    timestamp: 1698000000.0,
    sources: ["camera", "audio"],
    vecs: {},
    summary: "User interaction",
    labels: [],
    trust: 0.95,
    provenance: {},
    outcome: "success"
}
```

#### Concept - Mémoire Sémantique

Concept compressé avec confiance pour la knowledge base.

**Structure:**

```normil
Concept {
    concept_id: str,              // Identifiant unique
    centroid_vec: Vec,            // Vecteur centroïde
    doc_count: int,               // Nombre de documents
    provenance_versions: [str],   // Versions de provenance
    trust_score: float,           // Score de confiance
    labels: [str]                 // Labels textuels
}
```

**Exemple:**

```normil
let ai_concept = Concept {
    concept_id: "ai_ml_001",
    centroid_vec: vec(128, [1.0, 0.5, -0.3, 0.8]),
    doc_count: 42,
    provenance_versions: ["v1.0"],
    trust_score: 0.85,
    labels: ["AI", "ML"]
}
```

#### ProtoInstinct - Comportement Instinctif

Prototype d'instinct avec vecteur de référence et règle.

**Structure:**

```normil
ProtoInstinct {
    id: str,           // Identifiant unique
    vec_ref: Vec,      // Vecteur de référence
    rule: str,         // Règle symbolique (optionnel)
    weight: float      // Poids/priorité
}
```

**Exemple:**

```normil
let safety = ProtoInstinct {
    id: "privacy_guard",
    vec_ref: vec(64, [0.8, 0.9, 0.7, 0.95]),
    rule: "if similarity > 0.9 then activate",
    weight: 1.5
}
```

#### SparseVec - Vecteur Creux Optimisé

Vecteur creux stockant seulement les valeurs non-nulles.

**Structure:**

```normil
SparseVec {
    indices: [int],    // Indices des valeurs non-nulles
    values: [float],   // Valeurs correspondantes
    dim: int           // Dimension totale
}
```

**Exemple:**

```normil
let sparse = SparseVec {
    indices: [0, 100, 500, 999],
    values: [1.5, 2.0, -0.5, 3.0],
    dim: 1000
}

// Sparsité: 99.6% (4 non-zeros sur 1000)
```

**Cas d'usage:**

- NLP: word embeddings
- Réseaux de neurones creux
- Économie mémoire pour grandes dimensions

---

### Primitives Neurales (✨ Phase 6)

Primitives avancées pour l'apprentissage incrémental et l'optimisation mémoire.

#### lowrankupdate()

Mise à jour de rang faible d'une matrice : W' = W + u ⊗ v

**Signature:**

```normil
lowrankupdate(W: [[float]], u: Vec, v: Vec) -> [[float]]
```

**Paramètres:**

- `W` : Matrice à mettre à jour (list de lists)
- `u` : Vecteur gauche (Vec)
- `v` : Vecteur droit (Vec)

**Retourne:** Nouvelle matrice W' = W + u ⊗ v

**Exemple:**

```normil
let W = [[1.0, 0.0], [0.0, 1.0]]  // Matrice identité
let u = vec(2, [1.0, 0.0])
let v = vec(2, [0.0, 1.0])

let W_new = lowrankupdate(W, u, v)
// Résultat: [[1.0, 1.0], [0.0, 1.0]]
```

**Cas d'usage:**

- Adaptation de poids neuronaux sans ré-entraînement
- Apprentissage incrémental efficace
- Mise à jour de modèles avec faible coût

---

#### quantize()

Quantisation d'un vecteur sur n bits (8 ou 4).

**Signature:**

```normil
quantize(vec: Vec, bits: int) -> Vec
```

**Paramètres:**

- `vec` : Vecteur à quantifier
- `bits` : Nombre de bits (8 ou 4)

**Retourne:** Vecteur quantifié (dimension préservée)

**Exemple:**

```normil
let v = random(128, 0.0, 1.0)

// Haute précision (~1% erreur)
let v_q8 = quantize(v, 8)

// Haute compression (~5% erreur)
let v_q4 = quantize(v, 4)
```

**Comparaison:**

| Bits | Précision | Compression | Use Case                 |
| ---- | ---------- | ----------- | ------------------------ |
| 8    | ~1% erreur | 50%         | Production, faible perte |
| 4    | ~5% erreur | 75%         | Stockage, transmission   |

**Cas d'usage:**

- Stockage de vecteurs à grande échelle
- Transmission réseau optimisée
- Systèmes embarqués

---

#### onlinecluster_update()

Mise à jour incrémentale d'un centroïde : c' = (1 - lr) × c + lr × x

**Signature:**

```normil
onlinecluster_update(centroid: Vec, x: Vec, lr: float) -> Vec
```

**Paramètres:**

- `centroid` : Centroïde actuel
- `x` : Nouveau point à intégrer
- `lr` : Learning rate ∈ [0, 1]

**Retourne:** Nouveau centroïde après mise à jour

**Exemple:**

```normil
let c = zeros(64)
let lr = 0.1

// Ajouter progressivement des points
let x1 = random(64, 0.0, 1.0)
c = onlinecluster_update(c, x1, lr)

let x2 = random(64, 0.0, 1.0)
c = onlinecluster_update(c, x2, lr)

// c converge vers la moyenne des points
```

**Learning rate:**

| Valeur | Comportement       | Use Case   |
| ------ | ------------------ | ---------- |
| 0.0    | Aucun changement   | Freeze     |
| 0.1    | Adaptation lente   | Stabilité |
| 0.5    | Adaptation moyenne | Balance    |
| 1.0    | Remplacement       | Reset      |

**Cas d'usage:**

- Consolidation sémantique en temps réel
- Clustering sans stocker tous les points
- Adaptation continue de concepts

---

### Système de Transactions (✨ Phase 6)

Transactions avec audit logging automatique et rollback.

#### Déclaration

**Syntaxe:**

```normil
transaction name(params) -> ReturnType {
    // Corps de la transaction
    // Automatiquement loggé
}
```

**Exemple basique:**

```normil
transaction append_episode_safe(summary: str, trust: float) -> str {
    let v = random(128, 0.0, 1.0)
    let record = EpisodicRecord {
        id: generate_uuid(),
        timestamp: now(),
        sources: ["system"],
        vecs: {"default": v},
        summary: summary,
        labels: [],
        trust: trust,
        provenance: {"device_id": "prod", "signature": ""},
        outcome: "success"
    }
  
    let id = episodic_append(record)
    return id
}

// Appel
let ep_id = append_episode_safe("Important event", 0.95)
```

#### Transaction avec Rollback

**Syntaxe:**

```normil
transaction name(params) {
    // Corps principal
} rollback {
    // Bloc exécuté en cas d'erreur
}
```

**Exemple:**

```normil
transaction update_concept(concept_id: str, new_vec: Vec) {
    let old = semantic_query(concept_id, k: 1)[0]
    semantic_upsert(concept_id, new_vec)
    audit_log("concept_updated", concept_id)
} rollback {
    // Restaurer l'ancien en cas d'erreur
    semantic_upsert(concept_id, old.centroid_vec)
}
```

#### Audit Automatique

Chaque transaction enregistre automatiquement :

**Au début:**

```
transaction_start_<name>: {
    params: {...}
}
```

**En cas de succès:**

```
transaction_success_<name>: {
    result: "...",
    duration_ms: 123
}
```

**En cas d'erreur:**

```
transaction_failed_<name>: {
    error: "..."
}
```

**Avantages:**

- ✅ Traçabilité complète
- ✅ Intégrité garantie
- ✅ Rollback automatique
- ✅ Hash chaining (future)

---

### Plasticité Neuronale Avancée (✨ Phase 7)

La Phase 7 apporte la gestion automatique de la plasticité avec trois nouvelles primitives et l'enrichissement de l'annotation `@plastic`.

#### normalize_plasticity(weights: Vec) -> Vec

Normalise un vecteur à norme L2 = 1.0 pour maintenir une magnitude constante.

**Syntaxe:**

```normil
let normalized = normalize_plasticity(weights)
```

**Paramètres:**

- `weights` : Vecteur à normaliser

**Retour:**

- `Vec` : Vecteur normalisé (norme L2 = 1.0)

**Comportement:**

- Si norme < 1e-4 : Retourne le vecteur inchangé (évite division par zéro)
- Sinon : v' = v / ||v||₂

**Exemple:**

```normil
let w = vec(3, [3.0, 4.0, 0.0])
let w_norm = normalize_plasticity(w)
// w_norm ≈ [0.6, 0.8, 0.0], norme = 1.0

print("Norm: " + to_string(norm(w_norm)))  // 1.0
```

**Cas d'usage:**

- Maintenir la stabilité numérique pendant l'apprentissage
- Éviter l'explosion/extinction de gradient
- Garantir que les poids restent dans une plage stable

---

#### decay_learning_rate(lr: float, factor: float = 0.99) -> float

Applique une décroissance exponentielle au taux d'apprentissage.

**Syntaxe:**

```normil
let new_lr = decay_learning_rate(lr, factor)
```

**Paramètres:**

- `lr` : Taux d'apprentissage actuel
- `factor` : Facteur de decay (0 < factor ≤ 1.0, défaut: 0.99)

**Retour:**

- `float` : Nouveau taux d'apprentissage (lr' = lr × factor)

**Validation:**

- Lève `ValueError` si `factor` ≤ 0 ou `factor` > 1.0

**Exemple:**

```normil
let lr = 0.1

// Decay simple
lr = decay_learning_rate(lr, 0.95)  // lr = 0.095

// Decay progressif
for i in range(10) {
    lr = decay_learning_rate(lr, 0.95)
}
// lr ≈ 0.0599
```

**Cas d'usage:**

- Convergence progressive vers un optimum
- Réduction automatique du pas d'apprentissage
- Amélioration de la stabilité en fin de training

---

#### compute_stability(weights_old: Vec, weights_new: Vec, threshold: float) -> bool

Vérifie si deux vecteurs sont stables (changement relatif < seuil).

**Syntaxe:**

```normil
let is_stable = compute_stability(w_old, w_new, threshold)
```

**Paramètres:**

- `weights_old` : Vecteur avant mise à jour
- `weights_new` : Vecteur après mise à jour
- `threshold` : Seuil de stabilité (ex: 0.01 = 1%)

**Retour:**

- `bool` : `true` si changement < seuil, sinon `false`

**Calcul:**

- relative_change = ||w_new - w_old||₂ / ||w_old||₂
- Retourne `true` si relative_change < threshold
- Retourne aussi `true` si ||w_old||₂ < 1e-4 (poids quasi-nuls)

**Validation:**

- Lève `ValueError` si dimensions différentes

**Exemple:**

```normil
let w1 = vec(3, [1.0, 2.0, 3.0])
let w2 = vec(3, [1.001, 2.002, 3.001])

// Petit changement (~0.1%)
let stable = compute_stability(w1, w2, 0.01)
// stable = true

let w3 = vec(3, [1.5, 3.0, 4.5])
// Grand changement (~50%)
let unstable = compute_stability(w1, w3, 0.01)
// unstable = false
```

**Cas d'usage:**

- Critère d'arrêt pour l'apprentissage
- Détection de convergence
- Optimisation early stopping

---

#### Annotation @plastic Enrichie

L'annotation `@plastic` supporte maintenant le paramètre `stability_threshold` et la gestion automatique.

**Syntaxe:**

```normil
@plastic(rate: float, mode: string, stability_threshold: float)
fn learn_function(...) -> Vec {
    ...
}
```

**Paramètres:**

- `rate` : Taux d'apprentissage initial (sera automatiquement décru)
- `mode` : Mode de plasticité (`"hebbian"`, `"stdp"`, `"anti_hebbian"`)
- `stability_threshold` : Seuil de convergence (défaut: 0.01 = 1%)

**Métadonnées automatiques:**

- `step_count` : Compteur d'appels à la fonction
- `is_stable` : `true` quand la stabilité est atteinte
- `rate` : Taux d'apprentissage actuel (décroît automatiquement)

**Gestion automatique:**

Pour chaque appel à une fonction `@plastic`:

1. **Incrémentation:** `step_count++`
2. **Capture de poids:** Recherche automatique de variables nommées:

   - `weights`, `w`, `synapses`, `connections`
3. **Vérification stabilité:** Si poids capturés ET résultat Vec:

   ```
   is_stable = compute_stability(weights_before, result, stability_threshold)
   ```
4. **Normalisation automatique:** Si mode ∈ {hebbian, stdp, anti_hebbian}:

   ```
   result = normalize_plasticity(result)
   ```
5. **Decay learning rate:** Si non stable ET poids capturés:

   ```
   rate = decay_learning_rate(rate, 0.99)
   ```

**Exemple complet:**

```normil
@plastic(rate: 0.1, mode: "hebbian", stability_threshold: 0.01)
fn adapt_weights(input: Vec) -> Vec {
    let weights = random_vec(input.dim)
  
    // Mise à jour
    weights = onlinecluster_update(weights, input, 0.1)
  
    return weights
    // Automatiquement:
    // - Normalisé (norme = 1.0)
    // - Stabilité vérifiée
    // - LR décru si instable
}

// Utilisation
let data = vec(10, [0.5, 0.3, ...])
let w1 = adapt_weights(data)  // step_count=1, rate=0.1, norm=1.0
let w2 = adapt_weights(data)  // step_count=2, rate≈0.099, norm=1.0
// ... convergence automatique
```

**Modes disponibles:**

| Mode             | Description                   | Normalisation |
| ---------------- | ----------------------------- | ------------- |
| `hebbian`      | Renforcement corrélé (Hebb) | Oui (auto)    |
| `stdp`         | Spike-Timing Dependent        | Oui (auto)    |
| `anti_hebbian` | Décorrélation               | Oui (auto)    |
| Autre            | Mode personnalisé            | Non           |

**Avantages:**

- ✅ Zero boilerplate code
- ✅ Convergence garantie
- ✅ Stabilité numérique assurée
- ✅ Traçabilité complète

**Exemple multi-couches:**

```normil
@plastic(rate: 0.05, mode: "hebbian", stability_threshold: 0.01)
fn layer1(x: Vec) -> Vec {
    let w = zeros(x.dim)
    w = onlinecluster_update(w, x, 0.05)
    return w  // Auto-normalisé
}

@plastic(rate: 0.03, mode: "stdp", stability_threshold: 0.005)
fn layer2(h: Vec) -> Vec {
    let w = zeros(h.dim)
    w = onlinecluster_update(w, h, 0.03)
    return w  // Auto-normalisé
}

fn train(data: Vec) {
    let hidden = layer1(data)    // Norm ≈ 1.0
    let output = layer2(hidden)  // Norm ≈ 1.0
}
```

---

### Opérateurs

### Arithmétiques

| Opérateur | Types           | Exemple                | Résultat         |
| ---------- | --------------- | ---------------------- | ----------------- |
| `+`      | int, float      | `5 + 3`              | `8`             |
| `+`      | str (concat) ✨ | `"Hello" + " World"` | `"Hello World"` |
| `-`      | int, float      | `5 - 3`              | `2`             |
| `*`      | int, float      | `5 * 3`              | `15`            |
| `/`      | int, float      | `6 / 2`              | `3`             |

⚠️ **Note:**

- Pour les vecteurs, utilisez `vec_add`, `vec_mul`, etc.
- L'opérateur `+` supporte maintenant la concaténation de strings (Phase 3.3)

**Concaténation de strings:**

```normil
let greeting = "Hello" + " " + "World"  // "Hello World"
let message = "The answer is " + to_string(42)  // "The answer is 42"
```

### Comparaison

| Opérateur | Description         | Exemple    |
| ---------- | ------------------- | ---------- |
| `==`     | Égalité           | `x == 5` |
| `!=`     | Différence         | `x != 5` |
| `<`      | Inférieur          | `x < 5`  |
| `<=`     | Inférieur ou égal | `x <= 5` |
| `>`      | Supérieur          | `x > 5`  |
| `>=`     | Supérieur ou égal | `x >= 5` |

### Logiques

| Opérateur | Description | Exemple              |
| ---------- | ----------- | -------------------- |
| `and`    | ET logique  | `x > 0 and x < 10` |
| `or`     | OU logique  | `x < 0 or x > 10`  |
| `not`    | NON logique | `not active`       |

---

## Conventions et Best Practices

### Nommage

```normil
// Fonctions: snake_case
fn calculate_norm(v: Vec) -> float { ... }

// Variables: snake_case (avec ou sans annotation de type)
let learning_rate = 0.01                    // Inféré: float
let hidden_layer: Vec = zeros(dim: 256)     // Explicite

// Constantes: UPPER_SNAKE_CASE
const MAX_ITERATIONS = 1000                 // Inféré: int
const DEFAULT_LEARNING_RATE = 0.001         // Inféré: float
```

### Arguments Nommés

```normil
// ✅ Bon: Arguments clairs
let v = random(
    dim: 256,
    mean: 0.0,
    std: 1.0
)

// ❌ Éviter: Ordre implicite
let v = random(256, 0.0, 1.0)
```

### Inférence vs Annotations Explicites

```normil
// ✅ Bon: Inférence pour clarté
let count = 0
let total = 100.0
let name = "Alice"

// ✅ Bon: Annotations pour documentation
let max_epochs: int = 1000
let learning_rate: float = 0.001
let model_name: str = "transformer"

// ✅ Bon: Toujours annoter les paramètres de fonction
fn train(epochs: int, rate: float, data: Vec) -> Vec {
    let current_epoch = 0  // Inférence OK ici
    // ...
}
```

### Imports et Modules

```normil
// ✅ Bon: Imports au début du fichier
import math
import vectors as vec

fn main() {
    // Utilisation claire
    let x = math.abs(-10.0)
    let v = vec.create_normalized(ones(dim: 64))
}

// ✅ Bon: Utiliser des alias courts
import very_long_module_name as vm

// ❌ Éviter: Imports multiples du même module
import math
import math as m  // Redondant
```

### Opérations sur Strings

```normil
// ✅ Bon: Concaténation claire
let full_name = first_name + " " + last_name

// ✅ Bon: Conversion explicite
let message = "Score: " + to_string(score)

// ✅ Bon: Utiliser les primitives appropriées
let text = "  hello  "
let clean = string_trim(text)
let upper = string_upper(clean)
```

### Annotations

```normil
// ✅ Bon: Ordre @atomic puis @plastic
@atomic
@plastic(rate: 0.01, mode: "hebbian")
fn safe_learn(...) { ... }

// ✅ Bon: Spécifier les paramètres
@plastic(rate: 0.005, mode: "stdp")
fn precise_learning(...) { ... }
```

### Pattern Matching

```normil
// ✅ Bon: Cas du plus spécifique au plus général
match value {
    case int(x) where x == 0 -> { ... }
    case int(x) where x > 0 -> { ... }
    case int(x) where x < 0 -> { ... }
    case _ -> { ... }
}

// ✅ Bon: Toujours un cas par défaut
match value {
    case specific -> { ... }
    case _ -> { /* fallback */ }
}
```

---

## Limites Connues et Futures Fonctionnalités

### Phase 3 - Complétée ✅

**✅ Implémenté:**

- **Inférence de types** (Phase 3.1) - `let x = 42` sans type explicite
- **Système d'imports** (Phase 3.2) - Modules réutilisables
- **Opérations sur strings** (Phase 3.3) - Concaténation et 14 primitives
- **Accès indexé** (Phase 3.4) - `v[i]` et `list[i]`
- **Structures de données** (Phase 3.5) - `struct` avec fields

### Phase 4 - Complétée ✅

**✅ Implémenté:**

- **Import modules Python** (Phase 4.1) - `import math`, `import random`, etc.
- **Appels de fonctions Python** (Phase 4.2) - `math.sqrt()`, arguments multiples
- **Conversions automatiques** (Phase 4.3) - Types NORMiL ↔ Python seamless
- **Objets et classes Python** (Phase 4.4) - Instantiation, méthodes, attributs, chaînage

### Vecteurs

- ✅ Dimension fixe (comportement intentionnel)
- ✅ Accès par index - `v[i]` (Phase 3.4)
- ❌ Pas de slicing `v[start:end]` (futur)
- ❌ Pas de concaténation directe

### Strings

- ✅ Concaténation avec `+` (Phase 3.3)
- ✅ Conversion `to_string()` (Phase 3.3)
- ✅ 14 primitives (length, upper, lower, etc.) (Phase 3.3)
- ✅ Méthodes Python natives (upper, lower, split, replace, etc.) (Phase 4.4)
- ❌ Pas d'interpolation (futur)

### Types

- ✅ Inférence de type (Phase 3.1)
- ✅ Types doivent être explicites pour paramètres de fonction
- ✅ Conversions automatiques NORMiL ↔ Python (Phase 4.3)
- ❌ Pas de types génériques
- ❌ Pas de types union

### Imports

- ✅ Système de modules NORMiL (Phase 3.2)
- ✅ Import modules Python (Phase 4.1)
- ✅ Import avec alias (`import math as m`)
- ✅ Caching automatique
- ❌ Pas d'imports conditionnels

### Interopérabilité Python

- ✅ Import de modules Python (Phase 4.1)
- ✅ Accès constantes (`math.pi`, `math.e`)
- ✅ Appels de fonctions (`math.sqrt()`, `random.random()`)
- ✅ Types primitifs (int, float, str, bool, None)
- ✅ Listes et tuples Python
- ✅ Instantiation de classes (`datetime.datetime(2024, 1, 1)`)
- ✅ Méthodes d'objets (`date.weekday()`)
- ✅ Attributs d'objets (`date.year`, `date.month`)
- ✅ Chaînage de méthodes (`text.strip().upper()`)
- ✅ Méthodes natives (str.upper, list.append, etc.)
- ❌ Arguments nommés Python (kwargs) - `func(x=1, y=2)`
- ❌ Conversion automatique Vec ↔ numpy.ndarray

---

## Exemples de Référence

Consultez les fichiers dans `examples/`:

- `hello.nor` - Programme de base
- `type_inference.nor` - Démonstration d'inférence de types ✨
- `imports_test.nor` - Utilisation de modules ✨
- `string_operations.nor` - Toutes les opérations string ✨
- `python_interop.nor` - Intégration Python (modules, fonctions) 🚀
- `python_objects.nor` - Objets Python (classes, méthodes, attributs) 🚀
- `memory_simple.nor` - Opérations vectorielles
- `plastic_simple.nor` - Annotation @plastic
- `atomic_transactions.nor` - Annotation @atomic
- `advanced_patterns.nor` - Pattern matching avancé
- `neural_plasticity.nor` - Simulation complète
- `combined_features.nor` - Toutes les features

**Modules disponibles dans `modules/`:**

- `mathutils.nor` - Fonctions mathématiques (abs, max, min, clamp) ✨
- `vectors.nor` - Opérations vectorielles avancées ✨

**Accès à l'écosystème Python complet:**

- `math` - Fonctions mathématiques
- `random` - Génération aléatoire
- `datetime` - Manipulation de dates et heures
- `json` - Parsing et génération JSON
- `sys` - Informations système
- Et 450,000+ packages PyPI disponibles !

---

**NORMiL API Reference v0.6.0**
Dernière mise à jour : Phase 4 Complete (4.1, 4.2, 4.3, 4.4)
Pour plus d'informations : `TUTORIAL.md`, `PHASE2_FINAL_REPORT.md`
