# Guide de Démarrage Rapide NORMiL v0.4.0


**Date** : Novembre 2025
**Auteur** : Diego Morales Magri

---

Bienvenue dans NORMiL ! Ce guide vous permet de démarrer rapidement avec le langage.

---

## Installation et Premier Programme

### 1. Vérifier l'installation

```bash
python normil_cli.py --version
```

### 2. Créer votre premier programme

**Fichier:** `hello.nor`

```normil
fn main() {
    print("Hello NORMiL!")
}
```

### 3. Exécuter

```bash
python normil_cli.py run hello.nor
```

---

## Les Bases en 5 Minutes

### Variables et Inférence de Types ✨

```normil
fn main() {
    // Avec types explicites
    let x: int = 42
    let y: float = 3.14
    let name: str = "Alice"
  
    // Avec inférence automatique (Phase 3.1)
    let age = 25              // Déduit: int
    let temperature = 36.5    // Déduit: float
    let greeting = "Hello"    // Déduit: str
    let active = true         // Déduit: bool
  
    print(age)
    print(temperature)
}
```

### Strings et Concaténation ✨

```normil
fn main() {
    let first = "Alice"
    let last = "Dupont"
  
    // Concaténation avec + (Phase 3.3)
    let full_name = first + " " + last
    print(full_name)  // "Alice Dupont"
  
    // Conversion
    let age = 25
    let message = "J'ai " + to_string(age) + " ans"
    print(message)
  
    // Primitives
    print(string_length("NORMiL"))     // 6
    print(string_upper("hello"))        // "HELLO"
    print(string_contains("Hello World", "World"))  // true
}
```

### Fonctions

```normil
fn double(x: int) -> int {
    return x * 2
}

fn greet(name: str) -> str {
    return "Hello, " + name + "!"
}

fn main() {
    let result = double(21)
    print(result)  // 42
  
    let msg = greet("Alice")
    print(msg)  // "Hello, Alice!"
}
```

### Conditions et Pattern Matching

```normil
fn classify_age(age: int) -> str {
    match age {
        case int(a) where a < 18 -> { return "mineur" }
        case int(a) where a < 65 -> { return "adulte" }
        case _ -> { return "senior" }
    }
}

fn main() {
    print(classify_age(15))   // "mineur"
    print(classify_age(30))   // "adulte"
    print(classify_age(70))   // "senior"
}
```

### Boucles

```normil
fn main() {
    // For loop
    for i in range(0, 5) {
        print(i)  // 0, 1, 2, 3, 4
    }
  
    // While loop
    let count = 0
    while count < 3 {
        print(count)
        count = count + 1
    }
}
```

---

## Vecteurs en 3 Minutes

### Création de Vecteurs

```normil
fn main() {
    let v1 = zeros(dim: 64)                     // Vecteur de zéros
    let v2 = ones(dim: 64)                      // Vecteur de uns
    let v3 = fill(dim: 64, value: 0.5)          // Rempli avec 0.5
    let v4 = random(dim: 64, mean: 0.0, std: 1.0)  // Aléatoire
}
```

### Opérations Vectorielles

```normil
fn main() {
    let v1 = ones(dim: 64)
    let v2 = fill(dim: 64, value: 2.0)
  
    // Opérations
    let sum = vec_add(v1, v2)      // Addition
    let diff = vec_sub(v2, v1)     // Soustraction
    let prod = vec_mul(v1, v2)     // Multiplication élément par élément
    let scaled = scale(v1, 3.0)    // Multiplication scalaire
  
    // Métriques
    let n = norm(v1)               // Norme L2
    let d = dot(v1, v2)            // Produit scalaire
    let normalized = normalize(v1) // Normalisation
  
    print(n)  // ~8.0
    print(norm(normalized))  // ~1.0
}
```

---

## Système de Modules ✨ (Phase 3.2)

### Utiliser des Modules

```normil
import math
import vectors as vec

fn main() {
    // Module math
    let x = math.abs(-42.0)
    let max_val = math.max(10.0, 25.0)
    print(x)        // 42.0
    print(max_val)  // 25.0
  
    // Module vectors
    let v = random(dim: 64, mean: 1.0, std: 0.2)
    let normalized = vec.create_normalized(v)
    print(norm(normalized))  // ~1.0
}
```

### Créer Vos Propres Modules

**Fichier:** `modules/my_utils.nor`

```normil
fn square(x: float) -> float {
    return x * x
}

fn cube(x: float) -> float {
    return x * x * x
}
```

**Fichier:** `main.nor`

```normil
import my_utils as utils

fn main() {
    print(utils.square(5.0))  // 25.0
    print(utils.cube(3.0))    // 27.0
}
```

---

## Annotations Avancées

### @plastic - Plasticité Neuronale

```normil
@plastic(rate: 0.01, mode: "hebbian")
fn learn(weights: Vec, input: Vec) -> Vec {
    let delta = vec_mul(weights, input)
    let scaled = scale(delta, 0.01)
    return vec_add(weights, scaled)
}

fn main() {
    let w = random(dim: 64, mean: 0.0, std: 0.1)
    let x = random(dim: 64, mean: 1.0, std: 0.2)
  
    // Apprentissage sur 10 étapes
    for i in range(0, 10) {
        w = learn(w, x)
        print(norm(w))
    }
}
```

**Modes disponibles:**

- `"hebbian"` - Apprentissage Hebbien
- `"anti_hebbian"` - Oubli/désapprentissage
- `"stdp"` - Spike-Timing Dependent Plasticity
- `"competitive"` - Apprentissage compétitif
- `"backprop"` - Rétropropagation

### @atomic - Transactions Sécurisées

```normil
@atomic
fn safe_normalize(v: Vec) -> Vec {
    let n = norm(v)
  
    if n < 0.001 {
        // Évite division par zéro - rollback
        return ones(dim: 64)
    }
  
    return normalize(v)  // Commit si OK
}

fn main() {
    let v_zero = zeros(dim: 64)
    let v_safe = safe_normalize(v_zero)
    print(norm(v_safe))  // ~8.0 (fallback sur ones)
}
```

### Combinaison @atomic + @plastic

```normil
@atomic
@plastic(rate: 0.005, mode: "hebbian")
fn safe_learn(weights: Vec, input: Vec, max_norm: float) -> Vec {
    let delta = scale(vec_mul(weights, input), 0.005)
    let new_weights = vec_add(weights, delta)
  
    // Vérification de stabilité
    if norm(new_weights) > max_norm {
        // Rollback si trop instable
        return weights
    }
  
    return normalize(new_weights)
}

fn main() {
    let w = random(dim: 64, mean: 0.0, std: 0.1)
    let x_normal = random(dim: 64, mean: 0.5, std: 0.1)
    let x_large = random(dim: 64, mean: 5.0, std: 2.0)
  
    w = safe_learn(w, x_normal, max_norm: 2.0)  // OK
    print(norm(w))
  
    w = safe_learn(w, x_large, max_norm: 2.0)   // Rollback
    print(norm(w))  // Même valeur
}
```

---

## Arguments Nommés

Les arguments nommés rendent le code plus lisible et permettent de les passer dans n'importe quel ordre :

```normil
fn create_config(size: int, rate: float, mode: str, enabled: bool) -> str {
    let msg = "Config: size=" + to_string(size) + 
              " rate=" + to_string(rate) +
              " mode=" + mode
    return msg
}

fn main() {
    // Ordre flexible grâce aux noms
    let cfg1 = create_config(
        size: 128,
        rate: 0.01,
        mode: "hebbian",
        enabled: true
    )
  
    // Autre ordre, même résultat
    let cfg2 = create_config(
        enabled: true,
        mode: "stdp",
        size: 256,
        rate: 0.005
    )
  
    print(cfg1)
    print(cfg2)
}
```

---

## Exemples Complets

### 1. Classification Simple

```normil
fn classify_score(score: float) -> str {
    match score {
        case float(s) where s >= 0.9 -> { return "Excellent" }
        case float(s) where s >= 0.7 -> { return "Bien" }
        case float(s) where s >= 0.5 -> { return "Moyen" }
        case _ -> { return "Insuffisant" }
    }
}

fn main() {
    let scores = [0.95, 0.75, 0.55, 0.35]
  
    for s in scores {
        let grade = classify_score(s)
        let msg = "Score " + to_string(s) + ": " + grade
        print(msg)
    }
}
```

### 2. Opérations Vectorielles avec Module

```normil
import vectors as vec

fn main() {
    let v1 = random(dim: 128, mean: 1.0, std: 0.2)
    let v2 = random(dim: 128, mean: 0.5, std: 0.1)
  
    // Normalisation
    let n1 = vec.create_normalized(v1)
    let n2 = vec.create_normalized(v2)
  
    // Similarité
    let similarity = vec.compute_similarity(n1, n2)
    print("Similarité: " + to_string(similarity))
  
    // Combinaison
    let combined = vec.weighted_sum(
        v1: n1, w1: 0.7,
        v2: n2, w2: 0.3
    )
    print("Norme combinée: " + to_string(norm(combined)))
  
    // Distance
    let dist = vec.distance(v1, v2)
    print("Distance: " + to_string(dist))
}
```

### 3. Système d'Apprentissage

```normil
@plastic(rate: 0.01, mode: "hebbian")
fn hebbian_update(w: Vec, x: Vec) -> Vec {
    let delta = scale(vec_mul(w, x), 0.01)
    return vec_add(w, delta)
}

@atomic
fn validate_weights(w: Vec, threshold: float) -> bool {
    return norm(w) < threshold
}

fn train_network(weights: Vec, signals: int, max_norm: float) -> Vec {
    let w = weights
  
    for i in range(0, signals) {
        let signal = random(dim: 64, mean: 1.0, std: 0.2)
        w = hebbian_update(w, signal)
      
        // Validation
        let is_valid = validate_weights(w, threshold: max_norm)
        if not is_valid {
            print("Poids normalisés à l'étape " + to_string(i))
            w = normalize(w)
        }
    }
  
    return w
}

fn main() {
    let initial_weights = random(dim: 64, mean: 0.0, std: 0.1)
  
    let trained = train_network(
        weights: initial_weights,
        signals: 20,
        max_norm: 3.0
    )
  
    print("Norme finale: " + to_string(norm(trained)))
}
```

---

## Référence Rapide des Primitives

### Vecteurs

```normil
zeros(dim: int) -> Vec
ones(dim: int) -> Vec
fill(dim: int, value: float) -> Vec
random(dim: int, mean: float, std: float) -> Vec

vec_add(v1: Vec, v2: Vec) -> Vec
vec_sub(v1: Vec, v2: Vec) -> Vec
vec_mul(v1: Vec, v2: Vec) -> Vec
scale(v: Vec, scalar: float) -> Vec

dot(v1: Vec, v2: Vec) -> float
norm(v: Vec) -> float
normalize(v: Vec) -> Vec
```

### Strings ✨

```normil
to_string(value: any) -> str

string_length(s: str) -> int
string_upper(s: str) -> str
string_lower(s: str) -> str
string_substring(s: str, start: int, end: int) -> str
string_replace(s: str, old: str, new: str) -> str
string_contains(s: str, sub: str) -> bool
string_startswith(s: str, prefix: str) -> bool
string_endswith(s: str, suffix: str) -> bool
string_trim(s: str) -> str
string_repeat(s: str, n: int) -> str
string_char_at(s: str, index: int) -> str
string_index_of(s: str, sub: str) -> int
```

### Utilitaires

```normil
print(value: any)
range(start: int, end: int) -> iterable
```

---

## Prochaines Étapes

1. **Tutorial complet** : Consultez `TUTORIAL.md` pour des leçons progressives
2. **Référence API** : `API_REFERENCE.md` pour tous les détails
3. **Exemples** : Explorez le dossier `examples/` :
   - `type_inference.nor` - Inférence de types
   - `imports_test.nor` - Système de modules
   - `string_operations.nor` - Opérations string
   - `advanced_patterns.nor` - Pattern matching
   - `neural_plasticity.nor` - Réseaux neuronaux
4. **Créez vos modules** : Organisez votre code dans `modules/`

---

## Aide et Support

- **Documentation** : Dossier `docs/`
- **Tests** : `python run_tests.py` pour valider l'installation
- **Exemples** : Dossier `examples/` pour s'inspirer

**Version** : NORMiL v0.4.0
**Phase actuelle** : Phase 3 (Inférence, Imports, Strings) - 60% complétée

**Bon coding ! 🚀**
