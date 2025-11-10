# 📐 NORMiL Style Guide

**Version** : 0.2.0
**Date** : Novembre 2025
**Auteur :** Diego Morales Magri
**Public** : Développeurs NORMiL

---

## 🎯 Philosophie

NORMiL privilégie :

- **Clarté** > Concision
- **Explicite** > Implicite
- **Sécurité** > Performance
- **Auditabilité** > Simplicité

---

## 📝 Conventions de Nommage

### Variables

```normil
// ✅ Bon : snake_case, descriptif
let user_count: int = 42
let max_iterations: int = 1000
let is_active: bool = true

// ❌ Mauvais : camelCase, abréviations obscures
let userCount: int = 42
let mi: int = 1000
let a: bool = true
```

### Fonctions

```normil
// ✅ Bon : snake_case, verbes d'action
fn calculate_similarity(v1: Vec, v2: Vec) -> float {
    return dot(v1, v2) / (norm(v1) * norm(v2))
}

fn process_episode(record: EpisodicRecord) -> Concept {
    // ...
}

// ❌ Mauvais : noms vagues, non descriptifs
fn calc(v1: Vec, v2: Vec) -> float { ... }
fn do_stuff(x: EpisodicRecord) -> Concept { ... }
```

### Types Personnalisés

```normil
// ✅ Bon : PascalCase
type UserProfile = {
    id: uuid,
    name: str,
    preferences: Vec
}

type SessionContext = {
    start_time: timestamp,
    duration: float
}

// ❌ Mauvais : snake_case pour types
type user_profile = { ... }
```

### Constantes

```normil
// ✅ Bon : UPPER_SNAKE_CASE
let MAX_VECTOR_DIM: int = 1024
let DEFAULT_LEARNING_RATE: float = 0.001
let API_TIMEOUT_SECONDS: int = 30

// ❌ Mauvais : lowercase
let max_dim: int = 1024
```

---

## 🔤 Indentation et Formatage

### Espacement

```normil
// ✅ Bon : 4 espaces (ou 1 tab)
fn compute_average(values: list<float>) -> float {
    let sum = 0.0
    let count = 0
  
    for value in values {
        sum = sum + value
        count = count + 1
    }
  
    return sum / count
}

// ❌ Mauvais : 2 espaces, inconsistant
fn compute_average(values: list<float>) -> float {
  let sum = 0.0
    let count = 0
  for value in values {
      sum = sum + value
  }
  return sum / count
}
```

### Espaces autour des opérateurs

```normil
// ✅ Bon
let x = 42 + 10
let similarity = dot(v1, v2) / norm(v1)
if x > 10 && y < 20 { ... }

// ❌ Mauvais
let x=42+10
let similarity=dot(v1,v2)/norm(v1)
if x>10&&y<20{ ... }
```

### Lignes vides

```normil
// ✅ Bon : séparer les blocs logiques
fn process_data(input: Vec) -> Vec {
    // Normaliser
    let normalized = normalize(input)
  
    // Appliquer transformation
    let transformed = scale(normalized, 2.0)
  
    // Retourner résultat
    return transformed
}

// ❌ Mauvais : tout collé
fn process_data(input: Vec) -> Vec {
    let normalized = normalize(input)
    let transformed = scale(normalized, 2.0)
    return transformed
}
```

---

## 💬 Commentaires

### Commentaires de ligne

```normil
// ✅ Bon : expliquer le "pourquoi"
let threshold = 0.85  // Seuil empirique après tests
let max_retries = 3   // Conforme à la RFC-1234

// ❌ Mauvais : répéter le code
let threshold = 0.85  // Assigner 0.85 à threshold
```

### Commentaires de bloc

```normil
// ✅ Bon : documenter les fonctions complexes
/*
 * Consolide les épisodes similaires en concept.
 * 
 * Algorithme:
 * 1. Grouper par similarité (threshold)
 * 2. Calculer centroid
 * 3. Créer concept avec metadata
 * 
 * Complexité: O(n²) - optimiser si n > 1000
 */
fn consolidate_episodes(episodes: list<EpisodicRecord>, threshold: float) -> Concept {
    // ...
}
```

### En-têtes de fichier

```normil
// ============================================
// memory_system.nor
// Système de gestion de la mémoire épisodique
// 
// Auteur: Diego Morales
// Date: 2025-01-15
// Version: 1.0
// ============================================
```

---

## 🏗️ Structure du Code

### Ordre des déclarations

```normil
// 1. Imports (quand supportés)
import memory
import vectors

// 2. Constantes
let MAX_EPISODES: int = 10000
let DEFAULT_DIM: int = 256

// 3. Types personnalisés
type Episode = { ... }

// 4. Fonctions utilitaires
fn normalize_vector(v: Vec) -> Vec { ... }

// 5. Fonctions principales
fn main() { ... }
```

### Longueur des fonctions

```normil
// ✅ Bon : fonctions courtes (< 50 lignes)
fn process_episode(e: EpisodicRecord) -> Concept {
    let vec = normalize(e.vecs["default"])
    let similar = find_similar(vec, 10)
    return consolidate(similar, 0.8)
}

// ❌ Mauvais : fonction trop longue (> 100 lignes)
fn do_everything() {
    // 200 lignes de code...
}
```

---

## 🎨 Patterns Recommandés

### Arguments Nommés

```normil
// ✅ Bon : clarifier les paramètres
let v = random(dim: 256, mean: 0.0, std: 1.0)
let results = query_memory(vector: query_vec, k: 10, threshold: 0.7)

// ❌ Acceptable mais moins clair
let v = random(256, 0.0, 1.0)
let results = query_memory(query_vec, 10, 0.7)
```

### Early Return

```normil
// ✅ Bon : retourner tôt pour éviter nesting
fn validate_vector(v: Vec) -> bool {
    if v.dim < 1 {
        return false
    }
  
    if v.dim > MAX_DIM {
        return false
    }
  
    return true
}

// ❌ Mauvais : nesting profond
fn validate_vector(v: Vec) -> bool {
    if v.dim >= 1 {
        if v.dim <= MAX_DIM {
            return true
        } else {
            return false
        }
    } else {
        return false
    }
}
```

### Gestion d'Erreurs

```normil
// ✅ Bon : vérifier les conditions d'erreur d'abord
fn divide(a: float, b: float) -> float {
    if b == 0.0 {
        print("Erreur: division par zéro")
        return 0.0
    }
  
    return a / b
}
```

---

## 🔒 Bonnes Pratiques

### Types Explicites

```normil
// ✅ Bon : types explicites pour clarté
fn calculate_score(features: Vec, weights: Vec) -> float {
    let raw_score: float = dot(features, weights)
    let normalized: float = raw_score / norm(weights)
    return normalized
}

// ⚠️ Acceptable en REPL, éviter en production
fn calculate_score(features, weights) {
    let raw_score = dot(features, weights)
    return raw_score / norm(weights)
}
```

### Immutabilité

```normil
// ✅ Bon : préférer let (immutable)
let x = 42
let y = x + 10

// ⚠️ À éviter : mutation (quand var sera supporté)
var x = 42
x = x + 10  // Mutation = risque d'erreur
```

### Noms Significatifs

```normil
// ✅ Bon
let user_query_vector = random(256)
let similarity_threshold = 0.85
let top_k_results = 10

// ❌ Mauvais
let v = random(256)
let t = 0.85
let k = 10
```

---

## 📊 Organisation de Projet

### Structure Recommandée

```
project/
├── main.nor                # Point d'entrée
├── config/
│   └── constants.nor       # Constantes globales
├── types/
│   ├── memory_types.nor    # Types mémoire
│   └── vector_types.nor    # Types vectoriels
├── utils/
│   ├── vector_ops.nor      # Opérations vectorielles
│   └── validation.nor      # Validations
└── core/
    ├── memory.nor          # Logique mémoire
    └── learning.nor        # Logique apprentissage
```

### Fichiers Modulaires

```normil
// ✅ Bon : 1 fichier = 1 responsabilité
// memory_operations.nor - Opérations mémoire uniquement
// vector_operations.nor - Opérations vectorielles uniquement

// ❌ Mauvais : 1 fichier = tout
// everything.nor - 5000 lignes de code mixte
```

---

## 🧪 Tests et Documentation

### Fonctions Testables

```normil
// ✅ Bon : fonction pure, testable
fn cosine_similarity(v1: Vec, v2: Vec) -> float {
    return dot(v1, v2) / (norm(v1) * norm(v2))
}

// Test : cosine_similarity(ones(10), ones(10)) == 1.0
```

### Documentation Fonction

```normil
/*
 * Calcule la similarité cosinus entre deux vecteurs.
 * 
 * Paramètres:
 *   v1: Premier vecteur (normalisé ou non)
 *   v2: Second vecteur (normalisé ou non)
 * 
 * Retourne:
 *   Similarité entre -1.0 et 1.0
 * 
 * Exemple:
 *   let sim = cosine_similarity(v1, v2)
 *   if sim > 0.8 { print("Très similaire") }
 */
fn cosine_similarity(v1: Vec, v2: Vec) -> float {
    return dot(v1, v2) / (norm(v1) * norm(v2))
}
```

---

## ⚡ Performance

### Éviter les Calculs Répétés

```normil
// ✅ Bon : calculer une fois
let v_norm = norm(v)
let v1_normalized = scale(v, 1.0 / v_norm)

// ❌ Mauvais : recalculer norm() à chaque fois
let v1_normalized = scale(v, 1.0 / norm(v))
let similarity = dot(v1_normalized, v2) / norm(v)  // norm(v) recalculé
```

### Préférer les Primitives

```normil
// ✅ Bon : utiliser primitives optimisées
let sum_vec = vec_add(v1, v2)

// ❌ Mauvais : boucler manuellement (quand ce sera possible)
let sum_vec = zeros(256)
for i in range(256) {
    sum_vec[i] = v1[i] + v2[i]  // Lent
}
```

---

## 🚫 Anti-Patterns

### Magic Numbers

```normil
// ❌ Mauvais
let v = random(256)
if similarity > 0.85 { ... }

// ✅ Bon
let VECTOR_DIM: int = 256
let SIMILARITY_THRESHOLD: float = 0.85

let v = random(VECTOR_DIM)
if similarity > SIMILARITY_THRESHOLD { ... }
```

### Noms Trop Courts

```normil
// ❌ Mauvais
let a = random(256)
let b = ones(256)
let c = vec_add(a, b)

// ✅ Bon
let query_vector = random(256)
let bias_vector = ones(256)
let adjusted_query = vec_add(query_vector, bias_vector)
```

### Commentaires Obsolètes

```normil
// ❌ Mauvais : commentaire ne correspond plus au code
// Calculer la moyenne
let sum = calculate_median(values)  // Incohérent !

// ✅ Bon : commentaire à jour
// Calculer la médiane
let median = calculate_median(values)
```

---

## 📚 Ressources

- **Spécification NORMiL** : `SPECIFICATION.md`
- **Guide Démarrage** : `QUICKSTART.md`
- **Exemples** : `examples/*.nor`
- **Documentation API** : `README.md`

---

## ✅ Checklist Code Review

Avant de commiter du code NORMiL :

- [ ] Noms de variables/fonctions descriptifs
- [ ] Indentation cohérente (4 espaces)
- [ ] Types explicites sur les fonctions publiques
- [ ] Commentaires pour logique complexe
- [ ] Pas de magic numbers
- [ ] Fonctions < 50 lignes
- [ ] Tests pour fonctions critiques
- [ ] Documentation si API publique

---

**Ce guide évolue avec NORMiL. Vos suggestions sont bienvenues !** 🚀
