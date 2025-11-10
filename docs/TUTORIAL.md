# Tutorial NORMiL - De Zéro à Expert


**Date** : Novembre 2025
**Auteur** : Diego Morales Magri

---

## Apprenez NORMiL par la pratique

---

## Table des Matières

1. [Niveau 1 : Débutant](#niveau-1--débutant)
2. [Niveau 2 : Intermédiaire](#niveau-2--intermédiaire)
3. [Niveau 3 : Avancé](#niveau-3--avancé)
4. [Niveau 4 : Expert](#niveau-4--expert)
5. [Projets Complets](#projets-complets)

---

## Niveau 1 : Débutant

### Leçon 1.1 : Premier Programme

**Objectif** : Afficher du texte et faire des calculs simples

```normil
fn main() {
    print("Bonjour NORMiL!")
  
    let x: int = 10
    let y: int = 20
    let somme: int = x + y
  
    print("La somme est:")
    print(somme)
}
```

**Exercice** : Modifiez ce programme pour calculer la différence et le produit.

---

### Leçon 1.2 : Variables et Types

**Objectif** : Comprendre les types de données

```normil
fn main() {
    // Types de base avec annotations
    let age: int = 25
    let temperature: float = 36.6
    let nom: str = "Alice"
    let actif: bool = true
  
    print("Informations:")
    print(nom)
    print(age)
    print(temperature)
    print(actif)
}
```

**Exercice** : Créez une fiche d'identité avec 5 variables différentes.

---

### Leçon 1.2b : Inférence de Types (✨ Phase 3.1)

**Objectif** : Laisser NORMiL déduire les types automatiquement

```normil
fn main() {
    // Inférence automatique - pas besoin de spécifier le type!
    let age = 25              // Déduit: int
    let temperature = 36.6    // Déduit: float
    let nom = "Alice"         // Déduit: str
    let actif = true          // Déduit: bool
  
    // Même avec des expressions
    let somme = 10 + 20       // Déduit: int
    let moyenne = 10.5 / 2.0  // Déduit: float
    let message = "Bonjour"   // Déduit: str
  
    print("L'inférence fonctionne!")
    print(age)
    print(somme)
}
```

**Points clés** :

- ✅ `let x = 42` au lieu de `let x: int = 42`
- ✅ Fonctionne avec literals, expressions, retours de fonction
- ✅ Déduction intelligente : bool → int → float → str → Vec
- ✅ Compatible avec vecteurs et fonctions

**Exercice** : Réécrivez l'exercice 1.2 sans aucune annotation de type.

---

### Leçon 1.3 : Fonctions Simples

**Objectif** : Créer et appeler des fonctions

```normil
fn double(x: int) -> int {
    return x * 2
}

fn triple(x: int) -> int {
    return x * 3
}

fn main() {
    let nombre: int = 5
  
    let d = double(nombre)
    let t = triple(nombre)
  
    print("Double:")
    print(d)
    print("Triple:")
    print(t)
}
```

**Exercice** : Créez une fonction `quadruple` et une fonction `carre`.

---

### Leçon 1.4 : Conditions

**Objectif** : Utiliser if-else

```normil
fn evaluer_note(note: int) -> str {
    if note >= 90 {
        return "Excellent"
    } else if note >= 70 {
        return "Bien"
    } else if note >= 50 {
        return "Passable"
    } else {
        return "Insuffisant"
    }
}

fn main() {
    print(evaluer_note(95))
    print(evaluer_note(75))
    print(evaluer_note(45))
}
```

**Exercice** : Créez une fonction qui classe une température en "chaud", "tiède", "froid".

---

### Leçon 1.5 : Boucles

**Objectif** : Répéter des actions

```normil
fn compter_jusque(n: int) {
    let i = 0
    while i < n {
        print(i)
        i = i + 1
    }
}

fn compter_range(debut: int, fin: int) {
    for i in range(debut, fin) {
        print(i)
    }
}

fn main() {
    print("While:")
    compter_jusque(5)
  
    print("For:")
    compter_range(0, 5)
}
```

**Exercice** : Créez une fonction qui calcule la somme de 1 à N.

---

### Leçon 1.6 : Opérations sur Chaînes (✨ Phase 3.3)

**Objectif** : Manipuler des chaînes de caractères

```normil
fn main() {
    // Concaténation avec l'opérateur +
    let prenom = "Alice"
    let nom = "Dupont"
    let nom_complet = prenom + " " + nom
    print(nom_complet)  // "Alice Dupont"
  
    // Conversion vers string
    let age = 25
    let message = "J'ai " + to_string(age) + " ans"
    print(message)  // "J'ai 25 ans"
  
    // Primitives string
    let texte = "NORMiL"
    print(string_length(texte))        // 6
    print(string_upper(texte))         // "NORMIL"
    print(string_lower(texte))         // "normil"
  
    // Manipulation
    let phrase = "Hello World"
    print(string_substring(phrase, 0, 5))   // "Hello"
    print(string_replace(phrase, "World", "NORMiL"))  // "Hello NORMiL"
    print(string_contains(phrase, "World")) // true
  
    // Répétition
    print(string_repeat("Ha", 3))  // "HaHaHa"
}
```

**Primitives disponibles** :

- `string_length(s: str) -> int` - Longueur de la chaîne
- `string_upper(s: str) -> str` - En majuscules
- `string_lower(s: str) -> str` - En minuscules
- `string_substring(s: str, start: int, end: int) -> str` - Sous-chaîne
- `string_split(s: str, sep: str) -> str` - Découpe (retourne premier élément)
- `string_join(items: str, sep: str) -> str` - Joint avec séparateur
- `string_replace(s: str, old: str, new: str) -> str` - Remplace
- `string_contains(s: str, sub: str) -> bool` - Contient?
- `string_startswith(s: str, prefix: str) -> bool` - Commence par?
- `string_endswith(s: str, suffix: str) -> bool` - Finit par?
- `string_trim(s: str) -> str` - Enlève espaces début/fin
- `string_repeat(s: str, n: int) -> str` - Répète n fois
- `string_char_at(s: str, index: int) -> str` - Caractère à l'index
- `string_index_of(s: str, sub: str) -> int` - Position de sous-chaîne

**Exercice** : Créez une fonction qui formate un nom en "NOM, Prénom" (majuscules).

---

## Niveau 2 : Intermédiaire

### Leçon 2.1 : Vecteurs de Base

**Objectif** : Manipuler des vecteurs

```normil
fn main() {
    // Création de vecteurs - avec inférence de type!
    let v1 = zeros(dim: 64)
    let v2 = ones(dim: 64)
    let v3 = fill(dim: 64, value: 0.5)
    let v4 = random(dim: 64, mean: 0.0, std: 1.0)
  
    // Opérations
    let somme = vec_add(v1, v2)
    let produit = vec_mul(v2, v3)
    let double_v = scale(v2, 2.0)
  
    // Métriques
    print("Norme de v4:")
    print(norm(v4))
  
    print("Norme de somme:")
    print(norm(somme))
}
```

**Exercice** : Créez 3 vecteurs et calculez leur moyenne.

---

### Leçon 2.1b : Modules et Imports (✨ Phase 3.2)

**Objectif** : Réutiliser du code avec le système de modules

**Créez un module** : `modules/math_utils.nor`

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
```

**Utilisez le module** : `main.nor`

```normil
import math_utils

fn main() {
    // Appel de fonctions du module
    let valeur = math_utils.abs(-42.0)
    print(valeur)  // 42.0
  
    let maximum = math_utils.max(10.0, 25.0)
    print(maximum)  // 25.0
}
```

**Avec alias** :

```normil
import math_utils as math

fn main() {
    print(math.abs(-42.0))
    print(math.max(10.0, 25.0))
}
```

**Modules disponibles** :

- `modules/math.nor` - Fonctions mathématiques (abs, max, min, clamp)
- `modules/vectors.nor` - Opérations vectorielles avancées

**Points clés** :

- ✅ Modules dans le dossier `modules/`
- ✅ Import avec ou sans alias
- ✅ Accès via `module.fonction()`
- ✅ Caching automatique (module chargé une seule fois)
- ✅ Scopes isolés entre modules

**Exercice** : Créez un module `string_utils.nor` avec 3 fonctions utiles.

---

### Leçon 2.1c : Interopérabilité Python (🚀 Phase 4.1)

**Objectif** : Utiliser des bibliothèques Python depuis NORMiL

NORMiL peut importer et utiliser **n'importe quel module Python** ! Cela vous donne accès à tout l'écosystème Python : NumPy, SciPy, pandas, et plus encore.

**Import de modules Python standards** :

```normil
import math

fn main() {
    // Accès aux constantes Python
    let pi = math.pi
    print(pi)  // 3.141592653589793
  
    let e = math.e
    print(e)   // 2.718281828459045
  
    // Appel de fonctions Python
    let racine = math.sqrt(16.0)
    print(racine)  // 4.0
  
    let puissance = math.pow(2.0, 3.0)
    print(puissance)  // 8.0
}
```

**Avec alias** :

```normil
import math as m

fn main() {
    let aire_cercle = m.pi * 5.0 * 5.0
    print(aire_cercle)  // 78.53981633974483
}
```

**Fonctions trigonométriques** :

```normil
import math

fn main() {
    let angle = math.pi / 4.0  // 45 degrés en radians
  
    let sin_val = math.sin(angle)
    print(sin_val)  // 0.7071...
  
    let cos_val = math.cos(angle)
    print(cos_val)  // 0.7071...
  
    let tan_val = math.tan(angle)
    print(tan_val)  // 1.0
}
```

**Module random** :

```normil
import random

fn main() {
    // Fixer la seed pour reproductibilité
    random.seed(42)
  
    // Nombre aléatoire entre 0 et 1
    let val = random.random()
    print(val)
  
    // Entier aléatoire
    let dice = random.randint(1, 6)
    print(dice)
}
```

**Utilisation dans des fonctions** :

```normil
import math

fn aire_cercle(rayon: float) -> float {
    return math.pi * rayon * rayon
}

fn volume_sphere(rayon: float) -> float {
    return (4.0 / 3.0) * math.pi * rayon * rayon * rayon
}

fn main() {
    let r = 5.0
  
    print("Aire du cercle:")
    print(aire_cercle(r))
  
    print("Volume de la sphère:")
    print(volume_sphere(r))
}
```

**Mélanger modules NORMiL et Python** :

```normil
import math           // Module Python
import mathutils      // Module NORMiL

fn main() {
    // Fonctions Python
    let sqrt_val = math.sqrt(25.0)
    let sin_val = math.sin(math.pi)
  
    // Fonctions NORMiL
    let abs_val = mathutils.abs(-42)
    let max_val = mathutils.max(10, 20)
  
    print(sqrt_val)
    print(abs_val)
}
```

**Appels imbriqués** :

```normil
import math

fn main() {
    // Les appels Python peuvent être imbriqués
    let resultat = math.sqrt(math.pow(3.0, 2.0) + math.pow(4.0, 2.0))
    print(resultat)  // 5.0 (théorème de Pythagore)
  
    // Dans des expressions complexes
    let aire = math.pi * math.pow(math.sqrt(100.0), 2.0)
    print(aire)  // 314.159...
}
```

**Points clés** :

- ✅ Import transparent : `import math`, `import random`, `import sys`, etc.
- ✅ Accès aux constantes : `math.pi`, `math.e`, `math.inf`
- ✅ Appel de fonctions : `math.sqrt()`, `math.sin()`, `random.random()`
- ✅ Alias supportés : `import math as m`
- ✅ Détection automatique : NORMiL cherche d'abord `.nor`, puis Python
- ✅ Mix NORMiL/Python dans le même code
- ✅ Cache intelligent : module chargé une seule fois

**Modules Python utiles** :

- `math` - Fonctions mathématiques
- `random` - Génération aléatoire
- `datetime` - Manipulation de dates
- `json` - Parsing JSON
- `sys` - Informations système
- `os` - Opérations système
- `collections` - Structures de données
- (Et tous les autres modules Python disponibles !)

**Exercice** : Créez un programme qui utilise `math` et `random` pour générer des coordonnées aléatoires dans un cercle unitaire.

---

### Leçon 2.1d : Objets Python (🚀 Phase 4.4)

**Objectif** : Manipuler des objets et classes Python

NORMiL permet d'accéder aux objets Python, leurs méthodes et attributs de manière native.

#### Méthodes sur les Types Natifs

**Méthodes sur les chaînes** :

```normil
fn manipuler_texte() {
    let message = "bonjour le monde"
  
    // Conversion casse
    let upper = message.upper()
    print(upper)  // "BONJOUR LE MONDE"
  
    let lower = upper.lower()
    print(lower)  // "bonjour le monde"
  
    // Remplacement
    let nouveau = message.replace("monde", "NORMiL")
    print(nouveau)  // "bonjour le NORMiL"
  
    // Découpage
    let mots = message.split(" ")
    print(mots)  // ["bonjour", "le", "monde"]
  
    // Tests
    let commence = message.startswith("bonjour")
    print(commence)  // true
}
```

**Méthodes sur les listes** :

```normil
fn manipuler_listes() {
    let nombres = [1, 2, 3]
  
    // Ajouter des éléments
    nombres.append(4)
    nombres.append(5)
    print(nombres)  // [1, 2, 3, 4, 5]
}
```

#### Chaînage de Méthodes

Les méthodes peuvent être chaînées :

```normil
fn chainer_methodes() {
    let texte = "  hello world  "
  
    // Chaîner strip() puis upper()
    let resultat = texte.strip().upper()
    print(resultat)  // "HELLO WORLD"
  
    // Chaînes complexes
    let complexe = "  python rocks  "
        .strip()
        .replace("python", "NORMiL")
        .upper()
    print(complexe)  // "NORMIL ROCKS"
}
```

#### Instantiation de Classes Python

**Créer des objets** :

```normil
import datetime

fn utiliser_datetime() {
    // Instantiation d'une classe Python
    let noel = datetime.datetime(2024, 12, 25)
  
    // Accès aux attributs
    print(noel.year)   // 2024
    print(noel.month)  // 12
    print(noel.day)    // 25
  
    // Appel de méthodes
    let jour_semaine = noel.weekday()
    print(jour_semaine)  // 2 (mercredi, 0=lundi)
}
```

#### Accès aux Attributs

Les attributs d'objets Python sont accessibles avec `.` :

```normil
import datetime

fn explorer_attributs() {
    let date = datetime.datetime(2024, 6, 15)
  
    // Attributs simples
    let annee = date.year
    let mois = date.month
    let jour = date.day
  
    print(annee)  // 2024
    print(mois)   // 6
    print(jour)   // 15
}
```

#### Exemples Pratiques

**Validation d'email** :

```normil
fn valider_email(email: str) -> bool {
    // Utiliser les méthodes Python
    let parties = email.split("@")
  
    if parties.length == 2 {
        let commence_ok = parties[0].length > 0
        let domaine_ok = parties[1].length > 0
        return commence_ok && domaine_ok
    }
  
    return false
}

fn main() {
    let email1 = "user@example.com"
    let email2 = "@example.com"
  
    print(valider_email(email1))  // true
    print(valider_email(email2))  // false
}
```

**Parsing CSV simple** :

```normil
fn parser_csv(ligne: str) -> [str] {
    return ligne.split(",")
}

fn main() {
    let entetes = "nom,prenom,age"
    let colonnes = parser_csv(entetes)
  
    // Transformer en titres
    let titre1 = colonnes[0].upper()
    let titre2 = colonnes[1].upper()
    let titre3 = colonnes[2].upper()
  
    print(titre1)  // "NOM"
    print(titre2)  // "PRENOM"
    print(titre3)  // "AGE"
}
```

**Calculs avec datetime** :

```normil
import datetime

fn analyser_dates() {
    let nouvel_an = datetime.datetime(2024, 1, 1)
    let mi_annee = datetime.datetime(2024, 6, 15)
    let fin_annee = datetime.datetime(2024, 12, 31)
  
    // Extraire informations
    print(nouvel_an.month)  // 1
    print(mi_annee.month)   // 6
    print(fin_annee.month)  // 12
  
    // Jour de la semaine
    print(nouvel_an.weekday())  // Lundi = 0
}
```

**Points clés** :

- ✅ Méthodes natives : `.upper()`, `.lower()`, `.split()`, `.replace()`, etc.
- ✅ Chaînage : `text.strip().upper()`
- ✅ Classes Python : `datetime.datetime(2024, 1, 1)`
- ✅ Attributs objets : `date.year`, `date.month`, `date.day`
- ✅ Méthodes objets : `date.weekday()`
- ✅ Types Python : str, list, datetime, etc.
- ✅ Totalement transparent : comme du code NORMiL natif

**Limitations** :

- ⚠️ Pas de support kwargs Python (`func(x=1, y=2)`)
- ⚠️ Certains types complexes peuvent nécessiter des conversions
- ⚠️ Les exceptions Python sont propagées

**Exercice** : Créez un programme qui parse une date au format "JJ/MM/AAAA" en utilisant `.split()` et crée un objet `datetime.datetime`.

---

### Leçon 2.2 : Arguments Nommés

**Objectif** : Utiliser les arguments nommés pour la clarté

```normil
fn creer_vecteur_personnalise(
    taille: int,
    valeur_moyenne: float,
    deviation: float,
    normaliser: bool
) -> Vec {
    let v = random(dim: taille, mean: valeur_moyenne, std: deviation)
  
    if normaliser {
        return normalize(v)
    } else {
        return v
    }
}

fn main() {
    // Ordre des arguments clair et flexible
    let v1 = creer_vecteur_personnalise(
        taille: 128,
        valeur_moyenne: 1.0,
        deviation: 0.2,
        normaliser: true
    )
  
    let v2 = creer_vecteur_personnalise(
        normaliser: false,
        taille: 64,
        deviation: 0.5,
        valeur_moyenne: 0.0
    )
  
    print(norm(v1))  // Devrait être ~1.0
    print(norm(v2))  // Norme variable
}
```

**Exercice** : Créez une fonction avec 5 paramètres nommés pour configurer un réseau.

---

### Leçon 2.3 : Pattern Matching - Bases

**Objectif** : Utiliser le pattern matching

```normil
fn classifier_nombre(n: int) -> str {
    match n {
        case 0 -> {
            return "zero"
        }
        case 1 -> {
            return "un"
        }
        case int(x) where x < 0 -> {
            return "negatif"
        }
        case int(x) where x > 100 -> {
            return "tres grand"
        }
        case _ -> {
            return "autre"
        }
    }
}

fn main() {
    print(classifier_nombre(0))      // "zero"
    print(classifier_nombre(-5))     // "negatif"
    print(classifier_nombre(150))    // "tres grand"
    print(classifier_nombre(42))     // "autre"
}
```

**Exercice** : Créez un classifier pour les jours de la semaine (1-7).

---

### Leçon 2.4 : Pattern Matching Avancé

**Objectif** : Combiner patterns et conditions

```normil
fn analyser_score(score: float) -> str {
    match score {
        case float(s) where s >= 0.95 -> {
            return "Exceptionnel"
        }
        case float(s) where s >= 0.85 -> {
            return "Excellent"
        }
        case float(s) where s >= 0.70 -> {
            return "Tres bien"
        }
        case float(s) where s >= 0.55 -> {
            return "Bien"
        }
        case float(s) where s >= 0.40 -> {
            return "Moyen"
        }
        case _ -> {
            return "Insuffisant"
        }
    }
}

fn main() {
    let scores = [0.99, 0.87, 0.65, 0.42, 0.20]
  
    for s in scores {
        print(analyser_score(s))
    }
}
```

**Exercice** : Créez un analyseur de température avec 6 catégories.

---

## Niveau 3 : Avancé

### Leçon 3.1 : Annotation @plastic

**Objectif** : Implémenter la plasticité neuronale

```normil
@plastic(rate: 0.01, mode: "hebbian")
fn apprentissage_hebbien(poids: Vec, entree: Vec) -> Vec {
    // Hebbian learning: "Neurons that fire together, wire together"
    let produit = vec_mul(poids, entree)
    let increment = scale(produit, 0.01)
    let nouveaux_poids = vec_add(poids, increment)
    return normalize(nouveaux_poids)
}

fn main() {
    let poids = random(dim: 64, mean: 0.0, std: 0.1)
    let signal = random(dim: 64, mean: 1.0, std: 0.2)
  
    print("Norme initiale:")
    print(norm(poids))
  
    // 10 étapes d'apprentissage
    for i in range(0, 10) {
        poids = apprentissage_hebbien(poids, signal)
    }
  
    print("Norme finale:")
    print(norm(poids))
}
```

**Exercice** : Testez avec différents `rate` (0.001, 0.01, 0.1) et observez.

---

### Leçon 3.2 : Modes de Plasticité

**Objectif** : Comparer les différents modes

```normil
@plastic(rate: 0.01, mode: "hebbian")
fn hebb(w: Vec, x: Vec) -> Vec {
    let delta = scale(vec_mul(w, x), 0.01)
    return vec_add(w, delta)
}

@plastic(rate: 0.01, mode: "anti_hebbian")
fn anti_hebb(w: Vec, x: Vec) -> Vec {
    let delta = scale(vec_mul(w, x), 0.01)
    return vec_sub(w, delta)
}

@plastic(rate: 0.01, mode: "stdp")
fn stdp(w: Vec, x: Vec) -> Vec {
    // STDP avec timing simulé
    let timing_factor = scale(x, 0.8)  // Simule le timing
    let delta = scale(vec_mul(w, timing_factor), 0.01)
    return vec_add(w, delta)
}

fn main() {
    let w_init = random(dim: 32, mean: 0.0, std: 0.1)
    let signal = random(dim: 32, mean: 1.0, std: 0.1)
  
    let w_hebb = hebb(w_init, signal)
    let w_anti = anti_hebb(w_init, signal)
    let w_stdp = stdp(w_init, signal)
  
    print("Hebbian:")
    print(norm(w_hebb))
  
    print("Anti-Hebbian:")
    print(norm(w_anti))
  
    print("STDP:")
    print(norm(w_stdp))
}
```

**Exercice** : Ajoutez le mode "competitive" et comparez.

---

### Leçon 3.3 : Annotation @atomic

**Objectif** : Transactions avec rollback automatique

```normil
@atomic
fn mise_a_jour_securisee(valeur: int, increment: int) -> int {
    let temp = valeur + increment
  
    // Si erreur ici, rollback automatique
    if temp < 0 {
        return valeur  // Pas de changement
    }
  
    return temp
}

@atomic
fn normalisation_atomique(v: Vec) -> Vec {
    let n = norm(v)
  
    if n < 0.001 {
        // Éviter division par zéro
        return ones(dim: 64)
    }
  
    return normalize(v)
}

fn main() {
    let x = 10
    let y = mise_a_jour_securisee(x, 5)
    let z = mise_a_jour_securisee(x, -20)
  
    print(y)  // 15
    print(z)  // 10 (rollback)
  
    let v_zero = zeros(dim: 64)
    let v_safe = normalisation_atomique(v_zero)
    print(norm(v_safe))  // ~8.0 (norme de ones(64))
}
```

**Exercice** : Créez une fonction @atomic pour des transferts d'argent.

---

### Leçon 3.4 : Combinaison @atomic + @plastic

**Objectif** : Apprentissage sécurisé avec transactions

```normil
@atomic
@plastic(rate: 0.005, mode: "hebbian")
fn apprentissage_securise(poids: Vec, entree: Vec, seuil: float) -> Vec {
    // Calcul plastique
    let delta = scale(vec_mul(poids, entree), 0.005)
    let nouveau = vec_add(poids, delta)
  
    // Vérification de stabilité
    let n = norm(nouveau)
  
    if n > seuil {
        // Trop instable - rollback
        return poids
    }
  
    return normalize(nouveau)
}

fn main() {
    let poids = random(dim: 64, mean: 0.0, std: 0.1)
    let signal_normal = random(dim: 64, mean: 0.5, std: 0.1)
    let signal_fort = random(dim: 64, mean: 5.0, std: 2.0)
  
    print("Norme initiale:")
    print(norm(poids))
  
    // Signal normal - devrait fonctionner
    poids = apprentissage_securise(poids, signal_normal, seuil: 2.0)
    print("Après signal normal:")
    print(norm(poids))
  
    // Signal trop fort - rollback
    poids = apprentissage_securise(poids, signal_fort, seuil: 2.0)
    print("Après signal fort (rollback):")
    print(norm(poids))
}
```

**Exercice** : Ajoutez des seuils min et max pour la stabilité.

---

## Niveau 4 : Expert

### Projet 4.1 : Réseau de Neurones Simple

**Objectif** : Implémenter un perceptron avec apprentissage

```normil
@plastic(rate: 0.001, mode: "backprop")
fn backprop_update(poids: Vec, gradient: Vec) -> Vec {
    let correction = scale(gradient, 0.001)
    let nouveau = vec_sub(poids, correction)
    return normalize(nouveau)
}

@atomic
fn forward_propagation(entree: Vec, poids: Vec) -> Vec {
    let weighted = vec_mul(entree, poids)
    return normalize(weighted)
}

fn calculer_erreur(sortie: Vec, cible: Vec) -> Vec {
    return vec_sub(cible, sortie)
}

fn entrainer_reseau(
    poids_init: Vec,
    entrees: Vec,
    cibles: Vec,
    epochs: int
) -> Vec {
    let poids = poids_init
  
    for epoch in range(0, epochs) {
        // Forward
        let sortie = forward_propagation(entrees, poids)
      
        // Calcul erreur
        let erreur = calculer_erreur(sortie, cibles)
      
        // Backward
        poids = backprop_update(poids, erreur)
      
        if epoch % 10 == 0 {
            print("Epoch")
            print(epoch)
            print("Erreur:")
            print(norm(erreur))
        }
    }
  
    return poids
}

fn main() {
    let dim = 128
  
    let poids = random(dim: dim, mean: 0.0, std: 0.1)
    let entree = random(dim: dim, mean: 1.0, std: 0.2)
    let cible = random(dim: dim, mean: 0.5, std: 0.1)
  
    print("Entrainement...")
    let poids_entraines = entrainer_reseau(
        poids_init: poids,
        entrees: entree,
        cibles: cible,
        epochs: 50
    )
  
    print("Entrainement termine!")
    print("Norme finale:")
    print(norm(poids_entraines))
}
```

---

### Projet 4.2 : Système de Mémoire avec Consolidation

**Objectif** : Implémenter consolidation + oubli

```normil
@plastic(rate: 0.02, mode: "hebbian")
fn encoder_memoire(memoire: Vec, pattern: Vec) -> Vec {
    let association = vec_mul(memoire, pattern)
    let renforcement = scale(association, 0.02)
    return vec_add(memoire, renforcement)
}

@plastic(rate: 0.005, mode: "anti_hebbian")
fn oubli_progressif(memoire: Vec, bruit: Vec) -> Vec {
    let decay = scale(vec_mul(memoire, bruit), 0.005)
    return vec_sub(memoire, decay)
}

@atomic
fn consolider_memoire(memoire: Vec) -> Vec {
    let n = norm(memoire)
  
    if n < 0.1 {
        // Mémoire trop faible - réinitialiser
        return zeros(dim: 64)
    }
  
    if n > 2.0 {
        // Trop forte - normaliser
        return normalize(memoire)
    }
  
    return memoire
}

fn cycle_memoire(
    memoire_init: Vec,
    patterns: int,
    cycles: int
) -> Vec {
    let memoire = memoire_init
  
    for cycle in range(0, cycles) {
        // Encoder nouveau pattern
        let pattern = random(dim: 64, mean: 1.0, std: 0.2)
        memoire = encoder_memoire(memoire, pattern)
      
        // Oubli avec bruit
        let bruit = random(dim: 64, mean: 0.5, std: 0.1)
        memoire = oubli_progressif(memoire, bruit)
      
        // Consolidation
        memoire = consolider_memoire(memoire)
      
        print("Cycle")
        print(cycle)
        print("Norme:")
        print(norm(memoire))
    }
  
    return memoire
}

fn main() {
    let memoire = zeros(dim: 64)
  
    print("Simulation memoire...")
    let memoire_finale = cycle_memoire(
        memoire_init: memoire,
        patterns: 10,
        cycles: 20
    )
  
    print("Simulation terminee!")
}
```

---

### Projet 4.3 : Détecteur de Patterns avec Classification

**Objectif** : Combiner patterns, @plastic, @atomic

```normil
fn classifier_force(norme: float) -> str {
    match norme {
        case float(n) where n > 3.0 -> { return "tres fort" }
        case float(n) where n > 2.0 -> { return "fort" }
        case float(n) where n > 1.0 -> { return "moyen" }
        case float(n) where n > 0.5 -> { return "faible" }
        case _ -> { return "tres faible" }
    }
}

@plastic(rate: 0.01, mode: "competitive")
fn adapter_detecteur(detecteur: Vec, signal: Vec) -> Vec {
    let reponse = vec_mul(detecteur, signal)
    let adaptation = scale(reponse, 0.01)
    return vec_add(detecteur, adaptation)
}

@atomic
fn detecter_pattern(detecteur: Vec, signal: Vec, seuil: float) -> bool {
    let activation = dot(detecteur, signal)
  
    if activation > seuil {
        return true
    } else {
        return false
    }
}

fn entrainer_detecteur(
    detecteur_init: Vec,
    signaux_positifs: int,
    signaux_negatifs: int
) -> Vec {
    let detecteur = detecteur_init
  
    print("Phase 1: Apprentissage patterns positifs")
    for i in range(0, signaux_positifs) {
        let signal_pos = random(dim: 64, mean: 2.0, std: 0.3)
        detecteur = adapter_detecteur(detecteur, signal_pos)
      
        let classe = classifier_force(norm(detecteur))
        print(classe)
    }
  
    print("Phase 2: Adaptation patterns negatifs")
    for i in range(0, signaux_negatifs) {
        let signal_neg = random(dim: 64, mean: 0.2, std: 0.1)
        let inverse = scale(signal_neg, -0.5)
        detecteur = adapter_detecteur(detecteur, inverse)
      
        let classe = classifier_force(norm(detecteur))
        print(classe)
    }
  
    return normalize(detecteur)
}

fn main() {
    let detecteur = random(dim: 64, mean: 0.0, std: 0.1)
  
    detecteur = entrainer_detecteur(
        detecteur_init: detecteur,
        signaux_positifs: 10,
        signaux_negatifs: 5
    )
  
    print("Test de detection:")
    let test_signal = random(dim: 64, mean: 1.5, std: 0.2)
    let detected = detecter_pattern(detecteur, test_signal, seuil: 50.0)
  
    print("Pattern detecte:")
    print(detected)
}
```

---

## Projets Complets

### Projet Final 1 : Système d'Apprentissage Multi-Couches

```normil
// Couche 1: Encodage
@plastic(rate: 0.01, mode: "hebbian")
fn couche_encodage(entree: Vec, poids: Vec) -> Vec {
    let code = vec_mul(entree, poids)
    return normalize(code)
}

// Couche 2: Traitement
@atomic
@plastic(rate: 0.005, mode: "stdp")
fn couche_traitement(code: Vec, poids: Vec) -> Vec {
    let traite = vec_mul(code, poids)
    let n = norm(traite)
  
    if n > 5.0 {
        return normalize(traite)
    }
  
    return traite
}

// Couche 3: Sortie
@atomic
fn couche_sortie(traite: Vec, poids: Vec, seuil: float) -> str {
    let sortie = dot(traite, poids)
  
    match sortie {
        case float(s) where s > seuil * 2.0 -> {
            return "Classe A"
        }
        case float(s) where s > seuil -> {
            return "Classe B"
        }
        case _ -> {
            return "Classe C"
        }
    }
}

fn reseau_complet(
    entree: Vec,
    poids1: Vec,
    poids2: Vec,
    poids3: Vec
) -> str {
    let code = couche_encodage(entree, poids1)
    let traite = couche_traitement(code, poids2)
    let classe = couche_sortie(traite, poids3, seuil: 25.0)
    return classe
}

fn main() {
    let dim = 128
  
    let p1 = random(dim: dim, mean: 0.0, std: 0.1)
    let p2 = random(dim: dim, mean: 0.0, std: 0.1)
    let p3 = random(dim: dim, mean: 0.0, std: 0.1)
  
    print("Test 1:")
    let e1 = random(dim: dim, mean: 3.0, std: 0.5)
    print(reseau_complet(e1, p1, p2, p3))
  
    print("Test 2:")
    let e2 = random(dim: dim, mean: 1.0, std: 0.2)
    print(reseau_complet(e2, p1, p2, p3))
  
    print("Test 3:")
    let e3 = random(dim: dim, mean: 0.1, std: 0.05)
    print(reseau_complet(e3, p1, p2, p3))
}
```

---

## Exercices de Synthèse

### Exercice Avancé 1

Créez un système de reconnaissance de patterns avec:

- 3 types de patterns différents
- Apprentissage @plastic avec mode au choix
- Validation @atomic des résultats
- Classification par pattern matching

### Exercice Avancé 2

Implémentez une mémoire associative avec:

- Stockage de 5 patterns
- Rappel par similarité
- Consolidation progressive
- Oubli contrôlé

### Exercice Avancé 3

Développez un réseau compétitif avec:

- Plusieurs neurones en compétition
- Apprentissage winner-take-all
- Stabilisation @atomic
- Analyse des clusters formés

---

## Niveau 5 : Types O-RedMind (✨ Phase 5)

### Leçon 5.1 : EpisodicRecord - Mémoire Épisodique

**Objectif** : Stocker des événements bruts horodatés avec vecteurs multimodaux

```normil
fn main() {
    // Création d'un enregistrement épisodique
    let memory = EpisodicRecord {
        id: "event_001",
        timestamp: 1698000000.0,
        sources: ["camera", "audio"],
        vecs: {},
        summary: "User said hello",
        labels: [],
        trust: 0.95,
        provenance: {},
        outcome: "success"
    }
  
    // Accès aux champs
    print("Event ID: " + memory.id)
    print("Trust: " + to_string(memory.trust))
    print("Summary: " + memory.summary)
  
    // Modification
    memory.outcome = "completed"
    memory.trust = 0.98
}
```

**Cas d'usage** : Journalisation d'événements, traçabilité, analyse comportementale

---

### Leçon 5.2 : Concept - Mémoire Sémantique

**Objectif** : Représenter des concepts compressés avec confiance

```normil
fn main() {
    // Création d'un concept
    let ai_concept = Concept {
        concept_id: "ai_ml_001",
        centroid_vec: vec(128, [1.0, 0.5, -0.3, 0.8]),
        doc_count: 42,
        provenance_versions: ["v1.0", "v1.1"],
        trust_score: 0.85,
        labels: ["AI", "machine_learning", "neural_networks"]
    }
  
    // Accès et modification
    print("Concept: " + ai_concept.concept_id)
    print("Documents: " + to_string(ai_concept.doc_count))
    print("Trust: " + to_string(ai_concept.trust_score))
  
    // Mettre à jour après apprentissage
    ai_concept.doc_count = ai_concept.doc_count + 10
    ai_concept.trust_score = 0.90
}
```

**Cas d'usage** : Knowledge base, clustering sémantique, compression d'information

---

### Leçon 5.3 : ProtoInstinct - Instincts Prototypiques

**Objectif** : Définir des comportements instinctifs avec vecteurs de référence

```normil
fn main() {
    // Création d'un proto-instinct
    let safety_instinct = ProtoInstinct {
        id: "privacy_guard",
        vec_ref: vec(64, [0.8, 0.9, 0.7, 0.95]),
        rule: "if similarity > 0.9 then activate",
        weight: 1.5
    }
  
    // Utilisation dans une fonction
    fn should_activate(instinct: ProtoInstinct, threshold: float) -> bool {
        if instinct.weight > threshold {
            return true
        }
        return false
    }
  
    let active = should_activate(safety_instinct, 1.0)
    print("Instinct actif: " + to_string(active))
  
    // Ajustement dynamique
    safety_instinct.weight = 2.0
}
```

**Cas d'usage** : Systèmes de sécurité, comportements réactifs, priorités dynamiques

---

### Leçon 5.4 : SparseVec - Vecteurs Creux Optimisés

**Objectif** : Stocker efficacement des vecteurs avec beaucoup de zéros

```normil
fn main() {
    // Création d'un vecteur creux
    // Seulement 5 valeurs non-nulles sur 1000 dimensions
    let sparse = SparseVec {
        indices: [0, 100, 250, 500, 999],
        values: [1.5, 2.0, -0.5, 3.0, 0.8],
        dim: 1000
    }
  
    print("Dimension: " + to_string(sparse.dim))
    print("Non-zeros: " + to_string(len(sparse.indices)))
  
    // Calcul de sparsité
    fn sparsity(sv: SparseVec) -> float {
        let nnz = len(sv.indices)
        return (1.0 - (to_float(nnz) / to_float(sv.dim))) * 100.0
    }
  
    let sp = sparsity(sparse)
    print("Sparsité: " + to_string(sp) + "%")
  
    // Liste de vecteurs creux
    let sparse_list = [
        SparseVec {
            indices: [0, 1],
            values: [1.0, 2.0],
            dim: 100
        },
        SparseVec {
            indices: [50, 99],
            values: [0.5, 0.8],
            dim: 100
        }
    ]
  
    print("Nombre de vecteurs: " + to_string(len(sparse_list)))
}
```

**Cas d'usage** : NLP (word embeddings), réseaux de neurones creux, économie mémoire

---

### Leçon 5.5 : Combinaison des Types O-RedMind

**Objectif** : Utiliser tous les types ensemble pour un système complet

```normil
// Système de mémoire intelligent
fn systeme_memoire() {
    // Mémoire épisodique
    let events = [
        EpisodicRecord {
            id: "e001",
            timestamp: 1698000000.0,
            sources: ["sensor"],
            vecs: {},
            summary: "Temperature spike detected",
            labels: [],
            trust: 0.9,
            provenance: {},
            outcome: "analyzed"
        }
    ]
  
    // Concepts appris
    let concepts = [
        Concept {
            concept_id: "temperature_anomaly",
            centroid_vec: vec(64, [0.9, 0.8, 0.7, 0.95]),
            doc_count: 15,
            provenance_versions: ["v1"],
            trust_score: 0.88,
            labels: ["anomaly", "temperature"]
        }
    ]
  
    // Instincts de sécurité
    let instincts = [
        ProtoInstinct {
            id: "alert_system",
            vec_ref: vec(64, [0.85, 0.9, 0.75, 0.92]),
            rule: "if trust > 0.85 then alert",
            weight: 2.0
        }
    ]
  
    // Représentation creuse
    let feature_vec = SparseVec {
        indices: [5, 12, 28, 45],
        values: [1.0, 0.8, 0.6, 0.9],
        dim: 64
    }
  
    print("Système de mémoire initialisé")
    print("Events: " + to_string(len(events)))
    print("Concepts: " + to_string(len(concepts)))
    print("Instincts: " + to_string(len(instincts)))
    print("Features sparsity: " + to_string(len(feature_vec.indices)) + "/" + to_string(feature_vec.dim))
}

fn main() {
    systeme_memoire()
}
```

**Cas d'usage** : Agents intelligents, systèmes de mémoire hiérarchique, IA contextuelle

---

### Exemples Complets Phase 5

Consultez les fichiers d'exemples dans `examples/` :

- `test_episodic_record.nor` - Tous les cas d'usage EpisodicRecord
- `test_concept_simple.nor` - Manipulation de Concepts
- `test_protoinstinct_simple.nor` - Gestion d'instincts
- `test_sparsevec_simple.nor` - Vecteurs creux optimisés

---

## Niveau 6 : Primitives Neurales & Transactions

La Phase 6 introduit des primitives neurales avancées et un système de transactions avec audit automatique.

### Leçon 6.1 : Low-Rank Update

La primitive `lowrankupdate()` permet de mettre à jour une matrice de manière efficace avec un produit extérieur de rang 1.

**Formule** : W' = W + u ⊗ v

```normil
// Mise à jour de rang faible
let W = [[1.0, 0.0], [0.0, 1.0]]  // Matrice identité
let u = vec(2, [1.0, 0.0])
let v = vec(2, [0.0, 1.0])

// Ajouter u⊗v à W
let W_new = lowrankupdate(W, u, v)
// Résultat: [[1.0, 1.0], [0.0, 1.0]]
```

**Cas d'usage** :

- Adaptation de poids neuronaux sans ré-entraînement complet
- Apprentissage incrémental
- Mise à jour de modèles avec faible coût computationnel

---

### Leçon 6.2 : Quantization

La primitive `quantize()` compresse un vecteur en réduisant sa précision à 8 ou 4 bits.

```normil
// Quantisation pour économie mémoire
let v = random(128, 0.0, 1.0)

// Quantisation 8 bits (haute précision)
let v_q8 = quantize(v, 8)

// Quantisation 4 bits (haute compression)
let v_q4 = quantize(v, 4)

// La dimension est préservée
print(v.dim)      // 128
print(v_q8.dim)   // 128
print(v_q4.dim)   // 128
```

**Comparaison** :

- **8-bit** : ~1% d'erreur, 50% de compression
- **4-bit** : ~5% d'erreur, 75% de compression

**Cas d'usage** :

- Stockage de vecteurs en production
- Transmission réseau optimisée
- Systèmes embarqués avec mémoire limitée

---

### Leçon 6.3 : Online Clustering

La primitive `onlinecluster_update()` met à jour un centroïde de manière incrémentale.

**Formule** : c' = (1 - lr) × c + lr × x

```normil
// Clustering en ligne
let centroid = zeros(64)
let lr = 0.1  // Learning rate

// Ajouter progressivement des points
let x1 = random(64, 0.0, 1.0)
centroid = onlinecluster_update(centroid, x1, lr)

let x2 = random(64, 0.0, 1.0)
centroid = onlinecluster_update(centroid, x2, lr)

let x3 = random(64, 0.0, 1.0)
centroid = onlinecluster_update(centroid, x3, lr)

// Le centroïde converge vers la moyenne des points
```

**Paramètre learning rate** :

- `lr = 0.0` : Aucun changement
- `lr = 0.1` : Adaptation lente, stable
- `lr = 0.5` : Adaptation moyenne
- `lr = 1.0` : Remplacement complet

**Cas d'usage** :

- Consolidation sémantique en temps réel
- Clustering sans stocker tous les points
- Adaptation continue de concepts

---

### Leçon 6.4 : Système de Transactions

Les transactions garantissent la traçabilité et l'intégrité des opérations critiques avec audit logging automatique.

```normil
// Déclaration d'une transaction
transaction append_episode_safe(summary: str, trust: float) -> str {
    // Créer un enregistrement épisodique
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
  
    // Cette opération est automatiquement loggée
    let id = episodic_append(record)
  
    return id
}

// Appel de la transaction
let episode_id = append_episode_safe("Important event", 0.95)
```

**Avantages** :

- ✅ **Audit automatique** : Chaque transaction est loggée (start/success/failed)
- ✅ **Traçabilité** : Horodatage et paramètres enregistrés
- ✅ **Rollback** : En cas d'erreur, état restauré automatiquement
- ✅ **Intégrité** : Hash chaining pour vérification

**Transaction avec rollback** :

```normil
transaction update_concept(concept_id: str, new_vec: Vec) {
    let old = semantic_query(concept_id, k: 1)[0]
    semantic_upsert(concept_id, new_vec)
    audit_log("concept_updated", concept_id)
} rollback {
    // En cas d'erreur, restaurer l'ancien vecteur
    semantic_upsert(concept_id, old.centroid_vec)
}
```

---

### Leçon 6.5 : Exemple Complet - Système d'Apprentissage

Combinons toutes les primitives neurales dans un système complet :

```normil
// Système d'apprentissage incrémental avec transactions
transaction learn_from_experience(input: Vec, label: str, trust: float) -> str {
    // 1. Quantifier pour économie mémoire
    let input_q = quantize(input, 8)
  
    // 2. Chercher le concept le plus proche
    let similar = semantic_query(input_q, k: 1)
  
    let concept_id = ""
  
    if len(similar) > 0 {
        // Concept existant : mise à jour incrémentale
        let existing = similar[0]
        concept_id = existing.concept_id
      
        // Mettre à jour le centroïde
        let new_centroid = onlinecluster_update(
            existing.centroid_vec,
            input_q,
            0.1
        )
      
        // Low-rank update pour affiner
        let u = input_q
        let v = existing.centroid_vec
        let refined = vec(input.dim, [0.0])  // Placeholder pour matrice
      
        // Sauvegarder le concept mis à jour
        let updated = Concept {
            concept_id: concept_id,
            centroid_vec: new_centroid,
            doc_count: existing.doc_count + 1,
            provenance_versions: existing.provenance_versions,
            trust_score: (existing.trust_score + trust) / 2.0,
            labels: existing.labels + [label]
        }
      
        semantic_upsert(updated)
      
    } else {
        // Nouveau concept
        concept_id = generate_uuid()
      
        let new_concept = Concept {
            concept_id: concept_id,
            centroid_vec: input_q,
            doc_count: 1,
            provenance_versions: [],
            trust_score: trust,
            labels: [label]
        }
      
        semantic_upsert(new_concept)
    }
  
    // 3. Enregistrer l'épisode
    let record = EpisodicRecord {
        id: generate_uuid(),
        timestamp: now(),
        sources: ["learning_system"],
        vecs: {"input": input_q},
        summary: "Learned: " + label,
        labels: [{"label": label, "score": trust}],
        trust: trust,
        provenance: {"device_id": "learner", "signature": ""},
        outcome: "learned"
    }
  
    episodic_append(record)
  
    return concept_id
}

// Utilisation
let v1 = random(128, 0.0, 1.0)
let c1 = learn_from_experience(v1, "concept_A", 0.9)

let v2 = random(128, 0.0, 1.0)
let c2 = learn_from_experience(v2, "concept_A", 0.85)

print("Learned concept: " + c1)
```

**Ce système** :

- ✅ Quantifie les entrées (économie mémoire)
- ✅ Clustering incrémental (pas de ré-entraînement)
- ✅ Low-rank updates (adaptation fine)
- ✅ Transactions auditées (traçabilité complète)
- ✅ Mémoire épisodique + sémantique (O-RedMind complet)

---

## Conclusion

Vous maîtrisez maintenant :
✅ Les bases de NORMiL (variables, fonctions, conditions, boucles)
✅ **Inférence de types** automatique (Phase 3.1)
✅ **Opérations sur chaînes** et concaténation (Phase 3.3)
✅ Les vecteurs et opérations vectorielles
✅ **Système de modules** et imports (Phase 3.2)
✅ **Interopérabilité Python complète** (Phase 4) :

- Import de modules Python (Phase 4.1)
- Appel de fonctions Python (Phase 4.2)
- Conversions de types automatiques (Phase 4.3)
- Accès aux objets, classes et méthodes Python (Phase 4.4)
  ✅ **Types O-RedMind spécialisés** (Phase 5) :
- EpisodicRecord : mémoire épisodique horodatée
- Concept : mémoire sémantique compressée
- ProtoInstinct : comportements instinctifs
- SparseVec : vecteurs creux optimisés
  ✅ **Primitives neurales & transactions** (Phase 6) :
- lowrankupdate() : Mise à jour de rang faible W' = W + u⊗v
- quantize() : Quantisation 8/4 bits pour compression
- onlinecluster_update() : Clustering incrémental
- transaction : Système avec audit logging automatique

---

## Niveau 7 : Plasticité Neuronale Avancée

### Leçon 7.1 - Annotation @plastic avec Stabilité

L'annotation `@plastic` peut maintenant détecter automatiquement quand une fonction atteint la stabilité :

```normil
@plastic(rate: 0.01, mode: "hebbian", stability_threshold: 0.01)
fn learn_pattern(input: Vec, target: Vec) -> Vec {
    // La plasticité s'arrête automatiquement quand stable
    let weights = hebbian_update(input, target, 0.01)
    return weights
}
```

**Paramètres** :

- `rate` : Taux d'apprentissage initial (décroît automatiquement)
- `mode` : Type de plasticité (`"hebbian"`, `"stdp"`, `"anti_hebbian"`)
- `stability_threshold` : Seuil de convergence (défaut: 0.01 = 1%)

**Métadonnées automatiques** :

- `step_count` : Nombre d'appels à la fonction
- `is_stable` : True quand la stabilité est atteinte
- Learning rate décroît automatiquement jusqu'à stabilité

### Leçon 7.2 - Modes de Plasticité

Trois modes implémentés avec normalisation automatique :

```normil
// Mode Hebbian : Renforcement corrélé
@plastic(rate: 0.005, mode: "hebbian")
fn hebbian_learn(pre: Vec, post: Vec) -> Vec {
    let weights = outer_product(pre, post)
    return weights  // Automatiquement normalisé (norme L2 = 1.0)
}

// Mode STDP : Spike-Timing Dependent Plasticity
@plastic(rate: 0.01, mode: "stdp")
fn stdp_learn(spike_train: Vec, timing: Vec) -> Vec {
    let weights = time_dependent_update(spike_train, timing)
    return weights  // Auto-normalisé
}

// Mode Anti-Hebbian : Décorrélation
@plastic(rate: 0.003, mode: "anti_hebbian")
fn anti_hebbian_learn(pattern: Vec) -> Vec {
    let weights = decorrelate(pattern)
    return weights  // Auto-normalisé
}
```

**Caractéristiques communes** :

- ✅ Normalisation L2 automatique des résultats Vec
- ✅ Decay du learning rate quand non-stable
- ✅ Détection de convergence automatique

### Leçon 7.3 - Primitives de Gestion de Plasticité

#### normalize_plasticity()

Normalise un vecteur à norme L2 = 1.0 :

```normil
let weights = vec(3, [3.0, 4.0, 0.0])
let normalized = normalize_plasticity(weights)
// normalized = [0.6, 0.8, 0.0], norme = 1.0

print("Norme: " + to_string(norm(normalized)))  // 1.0
```

**Utilisation** : Maintenir la magnitude constante pendant l'apprentissage.

#### decay_learning_rate()

Décroissance exponentielle du taux d'apprentissage :

```normil
let lr = 0.1
let factor = 0.95  // Décroissance de 5% par étape

lr = decay_learning_rate(lr, factor)
// lr = 0.095

// Après 10 étapes
for i in range(10) {
    lr = decay_learning_rate(lr, 0.95)
}
// lr ≈ 0.0599
```

**Utilisation** : Convergence progressive vers un optimum.

#### compute_stability()

Vérifie si deux vecteurs sont stables (changement relatif < seuil) :

```normil
let w_old = vec(3, [1.0, 2.0, 3.0])
let w_new = vec(3, [1.001, 2.002, 3.001])

let is_stable = compute_stability(w_old, w_new, 0.01)
// is_stable = true (changement < 1%)

let w_diff = vec(3, [1.5, 3.0, 4.5])
let is_unstable = compute_stability(w_old, w_diff, 0.01)
// is_unstable = false (changement ≈ 50%)
```

**Utilisation** : Critère d'arrêt pour l'apprentissage.

### Leçon 7.4 - Gestion Automatique de la Plasticité

Les fonctions `@plastic` bénéficient d'une gestion automatique complète :

```normil
@plastic(rate: 0.1, mode: "hebbian", stability_threshold: 0.005)
fn adaptive_network(input: Vec) -> Vec {
    // Variables "weights", "w", "synapses" ou "connections" 
    // sont automatiquement trackées
    let weights = random_vec(input.dim)
  
    // Traitement
    weights = onlinecluster_update(weights, input, 0.1)
  
    return weights
    // À chaque appel :
    // 1. step_count++
    // 2. Vérification stabilité (si weights capturés)
    // 3. Normalisation automatique (mode hebbian/stdp/anti_hebbian)
    // 4. Decay du LR (si non-stable)
}

// Utilisation
let data = vec(10, [0.5, 0.3, ...])
let learned1 = adaptive_network(data)  // step 1, LR=0.1
let learned2 = adaptive_network(data)  // step 2, LR≈0.099
let learned3 = adaptive_network(data)  // step 3, LR≈0.098
// ... convergence automatique
```

**Bénéfices** :

- ✅ Zéro code boilerplate pour la plasticité
- ✅ Convergence garantie (via decay + stabilité)
- ✅ Poids toujours normalisés
- ✅ Traçabilité complète (step_count, is_stable)

### Leçon 7.5 - Scénario Complet : Apprentissage Multi-Couches

```normil
@plastic(rate: 0.05, mode: "hebbian", stability_threshold: 0.01)
fn layer1(input: Vec) -> Vec {
    let w = zeros(input.dim)
    w = onlinecluster_update(w, input, 0.05)
    return w  // Auto-normalisé
}

@plastic(rate: 0.03, mode: "stdp", stability_threshold: 0.005)
fn layer2(hidden: Vec) -> Vec {
    let w = zeros(hidden.dim)
    w = onlinecluster_update(w, hidden, 0.03)
    return w  // Auto-normalisé
}

fn train_network(data: Vec) {
    // Couche 1
    let hidden = layer1(data)
    print("Hidden norm: " + to_string(norm(hidden)))  // ≈1.0
  
    // Couche 2
    let output = layer2(hidden)
    print("Output norm: " + to_string(norm(output)))  // ≈1.0
  
    // Chaque couche converge indépendamment
}

// Entraînement progressif
let training_data = vec(20, [...])
for epoch in range(100) {
    train_network(training_data)
    // Convergence automatique de chaque couche
}
```

**Résultat** :

- Chaque couche apprend son niveau de représentation
- Normalisation garantit la stabilité numérique
- Convergence détectée automatiquement
- Pas de réglage manuel des hyperparamètres

### Leçon 7.6 - Combinaison avec Transactions

Plasticité + Transactions = Apprentissage traçable :

```normil
@atomic
@plastic(rate: 0.02, mode: "hebbian")
fn safe_learn(pattern: Vec, label: string) -> Vec {
    transaction {
        audit("Learning pattern: " + label)
      
        let weights = zeros(pattern.dim)
        weights = onlinecluster_update(weights, pattern, 0.02)
      
        audit("Weights norm: " + to_string(norm(weights)))
      
        return weights  // Auto-normalisé + logged
    }
}

// En cas d'erreur, rollback automatique
// Chaque étape d'apprentissage est tracée
```

**Avantages** :

- 🔍 Traçabilité complète de l'apprentissage
- 🔄 Rollback en cas de problème
- 📊 Audit logging automatique
- ✅ Convergence garantie

### Leçon 7.7 - Modes de Plasticité Personnalisés

Créez vos propres modes d'apprentissage au-delà de hebbian/stdp/anti_hebbian :

```normil
// Enregistrer un nouveau mode
let oja_registered = register_plasticity_mode(
    "oja",           // Nom du mode
    true,            // Auto-normaliser ?
    "Oja's learning rule"  // Description
)

// Utiliser le mode personnalisé
@plastic(rate: 0.01, mode: "oja")
fn oja_network(input: Vec) -> Vec {
    let w = zeros(input.dim)
    w = onlinecluster_update(w, input, 0.01)
    return w  // Normalisé automatiquement car normalize=true
}

// Lister tous les modes disponibles
let all_modes = list_plasticity_modes()
// ["hebbian", "stdp", "anti_hebbian", "oja", ...]
print("Available modes: " + to_string(len(all_modes)))
```

**Cas d'usage** :

- Implémenter des règles d'apprentissage spécifiques
- Contrôler finement la normalisation
- Organiser des expériences comparatives

### Leçon 7.8 - Decay Factor Configurable

Contrôlez la vitesse de décroissance du learning rate :

```normil
// Decay rapide (convergence rapide mais moins précise)
@plastic(rate: 0.1, decay_factor: 0.90)
fn fast_learner(data: Vec) -> Vec {
    let w = zeros(data.dim)
    w = onlinecluster_update(w, data, 0.1)
    return w
    // LR décroît vite : 0.1 → 0.09 → 0.081 → ...
}

// Decay lent (convergence lente mais très précise)
@plastic(rate: 0.1, decay_factor: 0.995)
fn precise_learner(data: Vec) -> Vec {
    let w = zeros(data.dim)
    w = onlinecluster_update(w, data, 0.1)
    return w
    // LR décroît lentement : 0.1 → 0.0995 → 0.099 → ...
}

// Pas de decay (LR constant)
@plastic(rate: 0.1, decay_factor: 1.0)
fn constant_learner(data: Vec) -> Vec {
    let w = zeros(data.dim)
    w = onlinecluster_update(w, data, 0.1)
    return w
    // LR reste à 0.1
}
```

**Stratégies** :

- `0.90-0.95` : Apprentissage rapide, exploration large
- `0.95-0.99` : Équilibre (défaut: 0.99)
- `0.99-0.999` : Convergence fine, précision maximale
- `1.0` : LR constant (pas de decay)

### Leçon 7.9 - Multi-Critères de Stabilité

Détection avancée de convergence avec plusieurs critères :

```normil
// Maintenir un historique des poids
let weight_history = []

for epoch in range(20) {
    let w = train_step(data)
    weight_history = weight_history + [w]
  
    // Critère 1: Stabilité sur fenêtre (tous les changements < seuil)
    let window_stable = compute_stability_window(weight_history, 0.01)
  
    // Critère 2: Variance faible
    let variance = compute_weight_variance(weight_history)
    let var_stable = variance < 0.001
  
    // Convergence si TOUS les critères sont satisfaits
    let converged = window_stable
    if converged {
        converged = var_stable
    }
  
    if converged {
        print("Convergence détectée à epoch " + to_string(epoch))
        break
    }
}
```

**Avantages** :

- Détection robuste (évite les faux positifs)
- Critères complémentaires (stabilité locale + globale)
- Arrêt précoce intelligent

### Leçon 7.10 - Scheduling du Learning Rate

Contrôle fin du LR avec différentes stratégies :

#### Warmup Linéaire

```normil
fn train_with_warmup(data: Vec, epochs: int) {
    let weights = zeros(data.dim)
    let warmup_steps = 10
    let target_lr = 0.01
  
    for epoch in range(epochs) {
        // Calculer LR avec warmup
        let current_lr = lr_warmup_linear(epoch, warmup_steps, target_lr)
      
        // Entraîner avec ce LR
        weights = onlinecluster_update(weights, data, current_lr)
      
        print("Epoch " + to_string(epoch) + ", LR: " + to_string(current_lr))
    }
}
// Epoch 0, LR: 0.0
// Epoch 5, LR: 0.005
// Epoch 10+, LR: 0.01
```

#### Cosine Annealing

```normil
fn train_with_cosine(data: Vec, total_epochs: int) {
    let weights = zeros(data.dim)
    let min_lr = 0.0001
    let max_lr = 0.01
  
    for epoch in range(total_epochs) {
        let current_lr = lr_cosine_annealing(epoch, total_epochs, min_lr, max_lr)
        weights = onlinecluster_update(weights, data, current_lr)
    }
    // LR décroît en cosinus: 0.01 → ... → 0.0001
}
```

#### Step Decay

```normil
fn train_with_steps(data: Vec, epochs: int) {
    let weights = zeros(data.dim)
    let initial_lr = 0.1
  
    for epoch in range(epochs) {
        // Diviser par 2 tous les 10 epochs
        let current_lr = lr_step_decay(epoch, initial_lr, 0.5, 10)
        weights = onlinecluster_update(weights, data, current_lr)
    }
    // Epochs 0-9: LR=0.1
    // Epochs 10-19: LR=0.05
    // Epochs 20-29: LR=0.025
}
```

#### Plateau Detection

```normil
fn train_with_plateau(data: Vec, epochs: int) {
    let weights = zeros(data.dim)
    let current_lr = 0.01
    let losses = []
  
    for epoch in range(epochs) {
        weights = onlinecluster_update(weights, data, current_lr)
      
        // Calculer loss
        let diff = data - weights
        let loss = dot(diff, diff)
        losses = losses + [loss]
      
        // Réduire LR si plateau
        let reduction_factor = lr_plateau_factor(losses, 3, 0.5, 0.01)
        current_lr = current_lr * reduction_factor
    }
    // LR réduit automatiquement si pas d'amélioration
}
```

#### Combinaison Warmup + Cosine

```normil
fn advanced_scheduling(data: Vec, total_epochs: int) {
    let weights = zeros(data.dim)
    let warmup_steps = 10
  
    for epoch in range(total_epochs) {
        let current_lr = 0.0
      
        // Phase 1: Warmup
        if epoch < warmup_steps {
            current_lr = lr_warmup_linear(epoch, warmup_steps, 0.01)
        }
      
        // Phase 2: Cosine annealing
        if epoch >= warmup_steps {
            let adjusted_epoch = epoch - warmup_steps
            let adjusted_total = total_epochs - warmup_steps
            current_lr = lr_cosine_annealing(adjusted_epoch, adjusted_total, 0.0001, 0.01)
        }
      
        weights = onlinecluster_update(weights, data, current_lr)
    }
}
```

**Stratégies recommandées** :

- **Warmup + Cosine** : Meilleure performance générale
- **Step Decay** : Simple et efficace pour réseaux profonds
- **Plateau Detection** : Adaptatif, idéal si incertitude sur durée
- **Cosine seul** : Bon compromis convergence/simplicité

---

## Conclusion

Félicitations ! Vous maîtrisez maintenant **NORMiL v0.7.0** avec :

✅ Les types de base et les opérations
✅ Les structures de contrôle
✅ Les fonctions et la récursion
✅ Les types O-RedMind avancés :

- EpisodicRecord : Mémoire épisodique avec vecteurs multiples
- Concept : Mémoire sémantique avec centroïdes
- ProtoInstinct : Comportements instinctifs avec règles
- SparseVec : Vecteurs creux optimisés
  ✅ Les primitives neurales (Phase 6) :
- lowrankupdate() : Mises à jour de rang faible
- quantize() : Quantisation 8/4 bits
- onlinecluster_update() : Clustering incrémental
- transaction : Système avec audit logging automatique
  ✅ La plasticité neuronale avancée (Phase 7) :
- @plastic avec détection de stabilité
- Modes hebbian, stdp, anti_hebbian
- Primitives normalize_plasticity, decay_learning_rate, compute_stability
- Gestion automatique complète
  ✅ Améliorations avancées de plasticité (Phase 7.6-7.9) :
- **Modes personnalisables** : register_plasticity_mode(), list_plasticity_modes()
- **Decay configurable** : decay_factor paramétrable (0.90-1.0)
- **Multi-critères de stabilité** : compute_stability_window(), compute_weight_variance()
- **Scheduling du learning rate** :
  * lr_warmup_linear() : Warmup linéaire
  * lr_cosine_annealing() : Décroissance cosinus
  * lr_step_decay() : Decay par paliers
  * lr_plateau_factor() : Détection de plateau
- **Opérations vectorielles** : +, -, * pour Vec
  ✅ Les arguments nommés
  ✅ Le pattern matching complet
  ✅ Les annotations @plastic et @atomic
  ✅ La combinaison de toutes les features
  ✅ La conception de systèmes complets

**Prochaines étapes** :

1. Explorez `examples/` pour plus d'inspiration :
   - `type_inference.nor` - Démonstration d'inférence
   - `imports_test.nor` - Utilisation de modules
   - `string_operations.nor` - Toutes les opérations string
   - `advanced_patterns.nor` - Pattern matching avancé
   - `neural_plasticity.nor` - Simulation complète
   - `python_interop.nor` - Exemples d'intégration Python (modules, fonctions)
   - `python_objects.nor` - Utilisation d'objets Python (classes, méthodes)
   - `test_episodic_record.nor` - Mémoire épisodique (Phase 5)
   - `test_concept_simple.nor` - Mémoire sémantique (Phase 5)
   - `test_protoinstinct_simple.nor` - Instincts (Phase 5)
   - `test_sparsevec_simple.nor` - Vecteurs creux (Phase 5)
   - `test_neural_primitives.nor` - Primitives neurales (Phase 6)
   - `test_transactions.nor` - Système de transactions (Phase 6)
   - `test_plasticity_primitives.nor` - Primitives de plasticité (Phase 7)
   - `test_advanced_plasticity.nor` - Gestion automatique plasticité (Phase 7)
   - `test_custom_plasticity_modes.nor` - Modes personnalisés (Phase 7.6)
   - `test_decay_factor.nor` - Decay configurable (Phase 7.7)
   - `test_multi_criteria_stability.nor` - Stabilité multi-critères (Phase 7.8)
   - `test_lr_scheduling.nor` - Scheduling du learning rate (Phase 7.9)
2. Créez vos propres modules réutilisables
3. Utilisez des bibliothèques Python (NumPy, SciPy, pandas, matplotlib, etc.)
4. Construisez des systèmes de mémoire avec les types O-RedMind
5. Appliquez les primitives neurales pour l'apprentissage incrémental
6. Utilisez les transactions pour la traçabilité critique
7. Exploitez la plasticité automatique pour l'apprentissage adaptatif
8. Consultez `API_REFERENCE.md` pour toutes les primitives
9. Consultez `PHASE2_FINAL_REPORT.md` pour les détails Phase 2
10. Contribuez au projet (voir `CONTRIBUTING.md`)

**Bon coding avec NORMiL ! 🚀**
