# Phase 6 - Primitives Neurales & Transactions

## Vue d'ensemble

La Phase 6 enrichit NORMiL avec des primitives neurales avancées et un système de transactions pour O-RedMind.

### Composants implémentés

1. **lowrankupdate()** - Mise à jour matricielle de rang faible
2. **quantize()** - Quantisation 8/4 bits pour compression
3. **onlinecluster_update()** - Clustering incrémental
4. **Système de transactions** - Transactions avec audit logging

---

## 1. lowrankupdate(W, u, v) -> W'

### Description
Effectue une mise à jour de rang faible sur une matrice : `W' = W + u ⊗ v`

### Utilisation
```normil
let W = [[1.0, 0.0], [0.0, 1.0]]
let u = vec(2, [1.0, 0.0])
let v = vec(2, [0.0, 1.0])
let W_new = lowrankupdate(W, u, v)
// W_new = [[1.0, 1.0], [0.0, 1.0]]
```

### Cas d'usage
- Adaptation de poids neuronaux sans ré-entraînement complet
- Apprentissage incrémental avec faible coût computationnel
- Mise à jour de matrices de connexions synaptiques

### Tests
- 5 tests pytest validés
- Tests : 2x2, 3x3, mises à jour multiples, préservation de forme

---

## 2. quantize(vec, bits) -> Vec

### Description
Quantise un vecteur sur n bits (8 ou 4) pour réduire la consommation mémoire.

### Utilisation
```normil
let v = random(128, 0.0, 1.0)
let v_q8 = quantize(v, 8)  // Quantisation 8-bit : haute précision
let v_q4 = quantize(v, 4)  // Quantisation 4-bit : haute compression
```

### Algorithme
1. Normalisation : `(data - min) / (max - min)` → [0, 1]
2. Quantisation : `round(normalized * (2^bits - 1)) / (2^bits - 1)`
3. Dé-normalisation : `quantized * (max - min) + min`

### Cas d'usage
- Compression de vecteurs pour stockage long terme
- Réduction de l'empreinte mémoire (jusqu'à 75% avec 4-bit)
- Trade-off précision/mémoire configurable

### Tests
- 6 tests pytest validés
- Tests : 8-bit, 4-bit, préservation dimension/range, vecteurs constants

---

## 3. onlinecluster_update(centroid, x, lr) -> Vec

### Description
Met à jour un centroïde de manière incrémentale : `c' = (1 - lr) × c + lr × x`

### Utilisation
```normil
let centroid = zeros(64)
let x1 = random(64, 0.0, 1.0)
let x2 = random(64, 0.0, 1.0)

let c = onlinecluster_update(centroid, x1, 0.3)  // Ajouter x1
c = onlinecluster_update(c, x2, 0.3)              // Ajouter x2
// Le centroïde converge progressivement vers la moyenne
```

### Paramètres
- `centroid` : Centroïde actuel
- `x` : Nouveau point à intégrer
- `lr` : Learning rate [0, 1] (vitesse d'adaptation)

### Cas d'usage
- Consolidation sémantique en temps réel
- Clustering sans garder tous les points en mémoire
- Mise à jour incrémentale de concepts

### Tests
- 8 tests pytest validés
- Tests : update basique, learning rates extrêmes, convergence, erreurs

---

## 4. Système de Transactions

### Description
Blocs de code avec audit logging automatique et support rollback.

### Syntaxe
```normil
transaction nom(params) -> Type {
    // Corps de la transaction
    return value
} rollback {
    // Optionnel : code de rollback en cas d'erreur
}
```

### Exemple complet
```normil
transaction append_episode_safe(summary: str, trust: float) -> str {
    let v = random(128, 0.0, 1.0)
    let record = EpisodicRecord {
        id: generate_uuid(),
        timestamp: now(),
        sources: ["test"],
        vecs: {"default": v},
        summary: summary,
        labels: [],
        trust: trust,
        provenance: {"device_id": "device1", "signature": ""},
        outcome: "pending"
    }
    
    let id = episodic_append(record)
    return id
}

// Appel de la transaction
let ep_id = append_episode_safe("Test episode", 0.9)
```

### Fonctionnalités
- **Audit automatique** : Logs de start/success/failed générés automatiquement
- **Snapshot/Restore** : État du scope sauvegardé avant exécution
- **Rollback** : En cas d'erreur, restauration automatique + exécution du bloc rollback
- **Traçabilité** : Timestamp, paramètres, durée, résultat enregistrés

### Cas d'usage
- Opérations critiques nécessitant traçabilité
- Gestion de la mémoire épisodique/sémantique
- Intégrité des données avec rollback automatique

### Tests
- 6 tests NORMiL validés
- Tests : transaction basique, calculs, multi-steps, erreurs, updates, chaînage

---

## Statistiques Phase 6

### Code
- **1 primitive ajoutée** : `onlinecluster_update()` (40 lignes)
- **1 méthode de parsing** : `parse_transaction_decl()` (75 lignes)
- **1 méthode d'exécution** : `call_transaction()` (110 lignes)
- **Total** : ~225 lignes de code Python

### Tests
- **25 tests pytest** (test_neural_primitives.py)
  - 5 tests lowrankupdate
  - 6 tests quantize
  - 8 tests onlinecluster_update
  - 3 tests scénarios combinés
  - 3 tests stabilité numérique
- **6 tests NORMiL** (test_transactions.nor)
- **203/203 tests de la suite complète** passent (100%)

### Documentation
- **TUTORIAL.md** : Niveau 6 ajouté (180+ lignes)
- **API_REFERENCE.md** : Section Primitives Neurales (200+ lignes)
- **README.md** : Roadmap mise à jour

### Fichiers créés
- `examples/test_neural_primitives.nor` (220 lignes)
- `examples/test_transactions.nor` (180 lignes)
- `tests/test_neural_primitives.py` (360 lignes)
- `docs/PHASE_6_SUMMARY.md` (ce fichier)

---

## Impact sur O-RedMind

### Mémoire épisodique
- Transactions garantissent l'intégrité des enregistrements
- Audit trail complet pour traçabilité

### Mémoire sémantique
- Clustering incrémental pour consolidation progressive
- Quantisation pour stockage long terme efficient

### Apprentissage
- Low-rank updates pour adaptation légère
- Learning rate configurable pour convergence contrôlée

---

## Prochaines étapes (Phase 7)

### Annotations de plasticité avancées
- @plastic enrichi avec stability_threshold
- @hebbian, @stdp, @anti_hebbian modes
- Gestion automatique de la plasticité synaptique

### Primitives supplémentaires
- normalize_plasticity(weights)
- decay_learning_rate(lr, factor)
- compute_stability(weights, threshold)

---

## Références

- TUTORIAL.md : Niveau 6 (lignes 1590-1770)
- API_REFERENCE.md : Primitives Neurales (lignes 2030-2230)
- README.md : Phase 6 roadmap (lignes 93-99)
- tests/test_neural_primitives.py : Suite de tests complète
- examples/test_neural_primitives.nor : Tests NORMiL
- examples/test_transactions.nor : Tests transactions
