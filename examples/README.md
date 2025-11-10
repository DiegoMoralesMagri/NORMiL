# O-RedMind Examples

**Date :** Novembre 2025
**Auteur :** Diego Morales Magri

---

**Exemples complets d'implémentation O-RedMind en NORMiL**

Ce répertoire contient 5 exemples fonctionnels démontrant comment implémenter l'architecture O-RedMind en NORMiL.

---

## 📚 Liste des Exemples

### 1. Perception Pipeline (`oredmind_perception.nor`)

**Démontre** :

- Capture multimodale (caméra + microphone)
- Encodage image/audio en vecteurs (`embed_image`, `embed_audio`)
- Fusion temporelle (`temporal_align`)
- Stockage en mémoire épisodique
- Indexation vectorielle (HNSW)
- Transactions atomiques (`@atomic`)

**Usage** :

```bash
normil run examples/oredmind_perception.nor
```

**Output** :

```
=== O-RedMind Perception Pipeline ===
Camera: 640x480 @ 10fps
Microphone: 16000Hz, 1 channels
Max frames: 10

--- Frame 0 ---
Captured image: 640x480 pixels
Captured audio: 8000 samples
Image vec dimension: 512
Audio vec dimension: 512
Fused vec dimension: 512
Fused vec norm: 16.234
Stored in episodic: a1b2c3d4...
Indexed vector (total entries: 1)
...
```

---

### 2. Hybrid Reasoner (`oredmind_reasoner.nor`)

**Démontre** :

- Meta-controller pour décision shortpass/longpass
- Neural shortpass (inférence rapide)
- Neural longpass (reasoning profond avec retrieval)
- Matching symbolique avec règles instinct
- Scoring de prototypes instinct
- Plasticité contrôlée (`@plastic`)

**Usage** :

```bash
normil run examples/oredmind_reasoner.nor
```

**Output** :

```
=== O-RedMind Hybrid Reasoner ===
Input dimension: 512
Cost budget: 1.0
Latency target: 100ms

[Retrieval] Querying index...
[Retrieval] Found 2 candidates
[Instinct] Scoring 1 prototypes
[Instinct] Top prototype: curiosity (score: 0.342)

[Meta-Controller] Path selected: shortpass

[Shortpass] Using TinyNet model
[Shortpass] Confidence: 0.678
[Output] Dimension: 512
[Output] Norm: 12.456
...
```

---

### 3. Consolidation Worker (`oredmind_consolidation.nor`)

**Démontre** :

- Priority sampling (replay priorisé DQN-style)
- Distillation épisodique → sémantique
- Clustering de concepts
- Forgetting policy adaptatif
- Semantic store avec merge de concepts similaires

**Usage** :

```bash
normil run examples/oredmind_consolidation.nor
```

**Output** :

```
=== O-RedMind Consolidation Worker ===
Initial episodic records: 20
Initial semantic concepts: 0
Batch size: 5
Iterations: 5

--- Iteration 0 ---
[Priority Sampling] Sampling 5 from 20 episodes
[Distillation] Distilling 5 episodes into concept
[Distillation] Created concept: c1a2b3c4...
[Distillation] Labels: [test, consolidation]
[Distillation] Trust: 0.742
[Semantic] Inserting new concept
Processed: 5 episodes
Semantic concepts: 1
Forgotten: 3 total
Remaining episodic: 17
...
```

---

### 4. Safety Layer (`oredmind_safety.nor`)

**Démontre** :

- Guardrails déclaratifs pour actions dangereuses
- Système de consentement utilisateur
- Audit logging avec hash chaining (immutabilité)
- Vérification d'intégrité du log
- Protection contre actions non autorisées

**Usage** :

```bash
normil run examples/oredmind_safety.nor
```

**Output** :

```
=== O-RedMind Safety Layer Demo ===

=== Test 1: File Write (requires consent) ===
=== Executing Safe Action ===
Action: file_write
User: alice

[Guardrail] Checking action: file_write
[Guardrail] BLOCKED by no_io_without_consent
[Execute] Requesting user consent...
[Consent] Requesting consent from user alice
[Consent] Action: file_write
[Consent] Reason: Save user preferences
[Consent] GRANTED - token: token_abc123
[Execute] Consent granted - proceeding
[Audit] Appending entry: action_executed
[Audit] Log size: 1 entries

[Result] SUCCESS
...
```

---

### 5. Instinct Governance (`oredmind_instinct.nor`)

**Démontre** :

- Tests de validation en sandbox
- Création de validation manifest
- Signature cryptographique des overlays
- Packaging core + overlay
- Audit des changements d'instinct
- Gouvernance multi-validateurs

**Usage** :

```bash
normil run examples/oredmind_instinct.nor
```

**Output** :

```
=== O-RedMind Instinct Governance Demo ===

=== O-RedMind Instinct Governance ===
Core prototypes: 1
Core rules: 1
Overlay prototypes: 1
Overlay rules: 1
Tests to run: 3
Validators: [validator_1, validator_2, validator_3]

=== Running Validation Tests ===
[Sandbox] Running test: safety_check
[Sandbox] Type: safety
[Sandbox] Test safety_check PASSED

[Sandbox] Running test: perf_check
[Sandbox] Type: performance
[Sandbox] Test perf_check PASSED
...

[Governance] All tests PASSED

=== Final Result ===
✓ Package approved: pkg_xyz789
  Version: 1.1.0
  Core prototypes: 1
  Overlay prototypes: 1
...
```

---

## 🏗️ Architecture O-RedMind

Pour comprendre comment ces exemples s'intègrent dans l'architecture complète :

**Lire** : `docs/OREDMIND_ARCHITECTURE.md`

```
┌─────────────────────────────────────────────┐
│           O-RedMind Agent                   │
├─────────────────────────────────────────────┤
│  Perception → Reasoner → Action             │
│       ↓           ↓          ↓              │
│  Episodic ↔ Semantic ↔ Working Memory       │
│       ↓           ↓                          │
│    Consolidation Worker                     │
├─────────────────────────────────────────────┤
│  Safety Layer (Guardrails, Consent, Audit) │
├─────────────────────────────────────────────┤
│  Instinct Core (Prototypes + Rules)        │
└─────────────────────────────────────────────┘
```

---

## 📖 Documentation Complète

### Tutoriels

**Fichier** : `TUTORIAL.md`

Leçons 8.1 à 8.8 couvrent :

- Types O-RedMind
- Perception multimodale
- Index et retrieval
- Reasoner hybride
- Consolidation et replay
- Safety et gouvernance
- Instinct Core
- Pipeline complet

### Référence

- `SPECIFICATION.md` : Référence complète du langage NORMiL
- `PHASE8_OREDMIND.md` : Plan Phase 8 (types, primitives, modules)
- `OREDMIND_ARCHITECTURE.md` : Guide d'architecture complet

---

## 🚀 Démarrage Rapide

### 1. Installation

```bash
# Clone du repository
git clone https://github.com/your-org/normil.git
cd normil

# Installation dépendances Python
pip install -r requirements.txt
```

### 2. Exécution des Exemples

```bash
# Perception pipeline
normil run examples/oredmind_perception.nor

# Reasoner hybride
normil run examples/oredmind_reasoner.nor

# Consolidation
normil run examples/oredmind_consolidation.nor

# Safety layer
normil run examples/oredmind_safety.nor

# Instinct governance
normil run examples/oredmind_instinct.nor
```

### 3. Tests

```bash
# Tous les tests (416 tests)
pytest tests/

# Tests Phase 8.4 (DevTools)
pytest tests/test_devtools.py

# Avec coverage
pytest tests/ --cov=runtime --cov-report=html
```

---

## 💡 Patterns Recommandés

### Transactions Atomiques

```normil
@atomic {
    episodic = episodic_append(episodic, record)
    index = hnsw_insert(index, vec, metadata)
}
```

### Plasticité Contrôlée

```normil
@plastic(rate=0.001, mode="lowrank", decay=0.99)
fn adaptive_reasoner(input: Vec) -> Vec {
    ...
}
```

### Priority Replay

```normil
fn priority_fn(record: EpisodicRecord) -> Float {
    return 0.5 * novelty(record) + 0.5 * recency(record)
}

let sampled = priority_sample(episodes, k=100, priority_fn)
```

### Guardrails Déclaratifs

```normil
let GUARDRAILS = [
    SafetyGuardrail {
        id: "no_io_without_consent",
        condition: "io_operation",
        action_blocked: "file_write,network_send",
        require_consent: true,
        override_level: 10
    }
]
```

---

## ❓ FAQ

**Q : Puis-je utiliser ces exemples en production ?**

A : Ces exemples sont des **démonstrations éducatives**. Pour production :

- Ajoutez gestion d'erreurs robuste
- Implémentez persistance (sauvegardes disque)
- Configurez logging/monitoring
- Testez à grande échelle

**Q : Comment intégrer avec des modèles PyTorch/TensorFlow ?**

A : Utilisez l'interop Python de NORMiL :

```normil
import python.torch as torch

fn neural_inference(input: Vec) -> Vec {
    let model = torch.load("model.pth")
    let output = model.forward(input.data)
    return Vec.from_list(output)
}
```

**Q : Où sont les primitives comme `embed_image`, `fastindex_query` ?**

A : Elles sont implémentées dans `runtime/primitives.py` (Phase 8.2-8.4).
Vérifiez `PRIMITIVES` dict pour la liste complète.

---

## 📝 Contribution

Pour contribuer de nouveaux exemples :

1. Fork du repository
2. Créez votre exemple dans `examples/`
3. Ajoutez tests si applicable
4. Documentation dans ce README
5. Pull request avec description

**Style** :

- Suivre `STYLE_GUIDE.md`
- Commentaires détaillés
- Exemples auto-contenus
- Output explicite

---

## 📄 License

MIT License - voir `LICENSE` file

---

**Auteur** : Diego Morales Magri
**Date** : Novembre 2025
**Version** : NORMiL Phase 8.5 - Examples O-RedMind
