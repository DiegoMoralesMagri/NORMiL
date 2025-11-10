# O-RedMind Architecture en NORMiL

**Guide complet pour implémenter l'architecture O-RedMind en NORMiL**

**Date :** Novembre 2025
**Auteur :** Diego Morales Magri
**Version :** 1.0
**Status :** 📚 Documentation Complète

---

## Table des Matières

1. [Introduction](#introduction)
2. [Vue d&#39;Ensemble](#vue-densemble)
3. [Mapping INDICATIONS_TECHNIQUES → NORMiL](#mapping-indications_techniques--normil)
4. [Modules NORMiL pour O-RedMind](#modules-normil-pour-o-redmind)
5. [Patterns Recommandés](#patterns-recommandés)
6. [Anti-Patterns à Éviter](#anti-patterns-à-éviter)
7. [Exemples Complets](#exemples-complets)
8. [FAQ](#faq)

---

## Introduction

### Qu'est-ce qu'O-RedMind ?

**O-RedMind** est une architecture d'IA humanoïde basée sur :

- **Mémoire épisodique** : Stockage d'expériences temporelles
- **Mémoire sémantique** : Concepts consolidés et généralisés
- **Perception multimodale** : Caméra, microphone, capteurs
- **Reasoner hybride** : Neural (fast/slow) + Symbolique (règles)
- **Instinct Core** : Comportements innés + Overlays validés
- **Safety & Governance** : Guardrails, consentement, audit
- **Plasticité contrôlée** : Apprentissage online avec régulation

### Pourquoi NORMiL ?

NORMiL est conçu **spécifiquement pour O-RedMind** :

✅ **Types natifs** pour mémoire épisodique/sémantique
✅ **Plasticité** avec 6 modes d'apprentissage
✅ **Transactions atomiques** avec audit
✅ **Primitives optimisées** pour retrieval, consolidation, multimodal
✅ **Safety by design** : guardrails, consent, immutabilité

**NORMiL n'est PAS un langage généraliste** - il est le langage d'implémentation NATIF d'O-RedMind.

---

## Vue d'Ensemble

### Architecture O-RedMind

```
┌─────────────────────────────────────────────────────────┐
│                     O-RedMind Agent                      │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐       ┌──────────────┐               │
│  │  Perception  │──────▶│   Reasoner   │               │
│  │  Pipeline    │       │   Hybrid     │               │
│  └──────────────┘       └──────────────┘               │
│         │                       │                        │
│         ▼                       ▼                        │
│  ┌─────────────────────────────────────┐                │
│  │       Working Memory (Vec)          │                │
│  └─────────────────────────────────────┘                │
│         │                       │                        │
│         ▼                       ▼                        │
│  ┌──────────────┐       ┌──────────────┐               │
│  │  Episodic    │       │  Semantic    │               │
│  │  Memory      │◀─────▶│  Memory      │               │
│  └──────────────┘       └──────────────┘               │
│         │                       │                        │
│         └───────┬───────────────┘                        │
│                 ▼                                        │
│  ┌─────────────────────────────────────┐                │
│  │      Consolidation Worker           │                │
│  └─────────────────────────────────────┘                │
│                                                          │
├─────────────────────────────────────────────────────────┤
│                    Safety Layer                          │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │ Guardrails  │  │  Consent     │  │  Audit Log   │  │
│  └─────────────┘  └──────────────┘  └──────────────┘  │
├─────────────────────────────────────────────────────────┤
│                   Instinct Core                          │
│  ┌─────────────┐  ┌──────────────┐                     │
│  │ Core        │  │  Overlay     │                     │
│  │ Prototypes  │  │  (validated) │                     │
│  └─────────────┘  └──────────────┘                     │
└─────────────────────────────────────────────────────────┘
```

### Workflow Typique

1. **Perception** : Capture multimodale → Encodage → Fusion → Vec
2. **Working Memory** : Vec temporaire pour raisonnement
3. **Reasoner** : Décision shortpass/longpass → Neural + Symbolic
4. **Action** : Vérification Safety → Exécution → Audit
5. **Consolidation** : Replay priorité → Distillation → Oubli adaptatif

---

## Mapping INDICATIONS_TECHNIQUES → NORMiL

### Types de Données

| INDICATIONS_TECHNIQUES | NORMiL Type            | Implémenté | Description                                |
| ---------------------- | ---------------------- | ------------ | ------------------------------------------ |
| `Vec`                | `Vec`                | ✅ Phase 1   | Vecteur dense pour embeddings              |
| `SparseVec`          | `SparseVec`          | ✅ Phase 1   | Vecteur sparse pour optimisation mémoire  |
| `EpisodicRecord`     | `EpisodicRecord`     | ✅ Phase 1   | Record mémoire épisodique avec timestamp |
| `Concept`            | `Concept`            | ✅ Phase 1   | Concept sémantique avec centroïde        |
| `WorkingMemoryEntry` | `WorkingMemoryEntry` | ✅ Phase 1   | Entrée mémoire de travail (court terme)  |
| `ProtoInstinct`      | `ProtoInstinct`      | ✅ Phase 7   | Prototype instinct avec exemplaires        |
| `InstinctPackage`    | `InstinctPackage`    | ✅ Phase 8.1 | Package core + overlay                     |
| `SafetyGuardrail`    | `SafetyGuardrail`    | ✅ Phase 8.1 | Guardrail de sécurité                    |
| `AuditLogEntry`      | `AuditLogEntry`      | ✅ Phase 8.1 | Entrée log d'audit avec hash chaining     |
| `IndexEntry`         | `IndexEntry`         | ✅ Phase 8.1 | Entrée index vectoriel (HNSW)             |
| `Rule`               | `Rule`               | ✅ Phase 8.3 | Règle symbolique (condition → action)    |
| `ImageTensor`        | `ImageTensor`        | ✅ Phase 8.2 | Données image pour perception             |
| `AudioSegment`       | `AudioSegment`       | ✅ Phase 8.2 | Segment audio pour perception              |

### Primitives Critiques

#### Perception & Multimodal

```normil
// Encodage image → vecteur
fn embed_image(pixels: List<Float>, width: Int, height: Int) -> Vec

// Encodage audio → vecteur
fn embed_audio(samples: List<Float>, sample_rate: Int) -> Vec

// Alignement temporel de streams
fn temporal_align(vec1: Vec, vec2: Vec, t1: Float, t2: Float) -> Vec

// Fusion multimodale
fn fusion_concat(vecs: List<Vec>) -> Vec
fn cross_attention(vec_img: Vec, vec_audio: Vec) -> Vec
```

#### Index & Retrieval

```normil
// Top-k retrieval vectoriel
fn fastindex_query(index: FastIndex, query: Vec, k: Int) -> List<IndexEntry>

// Insert dans index HNSW
fn hnsw_insert(index: FastIndex, vec: Vec, metadata: Map<String, String>) -> FastIndex

// Re-ranking neural
fn rerank_neural(candidates: List<IndexEntry>, query: Vec) -> List<IndexEntry>
```

#### Reasoner Hybride

```normil
// Shortpass (fast inference)
fn neural_shortpass(input: Vec, model: NeuralModel, context: Vec) -> (Vec, Float)

// Longpass (deep reasoning)
fn neural_longpass(input: Vec, model: NeuralModel, retrieved: List<EpisodicRecord>) -> (Vec, TraceLog)

// Symbolic matching
fn symbolic_match(context: Map<String, Any>, rules: List<Rule>) -> List<Rule>

// Meta-controller decision
fn meta_controller_decide(input: Vec, cost_budget: Float, latency_target_ms: Int) -> ReasoningPath
```

#### Consolidation

```normil
// Priority sampling (replay)
fn priority_sample(episodes: List<EpisodicRecord>, k: Int, priority_fn: Fn(EpisodicRecord) -> Float) -> List<EpisodicRecord>

// Distillation épisodique → sémantique
fn distill_to_semantic(episodes: List<EpisodicRecord>) -> Concept

// Forgetting policy
fn forgetting_policy(memory: EpisodicRecord, age: Float, utility: Float, threshold: Float) -> Bool
```

#### Safety & Governance

```normil
// Vérification guardrail
fn check_guardrail(action: String, context: Map<String, String>, guardrails: List<SafetyGuardrail>) -> Result<(), GuardrailViolation>

// Requête consentement
fn require_consent(request: ConsentRequest, user: User) -> Result<ConsentToken, String>

// Audit logging avec hash chain
fn audit_append(log: AuditLog, entry: AuditLogEntry) -> AuditLog

// Vérification intégrité
fn verify_hash_chain(entries: List<AuditLogEntry>) -> Bool
```

#### Instinct

```normil
// Scoring prototypes
fn score_prototypes(input: Vec, prototypes: List<ProtoInstinct>) -> List<(String, Float)>

// Validation overlay
fn run_test_sandboxed(overlay: InstinctOverlay, test: ValidationTest) -> TestResult

// Signature cryptographique
fn sign_overlay(overlay: InstinctOverlay, manifest: ValidationManifest, private_key: String) -> SignedOverlay
```

---

## Modules NORMiL pour O-RedMind

### 1. Module `perception`

**Fichier** : `oredmind_perception.nor`

**Responsabilités** :

- Capture multimodale (caméra, micro, capteurs)
- Encodage en vecteurs
- Fusion temporelle
- Stockage dans mémoire épisodique

**Types utilisés** :

- `ImageFrame`, `AudioFrame`
- `Vec`, `EpisodicRecord`
- `FastIndex`, `IndexEntry`

**Primitives clés** :

- `embed_image()`, `embed_audio()`
- `temporal_align()`
- `episodic_append()`, `hnsw_insert()`

**Pattern** :

```normil
fn perception_loop(camera, mic, episodic, index) {
    while true {
        // 1. Capture
        let img = camera.capture()
        let audio = mic.capture(500ms)
      
        // 2. Encodage
        let vec_img = embed_image(img)
        let vec_audio = embed_audio(audio)
      
        // 3. Fusion
        let vec_combined = temporal_align(vec_img, vec_audio, img.timestamp, audio.timestamp)
      
        // 4. Stockage atomique
        @atomic {
            let record = EpisodicRecord.create("perception", vec_combined, 0.9)
            episodic = episodic_append(episodic, record)
            index = hnsw_insert(index, vec_combined, {"id": record.id})
        }
    }
}
```

---

### 2. Module `reasoner`

**Fichier** : `oredmind_reasoner.nor`

**Responsabilités** :

- Meta-controller pour shortpass/longpass
- Reasoner neural rapide
- Reasoner neural profond avec retrieval
- Matching symbolique avec règles

**Types utilisés** :

- `Vec`, `EpisodicRecord`
- `Rule`, `InstinctPackage`
- `NeuralModel`, `TraceLog`

**Primitives clés** :

- `meta_controller_decide()`
- `neural_shortpass()`, `neural_longpass()`
- `symbolic_match()`, `score_prototypes()`

**Pattern** :

```normil
@plastic(rate=0.001, mode="lowrank")
fn hybrid_reasoner(input, index, instinct_pkg) {
    // 1. Retrieval
    let candidates = fastindex_query(index, input, k=16)
  
    // 2. Meta-controller
    let path = meta_controller_decide(input, cost_budget=1.0, latency_target_ms=200)
  
    // 3. Reasoner selon path
    let output = if path == "shortpass" {
        neural_shortpass(input, TinyNet, context_from(candidates))
    } else {
        // Longpass : symbolic + deep neural
        let symbolic_hits = symbolic_match(context_map, instinct_pkg.core.rules)
        let (out, trace) = neural_longpass(input, DeepNet, candidates)
        audit_append(trace)
        out
    }
  
    return output
}
```

---

### 3. Module `consolidation`

**Fichier** : `oredmind_consolidation.nor`

**Responsabilités** :

- Replay priorisé (DQN-style)
- Distillation épisodique → sémantique
- Clustering de concepts
- Forgetting policy adaptatif

**Types utilisés** :

- `EpisodicRecord`, `Concept`
- `EpisodicStore`, `SemanticStore`
- `ConsolidationMetrics`

**Primitives clés** :

- `priority_sample()`
- `distill_to_semantic()`
- `semantic_upsert()`
- `forgetting_policy()`

**Pattern** :

```normil
fn consolidation_worker(episodic, semantic, schedule_interval_ms) {
    while true {
        sleep_ms(schedule_interval_ms)
      
        // 1. Priority replay
        let priority_fn = fn(r) { novelty(r) * 0.5 + recency(r) * 0.5 }
        let episodes = priority_sample(episodic.get_all(), k=100, priority_fn)
      
        // 2. Distillation
        let concept = distill_to_semantic(episodes)
      
        // 3. Upsert semantic
        @atomic {
            semantic = semantic_upsert(semantic, concept)
        }
      
        // 4. Forgetting
        for ep in old_episodes {
            if forgetting_policy(ep, age, utility, threshold=0.1) {
                episodic.remove(ep.id)
            }
        }
    }
}
```

---

### 4. Module `safety`

**Fichier** : `oredmind_safety.nor`

**Responsabilités** :

- Vérification guardrails avant actions
- Système de consentement utilisateur
- Audit logging avec hash chaining
- Vérification d'intégrité

**Types utilisés** :

- `SafetyGuardrail`, `ConsentRequest`, `ConsentToken`
- `AuditLog`, `AuditLogEntry`
- `Action`, `User`

**Primitives clés** :

- `check_guardrail()`
- `require_consent()`
- `audit_append()`, `verify_hash_chain()`

**Pattern** :

```normil
fn safe_action(action, user, audit_log, guardrails) {
    // 1. Check guardrails
    let violation = check_guardrail(action.type, context, guardrails)
  
    if violation.is_err() {
        // 2. Requête consentement si applicable
        if guardrail.require_consent {
            let consent = require_consent(ConsentRequest {...}, user)
          
            if consent.is_err() {
                // Audit refus
                audit_append(audit_log, AuditLogEntry {
                    event_type: "consent_denied",
                    ...
                })
                return Err("Consent denied")
            }
        } else {
            return Err("Guardrail violation")
        }
    }
  
    // 3. Execute action
    let result = execute_action(action)
  
    // 4. Audit success
    audit_append(audit_log, AuditLogEntry {
        event_type: "action_executed",
        ...
    })
  
    return result
}
```

---

### 5. Module `instinct`

**Fichier** : `oredmind_instinct.nor`

**Responsabilités** :

- Gestion des overlays instinct
- Validation sandbox des overlays
- Signature cryptographique
- Packaging core + overlay

**Types utilisés** :

- `InstinctPackage`, `InstinctCore`, `InstinctOverlay`
- `ValidationManifest`, `ValidationTest`, `TestResult`
- `SignedOverlay`

**Primitives clés** :

- `run_test_sandboxed()`
- `sign_overlay()`, `verify_overlay_signature()`
- `create_instinct_package()`

**Pattern** :

```normil
fn instinct_governance(core, overlay_candidate, tests, audit_log) {
    // 1. Sandbox tests
    let test_results = []
    for test in tests {
        test_results.push(run_test_sandboxed(overlay_candidate, test))
    }
  
    // 2. Vérifier tous tests passent
    if !all_passed(test_results) {
        return Err("Tests failed")
    }
  
    // 3. Metrics before/after
    let metrics_before = compute_metrics(core)
    let metrics_after = compute_metrics_with_overlay(core, overlay_candidate)
  
    // 4. Validation manifest
    let manifest = ValidationManifest {...}
  
    // 5. Signature overlay
    let signed_overlay = sign_overlay(overlay_candidate, manifest, private_key)
  
    // 6. Package final
    let package = create_instinct_package(core, signed_overlay, version="1.1.0")
  
    // 7. Audit
    audit_append(audit_log, AuditLogEntry {
        event_type: "instinct_overlay_validated",
        ...
    })
  
    return Ok(package)
}
```

---

## Patterns Recommandés

### ✅ Pattern 1 : Transactions Atomiques pour Cohérence Mémoire

**Problème** : Garantir cohérence entre mémoire épisodique et index vectoriel.

**Solution** :

```normil
@atomic {
    episodic = episodic_append(episodic, record)
    index = hnsw_insert(index, vec, metadata)
}
```

**Pourquoi** : Les deux opérations sont atomiques - soit toutes deux réussissent, soit aucune.

---

### ✅ Pattern 2 : Plasticité Contrôlée avec Decay

**Problème** : Apprentissage online sans catastrophic forgetting.

**Solution** :

```normil
@plastic(rate=0.001, mode="lowrank", decay=0.99, schedule_fn=cosine_schedule)
fn adaptive_reasoner(input) {
    // Learning rate décroit progressivement
    ...
}
```

**Pourquoi** : Decay progressif évite l'oubli brutal tout en permettant l'adaptation.

---

### ✅ Pattern 3 : Priority Replay pour Consolidation

**Problème** : Quels épisodes consolider en priorité ?

**Solution** :

```normil
fn priority_fn(record) {
    let novelty = compute_novelty(record)
    let recency = 1.0 / (now() - record.timestamp + 1.0)
    let reward = record.metadata["reward"]
  
    return 0.4 * novelty + 0.3 * recency + 0.3 * reward
}

let sampled = priority_sample(episodes, k=100, priority_fn)
```

**Pourquoi** : Maximise l'impact de la consolidation (inspiré DQN, PER).

---

### ✅ Pattern 4 : Guardrails Déclaratifs

**Problème** : Comment spécifier les contraintes de sécurité ?

**Solution** :

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

**Pourquoi** : Déclaratif = facile à auditer, modifier, valider.

---

### ✅ Pattern 5 : Meta-Controller pour Reasoner Adaptatif

**Problème** : Comment choisir entre shortpass et longpass ?

**Solution** :

```normil
fn meta_controller_decide(input, cost_budget, latency_target_ms) {
    let complexity = norm(input) / 10.0
    let complexity_threshold = 0.5
    let latency_threshold = 200
  
    if complexity < complexity_threshold and latency_target_ms < latency_threshold {
        return "shortpass"
    } else {
        return "longpass"
    }
}
```

**Pourquoi** : Adaptation dynamique selon complexité et contraintes.

---

## Anti-Patterns à Éviter

### ❌ Anti-Pattern 1 : Mémoire Épisodique Illimitée

**Problème** :

```normil
// MAL : Jamais d'oubli
fn perception_loop() {
    while true {
        episodic.append(record)  // Croissance infinie !
    }
}
```

**Solution** :

```normil
// BIEN : Forgetting policy
if len(episodic.records) > max_size or forgetting_policy(old_record) {
    episodic.remove(old_record)
}
```

---

### ❌ Anti-Pattern 2 : Actions Sans Guardrails

**Problème** :

```normil
// MAL : Exécution directe sans vérification
fn execute_action(action) {
    do_io_operation(action.path)  // Dangereux !
}
```

**Solution** :

```normil
// BIEN : Toujours vérifier guardrails
fn safe_action(action, user, guardrails) {
    let violation = check_guardrail(action, context, guardrails)
    if violation.is_err() {
        return Err("Blocked")
    }
    do_io_operation(action.path)
}
```

---

### ❌ Anti-Pattern 3 : Overlay Sans Validation

**Problème** :

```normil
// MAL : Application directe overlay communauté
let instinct_pkg = InstinctPackage {
    core: core,
    overlay: untrusted_overlay  // Pas de tests !
}
```

**Solution** :

```normil
// BIEN : Sandbox tests + signature
let test_results = run_tests_sandboxed(overlay, tests)
if all_passed(test_results) {
    let signed_overlay = sign_overlay(overlay, manifest, key)
    let package = create_package(core, signed_overlay)
}
```

---

### ❌ Anti-Pattern 4 : Audit Log Sans Hash Chaining

**Problème** :

```normil
// MAL : Logs mutables
let audit_log = []
audit_log.append(entry)  // Peut être modifié a posteriori
```

**Solution** :

```normil
// BIEN : Hash chaining pour immutabilité
fn audit_append(log, entry) {
    entry.prev_hash = hash(log.last_entry())
    return AuditLog {
        entries: log.entries + [entry],
        chain_valid: verify_hash_chain(log.entries + [entry])
    }
}
```

---

### ❌ Anti-Pattern 5 : Plasticité Sans Régulation

**Problème** :

```normil
// MAL : Learning rate constant
@plastic(rate=0.1, mode="full")  // Trop agressif !
fn train_model(input) {
    ...
}
```

**Solution** :

```normil
// BIEN : Decay + lowrank
@plastic(rate=0.001, mode="lowrank", decay=0.99)
fn adaptive_train(input) {
    ...
}
```

---

## Exemples Complets

### 1. Perception Pipeline

**Fichier** : `examples/oredmind_perception.nor`

**Démontre** :

- Capture multimodale (caméra + micro)
- Encodage image/audio en vecteurs
- Fusion temporelle
- Stockage atomique épisodique + index

**Usage** :

```bash
normil run examples/oredmind_perception.nor
```

---

### 2. Hybrid Reasoner

**Fichier** : `examples/oredmind_reasoner.nor`

**Démontre** :

- Meta-controller shortpass/longpass
- Neural shortpass (fast inference)
- Neural longpass (deep reasoning + retrieval)
- Symbolic matching avec règles instinct

**Usage** :

```bash
normil run examples/oredmind_reasoner.nor
```

---

### 3. Consolidation Worker

**Fichier** : `examples/oredmind_consolidation.nor`

**Démontre** :

- Priority sampling (replay)
- Distillation épisodique → sémantique
- Clustering de concepts
- Forgetting policy adaptatif

**Usage** :

```bash
normil run examples/oredmind_consolidation.nor
```

---

### 4. Safety Layer

**Fichier** : `examples/oredmind_safety.nor`

**Démontre** :

- Guardrails déclaratifs
- Système de consentement utilisateur
- Audit logging avec hash chaining
- Vérification d'intégrité

**Usage** :

```bash
normil run examples/oredmind_safety.nor
```

---

### 5. Instinct Governance

**Fichier** : `examples/oredmind_instinct.nor`

**Démontre** :

- Tests sandbox pour overlays
- Validation manifest
- Signature cryptographique
- Packaging core + overlay

**Usage** :

```bash
normil run examples/oredmind_instinct.nor
```

---

## FAQ

### Q1 : Pourquoi utiliser NORMiL plutôt que Python pour O-RedMind ?

**Réponse** :

NORMiL est conçu **spécifiquement** pour O-RedMind :

✅ **Types natifs** : `EpisodicRecord`, `Concept`, `ProtoInstinct` sont des citizens de première classe
✅ **Plasticité** : `@plastic` avec 6 modes (full, lowrank, sparse, etc.) - pas possible en Python standard
✅ **Transactions** : `@atomic` garantit cohérence mémoire
✅ **Safety** : Guardrails, audit logging, hash chaining intégrés
✅ **Performance** : Primitives optimisées en Rust/C++ (sous le capot)

Python reste utilisable pour :

- Prototypage rapide
- Scripts utilitaires
- Visualisation (matplotlib, etc.)

Mais l'**architecture O-RedMind en production** doit être en NORMiL.

---

### Q2 : Comment intégrer des modèles PyTorch/TensorFlow ?

**Réponse** :

NORMiL peut appeler des modèles Python via **interop** :

```normil
import python.torch as torch

fn neural_inference(input: Vec) -> Vec {
    // Charge modèle PyTorch
    let model = torch.load("model.pth")
  
    // Inférence
    let tensor = torch.tensor(input.data)
    let output_tensor = model.forward(tensor)
  
    // Conversion tensor → Vec
    return Vec.from_list(output_tensor.to_list())
}
```

**Recommandations** :

- Utilisez interop pour **inférence** seulement
- Entraînement complexe en Python → export ONNX → import NORMiL
- Primitives NORMiL (`lowrankupdate`, `quantize`) pour apprentissage online

---

### Q3 : Comment gérer la persistance (sauvegarder mémoire sur disque) ?

**Réponse** :

Utilisez les primitives de sérialisation :

```normil
import normil.io as io

// Sauvegarde mémoire épisodique
fn save_episodic(episodic: EpisodicStore, path: String) {
    let serialized = episodic.to_json()
    io.write_file(path, serialized)
}

// Chargement
fn load_episodic(path: String) -> EpisodicStore {
    let data = io.read_file(path)
    return EpisodicStore.from_json(data)
}
```

**Formats supportés** :

- JSON (human-readable, debug)
- MessagePack (compact, production)
- Protobuf (avec schéma)

---

### Q4 : Comment débugger un reasoner NORMiL ?

**Réponse** :

Utilisez les outils de debug :

```normil
// 1. Annotation @trace
@trace
fn reasoner(input: Vec) -> Vec {
    let output = neural_longpass(input, model, retrieved)
    return output
}

// 2. Récupération traces
let traces = get_execution_traces("reasoner")
for trace in traces {
    print(f"Step: {trace.step}, Latency: {trace.latency_ms}ms")
}

// 3. Introspection runtime
let info = introspect_type(output)
print(f"Output type: {info.type_name}")
print(f"Output norm: {info.metadata.norm}")

// 4. Visualisation
import normil.viz as viz
viz.log_metric("output_norm", norm(output), step)
viz.plot_metrics("output_norm", "run_001")
```

---

### Q5 : Peut-on modifier un InstinctCore en production ?

**Réponse** :

**NON** - le Core est **immuable** en production.

✅ **Modifications autorisées** :

- Ajouter un **Overlay** validé
- Rollback vers une version précédente
- Mise à jour majeure (nouveau package signé)

❌ **Modifications interdites** :

- Modification directe du Core
- Overlay non validé/non signé
- Contournement des tests sandbox

**Workflow** :

1. Proposer Overlay candidat
2. Tests sandbox automatiques
3. Review multi-validateurs
4. Signature cryptographique
5. Packaging Core + Overlay
6. Déploiement avec audit trail

---

### Q6 : Comment optimiser la latence du reasoner ?

**Réponse** :

**Stratégies** :

1. **Meta-controller adaptatif** :

```normil
// Shortpass pour cas simples
if complexity < 0.5 and latency_target < 200ms {
    return neural_shortpass(input, TinyNet, context)
}
```

2. **Cache des retrieval** :

```normil
let cache = LRUCache.create(max_size=1000)
let cached = cache.get(input_hash)
if cached.is_some() {
    return cached.value
}
```

3. **Quantization** :

```normil
@plastic(mode="quantized_int8")
fn quantized_reasoner(input: Vec) -> Vec {
    // Model quantizé 8-bit
    ...
}
```

4. **Batch processing** :

```normil
// Grouper plusieurs inputs
let outputs = neural_batch_inference(inputs, model)
```

---

## Conclusion

NORMiL est le **langage natif d'implémentation d'O-RedMind**.

**Avantages** :
✅ Types et primitives conçus pour l'architecture
✅ Plasticité, safety, audit intégrés
✅ Performance optimisée
✅ Exemples complets fournis

**Prochaines étapes** :

1. Lire les 5 exemples (`examples/oredmind_*.nor`)
2. Suivre le TUTORIAL.md (Leçons 8.1-8.8)
3. Implémenter votre premier module O-RedMind
4. Tester avec `normil run` et `normil test`

**Ressources** :

- `TUTORIAL.md` : Leçons complètes
- `examples/` : 5 exemples O-RedMind
- `SPECIFICATION.md` : Référence langage
- `PHASE8_OREDMIND.md` : Planning Phase 8

---

**Auteur** : GitHub Copilot
**Date** : Novembre 2025
**Version** : NORMiL Phase 8.5 - Architecture O-RedMind
**License** : MIT
