# Phase 8 - NORMiL pour O-RedMind

## Compléter le Langage pour l'Architecture IA Humanoïde

**Date** : Novembre 2025
**Auteur :** Diego Morales Magri
**Status** : 🎯 PLANIFICATION ORIENTÉE O-REDMIND
**Contexte** : NORMiL doit servir à ÉCRIRE O-RedMind (INDICATIONS_TECHNIQUES.md)

---

## 🎯 Objectif Principal

**NORMiL n'est PAS un langage généraliste pour adoption grand public.**

**NORMiL est LE LANGAGE pour implémenter l'architecture O-RedMind** :

- IA humanoïde avec mémoire épisodique/sémantique
- Perception multimodale (caméra, micro, capteurs)
- Reasoner hybride (neural + symbolique)
- Plasticité contrôlée et auditabilité totale
- Instinct Core avec gouvernance
- Safety & Sandbox obligatoires

**Critère de succès Phase 8** :

> *"Peut-on ÉCRIRE les modules O-RedMind en NORMiL de manière NATIVE et ÉLÉGANTE ?"*

---

## 📊 Gap Analysis : Ce qui Manque

### ✅ Déjà Implémenté (Phase 1-7)

**Langage Core** :

- ✅ Types : `Vec`, `SparseVec`, `EpisodicRecord`, `Concept`, `WorkingMemoryEntry`, `ProtoInstinct`, `Policy`
- ✅ Fonctions, pattern matching, imports
- ✅ Transactions `@atomic` avec audit logging
- ✅ Plasticité `@plastic` avec 6 modes + scheduling + decay
- ✅ Primitives : `lowrankupdate`, `quantize`, `onlinecluster_update`
- ✅ Vectoriel : Operations NumPy, `norm`, dot product, etc.
- ✅ Interop Python : Import modules, appel fonctions

**Tests & Docs** :

- ✅ 273 tests pytest
- ✅ TUTORIAL.md complet
- ✅ Performance : 0.49s benchmark

### ❌ Manquant pour O-RedMind (Phase 8)

#### 1. **Types Manquants**

| Type INDICATIONS_TECHNIQUES.md | Status NORMiL | Priorité |
| ------------------------------ | ------------- | --------- |
| `InstinctPackage`            | ❌ Manquant   | CRITIQUE  |
| `IndexEntry` (HNSW-like)     | ❌ Manquant   | ÉLEVÉE  |
| `ValidationManifest`         | ❌ Manquant   | MOYENNE   |
| `MetaParams`                 | ❌ Manquant   | MOYENNE   |
| `SafetyGuardrail`            | ❌ Manquant   | CRITIQUE  |
| `AuditLogEntry`              | ❌ Manquant   | ÉLEVÉE  |

#### 2. **Primitives Manquantes**

**Index & Retrieval** :

- ❌ `fastindex_query(vec, k, filters)` - Top-k retrieval
- ❌ `hnsw_insert(vec, metadata)` - Insert dans index
- ❌ `bloom_contains(filter, key)` - Bloom filter check
- ❌ `lru_cache_get(cache, key)` - LRU cache access
- ❌ `rerank_neural(candidates, query)` - Re-scoring

**Consolidation** :

- ❌ `priority_sample(episodes, k)` - Replay priorisé
- ❌ `distill_to_semantic(episodes)` - Compression épisodique → sémantique
- ❌ `cluster_centroids(vecs, k)` - K-means online
- ❌ `forgetting_policy(memory, age, utility)` - Gestion oubli

**Instinct Core** :

- ❌ `validate_overlay(overlay, tests)` - Validation overlay
- ❌ `sign_package(package, private_key)` - Signature crypto
- ❌ `verify_signature(package, public_key)` - Vérification
- ❌ `score_prototypes(vec, protos)` - Matching instinct

**Safety & Governance** :

- ❌ `check_guardrail(action, context)` - Vérification sécurité
- ❌ `require_consent(action, user)` - Consentement obligatoire
- ❌ `audit_append(log, entry)` - Append-only logging
- ❌ `hash_chain_verify(log)` - Vérification intégrité
- ❌ `rollback_to_snapshot(state, snapshot_id)` - Rollback

**Multimodal** :

- ❌ `embed_image(image, encoder)` - Encodage image → Vec
- ❌ `embed_audio(audio, encoder)` - Encodage audio → Vec
- ❌ `temporal_align(streams, window_ms)` - Synchro temporelle
- ❌ `cross_attention(vec_img, vec_audio)` - Fusion multimodale

#### 3. **Syntaxe Manquante**

**Pattern Matching Temporel** :

```normil
// ❌ PAS ENCORE IMPLÉMENTÉ
match sequence_of_events {
    pattern [e1, e2, e3] where similarity(e1, e2) > 0.8 => {
        // Détection de pattern temporel
    }
}
```

**Hooks Runtime** :

```normil
// ❌ PAS ENCORE IMPLÉMENTÉ
@before_transaction(fn_name) {
    // Hook avant transaction
}

@after_transaction(fn_name) {
    // Hook après transaction
}
```

**Sandbox Déclaratif** :

```normil
// ❌ PAS ENCORE IMPLÉMENTÉ
@sandboxed(
    allow_memory: ["episodic_store"],
    allow_io: [],
    max_cpu_ms: 100
)
fn untrusted_plugin(input: Vec) -> Vec {
    // Code non vérifié, limité par sandbox
}
```

#### 4. **Modules Manquants**

- ❌ `normil.perception` - Encodeurs multimodaux
- ❌ `normil.index` - FastIndex, HNSW, Bloom
- ❌ `normil.consolidation` - Replay, distillation
- ❌ `normil.instinct` - Instinct Core, overlays
- ❌ `normil.safety` - Guardrails, audit, consent
- ❌ `normil.reasoner` - Symbolic + neural orchestration

---

## 🎯 Phase 8 - Plan d'Implémentation

### Phase 8.1 : Types & Primitives Critiques (Semaine 1-2)

**Objectif** : Pouvoir écrire les modules CORE d'O-RedMind

#### A. Nouveaux Types

**1. InstinctPackage** (CRITIQUE)

```normil
type InstinctPackage = {
    package_id: String,
    version: String,
    signature: String,
    timestamp: Float,
    core: InstinctCore,
    overlay: InstinctOverlay,
    validation_manifest: ValidationManifest
}

type InstinctCore = {
    prototypes: List<ProtoInstinct>,
    rules: List<Rule>,
    meta_params: MetaParams
}

type InstinctOverlay = {
    prototypes: List<ProtoInstinct>,
    rules: List<Rule>,
    provenance: String,
    validation_signature: String
}
```

**2. SafetyGuardrail** (CRITIQUE)

```normil
type SafetyGuardrail = {
    id: String,
    condition: String,      // Expression booléenne
    action_blocked: String, // Action à bloquer
    require_consent: Bool,  // Nécessite consentement humain
    override_level: Int     // Niveau de privilège requis
}

type ConsentRequest = {
    action: String,
    reason: String,
    data_accessed: List<String>,
    expiry_ttl: Int
}
```

**3. AuditLogEntry** (ÉLEVÉE)

```normil
type AuditLogEntry = {
    id: String,
    timestamp: Float,
    event_type: String,
    actor: String,
    action: String,
    data_hash: String,
    prev_hash: String,     // Hash chaining
    signature: String
}
```

**4. IndexEntry** (ÉLEVÉE)

```normil
type IndexEntry = {
    id: String,
    vec: Vec,
    metadata: Map<String, String>,
    neighbors: List<String>,  // HNSW neighbors
    layer: Int,
    timestamp: Float
}
```

#### B. Primitives Critiques

**Index & Retrieval** :

```normil
// Top-k retrieval avec filtres
fn fastindex_query(vec: Vec, k: Int, filters: Map<String, String>) -> List<IndexEntry>

// Insert avec HNSW
fn hnsw_insert(index: Index, vec: Vec, metadata: Map<String, String>) -> IndexEntry

// Re-ranking neural
fn rerank_neural(candidates: List<IndexEntry>, query: Vec, model: RerankModel) -> List<IndexEntry>
```

**Safety & Governance** :

```normil
// Vérification guardrail
fn check_guardrail(action: String, context: Map<String, Any>) -> Result<(), GuardrailViolation>

// Requête de consentement
fn require_consent(request: ConsentRequest, user: User) -> Result<ConsentToken, ConsentDenied>

// Append-only audit log
fn audit_append(log: AuditLog, entry: AuditLogEntry) -> ()

// Vérification hash chain
fn hash_chain_verify(log: AuditLog) -> Result<(), IntegrityError>
```

**Instinct Core** :

```normil
// Signature cryptographique
fn sign_package(package: InstinctPackage, private_key: String) -> InstinctPackage

// Vérification signature
fn verify_signature(package: InstinctPackage, public_key: String) -> Result<(), SignatureError>

// Scoring prototypes
fn score_prototypes(vec: Vec, protos: List<ProtoInstinct>) -> List<(String, Float)>
```

**Consolidation** :

```normil
// Replay priorisé
fn priority_sample(episodes: List<EpisodicRecord>, k: Int, 
                   priority_fn: Fn(EpisodicRecord) -> Float) -> List<EpisodicRecord>

// Distillation épisodique → sémantique
fn distill_to_semantic(episodes: List<EpisodicRecord>) -> Concept

// Politique d'oubli
fn forgetting_policy(memory: EpisodicRecord, age: Float, utility: Float, 
                     threshold: Float) -> Bool  // True = forget
```

#### C. Tests

- [ ] 20 tests pour nouveaux types
- [ ] 30 tests pour primitives index
- [ ] 25 tests pour safety/governance
- [ ] 20 tests pour instinct core
- [ ] 15 tests pour consolidation

**Total** : 110 nouveaux tests (→ 383 tests)

---

### Phase 8.2 : Multimodal & Perception (Semaine 3)

**Objectif** : Pipeline perception multimodale natif en NORMiL

#### A. Primitives Multimodales

```normil
// Encodage image → Vec
fn embed_image(image: ImageData, encoder: ImageEncoder) -> Vec

// Encodage audio → Vec
fn embed_audio(audio: AudioData, encoder: AudioEncoder) -> Vec

// Synchronisation temporelle
fn temporal_align(streams: Map<String, Stream>, window_ms: Int) -> AlignedFrame

// Fusion cross-attention
fn cross_attention_sparse(vec_img: Vec, vec_audio: Vec, 
                          attention_heads: Int) -> Vec
```

#### B. Module `normil.perception`

```normil
import normil.perception as perc

// Pipeline complet
fn perception_pipeline(camera: Camera, mic: Microphone) -> Vec {
    let img_frame = camera.capture()
    let audio_frame = mic.capture(window_ms=500)
  
    // Encodage
    let vec_img = perc.embed_image(img_frame, perc.MobileNetQ8)
    let vec_audio = perc.embed_audio(audio_frame, perc.TinyWavenet)
  
    // Fusion
    let vec_combined = perc.fusion_concat([vec_img, vec_audio])
  
    return vec_combined
}
```

#### C. Tests

- [ ] 15 tests embed_image
- [ ] 15 tests embed_audio
- [ ] 10 tests temporal_align
- [ ] 10 tests cross_attention

**Total** : 50 nouveaux tests (→ 433 tests)

---

### Phase 8.3 : Reasoner Hybride (Semaine 4)

**Objectif** : Reasoner neural + symbolique orchestré

#### A. Primitives Reasoning

```normil
// Symbolic pattern matching
fn symbolic_match(context: Map<String, Any>, rules: List<Rule>) -> List<Rule>

// Neural shortpass (fast inference)
fn neural_shortpass(vec: Vec, model: NeuralModel, context: Vec) -> (Vec, Float)

// Neural longpass (deep reasoning)
fn neural_longpass(vec: Vec, model: NeuralModel, 
                   retrieved: List<EpisodicRecord>) -> (Vec, TraceLog)

// Orchestration
fn meta_controller_decide(vec: Vec, cost_budget: Float, 
                         latency_target_ms: Int) -> ReasoningPath
```

#### B. Module `normil.reasoner`

```normil
import normil.reasoner as reason

@plastic(rate=0.001, mode="lowrank")
fn hybrid_reasoner(input: Vec, context: List<EpisodicRecord>) -> Vec {
    // Meta-controller décide shortpass vs longpass
    let path = reason.meta_controller_decide(
        input, 
        cost_budget=1.0, 
        latency_target_ms=200
    )
  
    let result = if path == "shortpass" {
        let (output, confidence) = reason.neural_shortpass(
            input, 
            reason.TinyNet, 
            context_vec(context)
        )
        output
    } else {
        // Longpass avec symbolic + extended retrieval
        let symbolic_hits = reason.symbolic_match(
            context_to_map(context),
            instinct_rules()
        )
      
        let (output, trace) = reason.neural_longpass(
            input,
            reason.DeepNet,
            context
        )
      
        audit_append(trace)
        output
    }
  
    return result
}
```

#### C. Tests

- [ ] 20 tests symbolic_match
- [ ] 15 tests neural_shortpass
- [ ] 15 tests neural_longpass
- [ ] 10 tests meta_controller

**Total** : 60 nouveaux tests (→ 493 tests)

---

### Phase 8.4 : Outils de Développement (Semaine 5-6)

**Objectif** : Faciliter le développement O-RedMind

#### A. REPL Interactif

```bash
normil repl
>>> let v = vec([1.0, 2.0, 3.0])
Vec([1.0, 2.0, 3.0])
>>> let record = EpisodicRecord.create("test", v, 0.9)
EpisodicRecord(id=a1b2c3d4..., timestamp=1730419200.0, trust=0.90)
>>> fastindex_query(v, k=5, {})
[IndexEntry(...), IndexEntry(...)]
```

**Features** :

- Exécution interactive ligne par ligne
- Historique de commandes (readline)
- Introspection de variables (`inspect(v)`)
- Import de modules NORMiL
- Accès aux primitives

#### B. Introspection Runtime

```normil
@trace
fn traced_function(input: Vec) -> Vec {
    // Automatiquement tracé
    let w = train_step(input)
    return w
}

// Accès aux traces
let traces = get_execution_traces("traced_function")
for trace in traces {
    print(f"Step: {trace.step}, Latency: {trace.latency_ms}ms")
}
```

**Features** :

- Annotation `@trace` pour logging automatique
- `get_execution_traces(fn_name)` pour récupérer traces
- Profiling par fonction (CPU, mémoire)
- Hooks before/after pour debugging

#### C. Visualisation Basique

```normil
import normil.viz as viz

// Logging de métriques
viz.log_metric("loss", compute_loss(w), step)
viz.log_histogram("weights", w, step)

// Export CSV/JSON
viz.export_run("run_001", format="csv")

// Génération graphiques matplotlib
viz.plot_metrics("loss", "run_001")
```

**Features** :

- `log_metric(name, value, step)` - Logging scalaire
- `log_histogram(name, vec, step)` - Distribution
- Export CSV/JSON pour analyse externe
- Plots matplotlib basiques

#### D. CLI Amélioré

```bash
# Exécution avec profiling
normil run --profile examples/perception.nor

# Benchmark automatique
normil benchmark examples/

# Export de métriques
normil run --export-metrics run_001.json examples/training.nor

# Mode debug
normil run --debug --breakpoints="train_step,update_weights" examples/debug.nor
```

**Features** :

- `--profile` : Profiling CPU/mémoire
- `--export-metrics` : Export JSON automatique
- `--debug` : Mode debug avec breakpoints
- `--trace` : Trace complète d'exécution

#### E. Tests

- [ ] 10 tests REPL
- [ ] 15 tests introspection
- [ ] 10 tests visualisation
- [ ] 5 tests CLI

**Total** : 40 nouveaux tests (→ 533 tests)

---

### Phase 8.5 : Documentation & Exemples O-RedMind (Semaine 7)

**Objectif** : Documenter comment écrire O-RedMind en NORMiL

#### A. Exemples Complets

**1. Perception Pipeline** (`examples/oredmind_perception.nor`)

```normil
import normil.perception as perc
import normil.index as idx

fn oredmind_perception_loop(camera: Camera, mic: Mic, 
                            episodic: EpisodicStore, 
                            index: FastIndex) {
    let frame_count = 0
  
    while true {
        // 1. Capture multimodale
        let img = camera.capture()
        let audio = mic.capture(window_ms=500)
      
        // 2. Encodage
        let vec_img = perc.embed_image(img, perc.MobileNetQ8)
        let vec_audio = perc.embed_audio(audio, perc.TinyWavenet)
      
        // 3. Fusion temporelle
        let aligned = perc.temporal_align({
            "img": vec_img,
            "audio": vec_audio
        }, window_ms=500)
      
        let vec_combined = perc.fusion_concat([aligned.img, aligned.audio])
      
        // 4. Append episodic
        let record = EpisodicRecord.create(
            summary=f"Frame {frame_count}",
            vec=vec_combined,
            trust=0.9
        )
      
        @atomic {
            episodic.append(record)
            idx.hnsw_insert(index, vec_combined, {
                "id": record.id,
                "timestamp": record.timestamp
            })
        }
      
        frame_count = frame_count + 1
      
        // 5. Sleep 100ms (10 FPS)
        sleep_ms(100)
    }
}
```

**2. Hybrid Reasoner** (`examples/oredmind_reasoner.nor`)

```normil
import normil.reasoner as reason
import normil.index as idx
import normil.instinct as inst

@plastic(rate=0.001, mode="lowrank")
fn oredmind_reasoner(input: Vec, 
                     index: FastIndex,
                     instinct_pkg: InstinctPackage) -> Vec {
    // 1. Retrieval
    let candidates = idx.fastindex_query(input, k=16, {})
  
    // 2. Instinct scoring
    let proto_scores = inst.score_prototypes(input, instinct_pkg.core.prototypes)
  
    // 3. Meta-controller
    let path = reason.meta_controller_decide(input, cost_budget=1.0, latency_target_ms=200)
  
    // 4. Shortpass ou longpass
    let output = if path == "shortpass" {
        let (out, confidence) = reason.neural_shortpass(
            input, 
            reason.TinyNet,
            merge_context(candidates)
        )
        out
    } else {
        // Longpass avec symbolic
        let symbolic_hits = reason.symbolic_match(
            context_from_candidates(candidates),
            instinct_pkg.core.rules
        )
      
        let (out, trace) = reason.neural_longpass(
            input,
            reason.DeepNet,
            candidates
        )
      
        audit_append(trace)
        out
    }
  
    return output
}
```

**3. Consolidation Worker** (`examples/oredmind_consolidation.nor`)

```normil
import normil.consolidation as cons
import normil.index as idx

fn oredmind_consolidation_worker(episodic: EpisodicStore,
                                 semantic: SemanticStore,
                                 schedule_interval_ms: Int) {
    while true {
        sleep_ms(schedule_interval_ms)
      
        // 1. Priority sampling (replay)
        let priority_fn = fn(record: EpisodicRecord) -> Float {
            // Priorité = f(reward, novelty, recency)
            let novelty = compute_novelty(record)
            let recency = 1.0 / (now() - record.timestamp + 1.0)
            return novelty * 0.5 + recency * 0.5
        }
      
        let episodes = cons.priority_sample(
            episodic.get_all(),
            k=100,
            priority_fn
        )
      
        // 2. Distillation → sémantique
        let concept = cons.distill_to_semantic(episodes)
      
        // 3. Upsert semantic store
        @atomic {
            semantic.upsert(concept)
          
            // 4. Marquer épisodes comme consolidés
            for ep in episodes {
                ep.metadata["consolidated"] = true
            }
        }
      
        // 5. Forgetting policy
        let old_episodes = episodic.get_older_than(days=30)
        for ep in old_episodes {
            let should_forget = cons.forgetting_policy(
                ep,
                age=now() - ep.timestamp,
                utility=compute_utility(ep),
                threshold=0.1
            )
          
            if should_forget {
                episodic.remove(ep.id)
            }
        }
    }
}
```

**4. Safety Layer** (`examples/oredmind_safety.nor`)

```normil
import normil.safety as safe

let guardrails = [
    SafetyGuardrail {
        id: "no_io_without_consent",
        condition: "action.type == 'file_write' or action.type == 'network_send'",
        action_blocked: "*",
        require_consent: true,
        override_level: 10
    },
    SafetyGuardrail {
        id: "no_memory_delete_critical",
        condition: "action.type == 'memory_delete' and memory.tags.contains('critical')",
        action_blocked: "delete",
        require_consent: true,
        override_level: 10
    }
]

fn oredmind_safe_action(action: Action, user: User, audit_log: AuditLog) -> Result<(), Error> {
    // 1. Check guardrails
    for guardrail in guardrails {
        let violation = safe.check_guardrail(
            action.type,
            {"action": action, "user": user}
        )
      
        if violation.is_err() {
            if guardrail.require_consent {
                // 2. Requête consentement
                let consent_req = ConsentRequest {
                    action: action.type,
                    reason: action.reason,
                    data_accessed: action.data_paths,
                    expiry_ttl: 3600000  // 1h
                }
              
                let consent = safe.require_consent(consent_req, user)
              
                if consent.is_err() {
                    // Audit refus
                    safe.audit_append(audit_log, AuditLogEntry {
                        id: generate_uuid(),
                        timestamp: now(),
                        event_type: "consent_denied",
                        actor: user.id,
                        action: action.type,
                        data_hash: hash(action),
                        prev_hash: audit_log.last_hash(),
                        signature: sign(user.private_key, action)
                    })
                  
                    return Err("Consent denied")
                }
            } else {
                return Err("Guardrail violation: " + guardrail.id)
            }
        }
    }
  
    // 3. Execute action
    let result = execute_action(action)
  
    // 4. Audit success
    safe.audit_append(audit_log, AuditLogEntry {
        id: generate_uuid(),
        timestamp: now(),
        event_type: "action_executed",
        actor: user.id,
        action: action.type,
        data_hash: hash(action),
        prev_hash: audit_log.last_hash(),
        signature: sign(user.private_key, action)
    })
  
    return result
}
```

**5. Instinct Governance** (`examples/oredmind_instinct.nor`)

```normil
import normil.instinct as inst
import normil.safety as safe

fn oredmind_instinct_governance(
    core: InstinctCore,
    overlay_candidate: InstinctOverlay,
    tests: List<ValidationTest>,
    audit_log: AuditLog
) -> Result<InstinctPackage, Error> {
  
    // 1. Sandbox tests
    let test_results = []
    for test in tests {
        let result = inst.run_test_sandboxed(overlay_candidate, test)
        test_results.push(result)
    }
  
    // 2. Vérifier tous tests passent
    let all_passed = test_results.all(fn(r) { r.passed })
  
    if !all_passed {
        return Err("Overlay tests failed")
    }
  
    // 3. Validation signature
    let validation_manifest = ValidationManifest {
        tests_passed: test_results.map(fn(r) { r.test_id }),
        metrics_before: compute_metrics(core),
        metrics_after: compute_metrics_with_overlay(core, overlay_candidate),
        validators: ["validator_1", "validator_2"],
        timestamp: now()
    }
  
    // 4. Signature overlay
    let signed_overlay = inst.sign_overlay(
        overlay_candidate,
        validation_manifest,
        private_key="org_private_key"
    )
  
    // 5. Créer package
    let package = InstinctPackage {
        package_id: generate_uuid(),
        version: "1.1.0",
        signature: inst.sign_package_hash({core, signed_overlay}),
        timestamp: now(),
        core: core,
        overlay: signed_overlay,
        validation_manifest: validation_manifest
    }
  
    // 6. Audit
    safe.audit_append(audit_log, AuditLogEntry {
        id: generate_uuid(),
        timestamp: now(),
        event_type: "instinct_overlay_validated",
        actor: "governance_system",
        action: "create_package",
        data_hash: hash(package),
        prev_hash: audit_log.last_hash(),
        signature: sign("org_private_key", package)
    })
  
    return Ok(package)
}
```

#### B. Documentation TUTORIAL.md

- [ ] Leçon 8.1 : Types O-RedMind (InstinctPackage, SafetyGuardrail, etc.)
- [ ] Leçon 8.2 : Perception multimodale
- [ ] Leçon 8.3 : Index et retrieval
- [ ] Leçon 8.4 : Reasoner hybride
- [ ] Leçon 8.5 : Consolidation et replay
- [ ] Leçon 8.6 : Safety et gouvernance
- [ ] Leçon 8.7 : Instinct Core
- [ ] Leçon 8.8 : Pipeline complet O-RedMind

#### C. Guide Architecture

**`docs/OREDMIND_ARCHITECTURE.md`** :

- Mapping INDICATIONS_TECHNIQUES.md → NORMiL
- Modules NORMiL pour chaque brique O-RedMind
- Patterns recommandés
- Anti-patterns à éviter

---

## 📊 Résumé Phase 8

### Livrables

| Semaine | Focus              | Livrables               | Tests |
| ------- | ------------------ | ----------------------- | ----- |
| 1-2     | Types & Primitives | 4 types + 15 primitives | +110  |
| 3       | Multimodal         | Module perception       | +50   |
| 4       | Reasoner           | Module reasoner         | +60   |
| 5-6     | Dev Tools          | REPL + Viz + CLI        | +40   |
| 7       | Documentation      | 5 exemples + 8 leçons  | -     |

**Total** :

- ✅ 4 nouveaux types complexes
- ✅ ~25 nouvelles primitives
- ✅ 3 nouveaux modules (`perception`, `reasoner`, `safety`)
- ✅ 260 nouveaux tests (→ **533 tests total**)
- ✅ 5 exemples complets O-RedMind
- ✅ 8 nouvelles leçons TUTORIAL
- ✅ REPL + Visualisation + CLI amélioré

### Critère de Succès

**Phase 8 réussie si** :

1. ✅ Tous les types INDICATIONS_TECHNIQUES.md sont implémentés
2. ✅ Pipeline perception multimodale fonctionne nativement en NORMiL
3. ✅ Reasoner hybride (neural + symbolique) est codable élégamment
4. ✅ Safety & governance sont natifs (guardrails, audit, consent)
5. ✅ Instinct Core avec validation et signature fonctionne
6. ✅ Consolidation et replay sont scriptables
7. ✅ 533 tests passent (dont 260 nouveaux)
8. ✅ 5 exemples O-RedMind complets fonctionnent

**Validation finale** :

> *"Peut-on ÉCRIRE le système O-RedMind complet en NORMiL pur, de manière NATIVE et ÉLÉGANTE, sans hacks Python ?"*

Si **OUI** → Phase 8 succès ✅
Si **NON** → Identifier gaps et itérer

---

## 🚀 Prochaines Actions Immédiates

1. **Valider cette planification** avec vous
2. **Créer les types** (InstinctPackage, SafetyGuardrail, etc.)
3. **Implémenter primitives critiques** (fastindex_query, check_guardrail, etc.)
4. **Tests systématiques** (+110 tests semaine 1-2)
5. **Itérer** selon feedback

**Question** : Cette approche est-elle **alignée avec votre vision** d'O-RedMind ?

---

**Auteur** : GitHub Copilot
**Date** : Novembre 2025
**Version** : NORMiL Phase 8 - O-RedMind Focus
**Status** : 🎯 PRÊT POUR VALIDATION
