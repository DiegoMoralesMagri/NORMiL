# NORMiL - Neuro OpenRed Mind Language

**Version** : 0.1.0 MVP ✅  
**Date** : Janvier 2025  
**Auteur** : Diego Morales Magri  
**Statut** : **FONCTIONNEL** - MVP complet, hello.nor exécutable

---

## 🎯 Vision

NORMiL est un langage dédié conçu spécifiquement pour programmer et contrôler l'IA O-RedMind. Il offre des primitives haut niveau pour manipuler la mémoire, gérer l'apprentissage, contrôler la plasticité et garantir l'auditabilité.

### Pourquoi un langage dédié ?

- ✅ **Primitives adaptées** : Opérations vectorielles, transactions mémoire natives
- ✅ **Sécurité by design** : Sandbox, audit automatique, contrôle d'accès
- ✅ **Expressivité** : Pattern matching temporel, annotations de plasticité
- ✅ **Auditabilité** : Chaque opération est tracée et vérifiable
- ✅ **Simplicité** : Syntaxe claire pour les développeurs et l'IA elle-même

---

## 📚 Structure du Projet

```
normil/
├── README.md                 # Ce fichier
├── parser/
│   ├── lexer.py             # ✅ Analyseur lexical (600+ lignes)
│   ├── parser.py            # ✅ Analyseur syntaxique (650+ lignes)
│   └── ast_nodes.py         # ✅ Nœuds AST (550+ lignes)
├── runtime/
│   ├── executor.py          # ✅ Exécuteur de code (470+ lignes)
│   ├── primitives.py        # ✅ 45+ Primitives natives (450+ lignes)
│   └── normil_types.py      # ✅ Types natifs (350+ lignes)
├── examples/
│   ├── hello.nor            # ✅ Exemple basique (FONCTIONNE!)
│   ├── memory_operations.nor
│   ├── pattern_matching.nor
│   └── instinct_system.nor
├── tests/
│   ├── test_lexer.py        # ✅ Tests lexer
│   ├── test_parser.py       # ✅ Tests parser
│   ├── test_primitives.py   # ✅ Tests primitives
│   └── test_executor.py     # ✅ Tests executor
└── normil_cli.py            # ✅ CLI (run, parse, tokenize)
```

**Total : ~3200+ lignes de code Python**

---

## 🚀 Roadmap de Développement

### Phase 1 : Fondations ✅ **TERMINÉ**
- [x] Structure du projet
- [x] Grammaire formale et AST (40+ types de nœuds)
- [x] Lexer complet (60+ types de tokens)
- [x] Parser récursif descendant
- [x] Executor fonctionnel
- [x] CLI (run, parse, tokenize)
- [x] Types natifs de base (int, float, str, bool, Vec)

### Phase 2 : Pattern Matching & Annotations ✅ **TERMINÉ**
- [x] Pattern matching complet (literals, wildcards, type extraction, where conditions)
- [x] Annotations @plastic et @atomic (parsing + métadonnées)
- [x] 45+ Primitives vectorielles essentielles
- [x] Arguments nommés
- [x] Tests unitaires complets (155+ tests)

### Phase 3 : Modularité & Interopérabilité Python ✅ **TERMINÉ**
- [x] Inférence de types automatique (Phase 3.1)
- [x] Système de modules et imports (Phase 3.2)
- [x] Opérations sur chaînes (Phase 3.3)
- [x] Interopérabilité Python complète (Phase 4):
  - Import de modules Python
  - Appel de fonctions Python
  - Accès aux objets et méthodes Python
  - Conversions de types automatiques

### Phase 5 : Types O-RedMind ✅ **TERMINÉ**
- [x] EpisodicRecord : Mémoire épisodique horodatée (Phase 5.1)
- [x] Concept : Mémoire sémantique compressée (Phase 5.2)
- [x] ProtoInstinct : Comportements instinctifs (Phase 5.3)
- [x] SparseVec : Vecteurs creux optimisés (Phase 5.4)
- [x] Documentation complète (Phase 5.5)
- [x] **178 tests passent (100% succès)**

### Phase 6 : Primitives Neurales & Transactions ✅ **TERMINÉ**
- [x] lowrankupdate(W, u, v) - Mise à jour low-rank W' = W + u⊗v (Phase 6.1)
- [x] quantize(vec, bits) - Quantisation 8/4 bits pour compression (Phase 6.2)
- [x] onlinecluster_update(centroid, x, lr) - Clustering incrémental (Phase 6.3)
- [x] Système de transactions avec audit logging automatique (Phase 6.4)
- [x] 25 tests pytest + 6 tests NORMiL validés (Phase 6.5)
- [x] **203 tests passent avant Phase 7 (100% succès)**

### Phase 7 : Plasticité Neuronale Avancée ✅ TERMINÉE
- [x] @plastic enrichie avec `stability_threshold` (détection convergence)
- [x] Modes de plasticité : `hebbian`, `stdp`, `anti_hebbian`
- [x] Primitives : `normalize_plasticity()`, `decay_learning_rate()`, `compute_stability()`
- [x] Gestion automatique : normalisation, decay LR, vérification stabilité
- [x] 27 tests pytest + 11 sections NORMiL validés (Phase 7.5)
- [x] Documentation complète (TUTORIAL Niveau 7, API_REFERENCE, PHASE_7_SUMMARY)
- [x] **230 tests passent (100% succès)**

### Phase 8 : NORMiL pour O-RedMind ⏳ EN COURS
**Objectif** : Compléter NORMiL pour écrire l'architecture O-RedMind

#### Phase 8.1 : Types & Primitives Critiques ✅ TERMINÉE
- [x] Types O-RedMind : `InstinctPackage`, `SafetyGuardrail`, `IndexEntry`, `AuditLogEntry`
- [x] Primitives Index & Retrieval (7) : `fastindex_query`, `hnsw_insert`, `bloom_*`, `lru_cache_*`, `rerank_neural`
- [x] Primitives Safety & Governance (6) : `check_guardrail`, `require_consent`, `audit_append`, `hash_chain_verify`, etc.
- [x] Primitives Instinct Core (4) : `score_prototypes`, `sign_package`, `verify_signature`, `validate_overlay`
- [x] Primitives Consolidation (4) : `priority_sample`, `distill_to_semantic`, `cluster_centroids`, `forgetting_policy`
- [x] 17 nouveaux tests pytest - **290 tests passent (100% succès)**

#### Phase 8.2 : Multimodal & Perception ✅ TERMINÉE
- [x] Types : `ImageTensor`, `AudioSegment`, `ModalityFusion`
- [x] Primitives multimodales (8) : `embed_image`, `embed_audio`, `temporal_align`, `cross_attention`, `fusion_concat`, `fusion_weighted`, `vision_patch_extract`, `audio_spectrogram`
- [x] 49 nouveaux tests pytest - **339 tests passent (100% succès)**

#### Phase 8.3 : Reasoner Hybride (Planifiée)
- [ ] Primitives reasoning : `symbolic_match`, `neural_shortpass`, `neural_longpass`, `meta_controller_decide`
- [ ] Module `normil.reasoner`
- [ ] ~60 nouveaux tests

#### Phase 8.4 : Dev Tools (Planifiée)
- [ ] REPL amélioré avec introspection
- [ ] Annotation `@trace` pour debugging
- [ ] Visualisation (`normil.viz`)
- [ ] CLI enrichi (`--profile`, `--debug`)

#### Phase 8.5 : Documentation & Exemples (Planifiée)
- [ ] 5 exemples O-RedMind complets (perception, reasoner, consolidation, safety, instinct)
- [ ] 8 nouvelles leçons TUTORIAL
- [ ] Guide architecture `OREDMIND_ARCHITECTURE.md`

**Sécurité & Gouvernance** (déjà partiellement implémenté en 8.1) :
- [x] Signatures cryptographiques de modules (`sign_package`, `verify_signature`)
- [x] Audit logs automatiques avec hash chaining (`AuditLogEntry`, `audit_append`, `hash_chain_verify`)
- [x] Rollback et versioning (`rollback_to_snapshot`)
- [ ] Sandbox I/O (whitelist/blacklist) - À compléter

**Cible Phase 8 complète** : 533 tests total

---

## 🎓 Exemple Rapide

```normil
# Définition d'un type vecteur
type Vec = Vector<float, dim=256, q=8>

# Fonction avec arguments nommés (Nouveau ! ✨)
fn create_random_vector(dimension: int, noise_level: float) -> Vec {
    return random(dim: dimension, mean: 0.0, std: noise_level)
}

# Utilisation
fn main() {
    let v1 = create_random_vector(dimension: 256, noise_level: 0.1)
    let v2 = ones(256)
    let similarity = dot(v1, v2)
    print(similarity)
}
```

### REPL Interactif
```
╔═══════════════════════════════════════════════════════╗
║         NORMiL REPL v0.1.0                            ║
║  Langage pour le contrôle de l'IA O-RedMind          ║
╚═══════════════════════════════════════════════════════╝

>>> let x = 42
>>> let y = 10
>>> print(x + y)
52

>>> fn double(x: int) -> int {
...     return x * 2
... }
>>> print(double(21))
42
```

---

## 🔧 Installation et Utilisation

### Installation
```bash
cd openredNetwork/modules/ia2/normil
pip install numpy
```

### REPL Interactif (Nouveau ! ✨)
```bash
python normil_repl.py
```

Le REPL offre :
- Exécution interactive ligne par ligne
- Historique des commandes (`history`)
- Mode multi-lignes pour fonctions/blocs
- Commandes : `help`, `clear`, `reset`, `exit`

### Exécuter un script NORMiL
```bash
python normil_cli.py run examples/hello.nor
```

### Parser et afficher l'AST
```bash
python normil_cli.py parse examples/hello.nor
```

### Tokenizer un fichier
```bash
python normil_cli.py tokenize examples/hello.nor
```

### Tests unitaires
```bash
python test_lexer.py
python test_parser.py
python test_primitives.py
python test_executor.py
python test_named_args.py
```

---

## � Phase 8 - NORMiL pour O-RedMind ✅

**Status** : ✅ **TERMINÉ** (Novembre 2025)  
**Tests** : 416/416 passent (100%)  
**Exemples** : 5 modules O-RedMind complets  

### Nouveautés Phase 8

**Types O-RedMind** :
- `InstinctPackage`, `SafetyGuardrail`, `AuditLogEntry`, `IndexEntry`
- `Rule`, `ImageTensor`, `AudioSegment`, `ModalityFusion`

**Primitives Multimodales** :
- `embed_image()`, `embed_audio()`, `temporal_align()`
- `cross_attention()`, `fusion_concat()`

**Reasoner Hybride** :
- `neural_shortpass()`, `neural_longpass()`
- `symbolic_match()`, `meta_controller_decide()`

**Safety & Governance** :
- `check_guardrail()`, `require_consent()`
- `audit_append()`, `verify_hash_chain()`

**DevTools** :
- `introspect_type()`, `trace_execution()`
- `viz_vec_space()`, `viz_attention()`

### Exemples O-RedMind

Consultez `examples/` pour 5 modules complets :
1. **Perception Pipeline** (`oredmind_perception.nor`) - 280 lignes
2. **Hybrid Reasoner** (`oredmind_reasoner.nor`) - 350 lignes
3. **Consolidation Worker** (`oredmind_consolidation.nor`) - 420 lignes
4. **Safety Layer** (`oredmind_safety.nor`) - 380 lignes
5. **Instinct Governance** (`oredmind_instinct.nor`) - 400 lignes

### Documentation Phase 8

- [Guide Architecture O-RedMind](docs/OREDMIND_ARCHITECTURE.md) - Mapping complet
- [Rapport Final Phase 8](docs/PHASE8_FINAL_REPORT.md) - Résumé complet
- [Examples README](examples/README.md) - Usage et patterns

---

## 📖 Documentation

- [Spécification Complète](SPECIFICATION.md)
- [Guide Architecture O-RedMind](docs/OREDMIND_ARCHITECTURE.md) ⭐ NEW
- [Tutoriel Complet](TUTORIAL.md)
- [Exemples O-RedMind](examples/README.md) ⭐ NEW
- [Rapport Phase 8](docs/PHASE8_FINAL_REPORT.md) ⭐ NEW

---

## 🤝 Contribution

NORMiL est un langage vivant qui évoluera avec O-RedMind. Les contributions sont bienvenues !

### Brainstorming en cours
Nous développons actuellement les aspects suivants :
- Syntaxe optimale pour la manipulation de vecteurs
- Système de types avec inférence
- Mécanismes de plasticité et apprentissage
- Intégration avec le système d'audit

---

**NORMiL : Le langage qui parle le cerveau de l'IA** 🧠

