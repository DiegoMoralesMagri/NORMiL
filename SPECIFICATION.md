# NORMiL - Spécification du Langage v0.1


**Date** : Novembre 2025
**Auteur** : Diego Morales Magri

---

**Extension de fichier : `*.nor`**

## 📋 Table des Matières

1. [Introduction](#introduction)
2. [Types Natifs](#types-natifs)
3. [Syntaxe de Base](#syntaxe-de-base)
4. [Primitives](#primitives)
5. [Annotations](#annotations)
6. [Pattern Matching](#pattern-matching)
7. [Transactions](#transactions)
8. [Sandbox et Sécurité](#sandbox-et-sécurité)
9. [Système d&#39;Audit](#système-daudit)

---

## 1. Introduction

NORMiL (fichiers `*.nor`) est un langage **typé statiquement** avec **inférence de types** partielle, conçu pour manipuler la mémoire, les vecteurs et l'apprentissage de l'IA O-RedMind.

### Philosophie de Design

- **Explicite > Implicite** : Les opérations sur la mémoire et l'apprentissage doivent être claires
- **Sécurité par défaut** : Sandbox et audit automatiques
- **Expressivité** : Syntaxe concise mais lisible
- **Auditabilité** : Chaque opération laisse une trace

---

## 2. Types Natifs

### 2.1. Types Primitifs

```normil
# Types de base
int       # Entier 64 bits
float     # Flottant 64 bits
bool      # Booléen
str       # Chaîne de caractères UTF-8
timestamp # Timestamp Unix (float)
uuid      # UUID v4
```

### 2.2. Types Vectoriels

```normil
# Vecteur dense
type Vec = Vector<float, dim=256, q=8>
# - float : type des éléments
# - dim : dimension (fixe)
# - q : quantisation (bits par élément, optionnel)

# Vecteur sparse
type SparseVec = SparseVector<float, dim=1024, sparsity=0.9>
# - sparsity : proportion d'éléments à zéro

# BRAINSTORM : Devrait-on permettre des vecteurs de dimension dynamique ?
# Option 1 : type DynVec = Vector<float, dim=?, q=8>
# Option 2 : Garder dim fixe pour la sécurité de type
```

### 2.3. Types Mémoire

```normil
# Souvenir épisodique
type EpisodicRecord = {
    id: uuid,
    timestamp: timestamp,
    sources: list<str>,
    vecs: map<str, Vec>,        # "image" -> vec, "audio" -> vec
    summary: str,
    labels: list<Label>,
    trust: float,               # 0.0 à 1.0
    provenance: Provenance,
    outcome: optional<str>
}

# Label avec score
type Label = {
    label: str,
    score: float
}

# Provenance (traçabilité)
type Provenance = {
    device_id: str,
    signature: str,             # Signature cryptographique
    timestamp: timestamp
}

# Entrée de mémoire de travail
type WorkingMemoryEntry = {
    id: uuid,
    vec_combined: Vec,
    last_access_ms: timestamp,
    relevance_score: float,
    expire_ttl: int,            # millisecondes
    refs_to_episodic_ids: list<uuid>
}

# Concept sémantique
type Concept = {
    concept_id: uuid,
    centroid_vec: Vec,
    doc_count: int,
    provenance_versions: list<str>,
    trust_score: float,
    labels: list<str>
}
```

### 2.4. Types Instinct

```normil
# Prototype d'instinct
type ProtoInstinct = {
    id: str,
    vec_ref: Vec,
    rule: optional<Rule>,
    weight: float
}

# Règle symbolique
type Rule = {
    id: str,
    condition: str,             # Expression booléenne
    action: str,                # Action à effectuer
    priority: int
}

# Politique (meta-règle)
type Policy = {
    name: str,
    rules: list<Rule>,
    activation_threshold: float
}
```

### 2.5. BRAINSTORM : Types Additionnels ?

```normil
# Devrait-on ajouter :

# 1. Type pour les séquences temporelles ?
type Sequence<T> = list<(timestamp, T)>

# 2. Type pour les graphes de mémoire ?
type MemoryGraph = {
    nodes: map<uuid, Node>,
    edges: list<Edge>
}

# 3. Type pour les événements ?
type Event = {
    event_type: str,
    data: map<str, any>,
    timestamp: timestamp
}

# QUESTION : Quels autres types seraient utiles ?
```

---

## 3. Syntaxe de Base

### 3.1. Déclarations de Variables

```normil
# Déclaration avec type explicite
let x: int = 42
let name: str = "O-RedMind"
let vec: Vec = zeros(256)

# Déclaration avec inférence de type
let y = 3.14              # inféré comme float
let active = true         # inféré comme bool

# Constantes
const PI: float = 3.14159
const MAX_EPISODES: int = 10000
```

### 3.2. Fonctions

```normil
# Fonction simple
fn add(a: int, b: int) -> int {
    return a + b
}

# Fonction avec types vectoriels
fn similarity(v1: Vec, v2: Vec) -> float {
    return dot(v1, v2) / (norm(v1) * norm(v2))
}

# Fonction générique (BRAINSTORM : supporter ?)
fn first<T>(list: list<T>) -> T {
    return list[0]
}

# Fonction avec valeurs par défaut
fn create_vec(dim: int = 256, init: float = 0.0) -> Vec {
    return fill(dim, init)
}
```

### 3.3. Structures de Contrôle

```normil
# If / else
if condition {
    // code
} else if other_condition {
    // code
} else {
    // code
}

# Boucles
for i in range(0, 10) {
    print(i)
}

for item in list {
    process(item)
}

while condition {
    // code
}

# BRAINSTORM : Pattern matching sur types ?
match value {
    case int(x) -> print("Entier: " + str(x))
    case str(s) -> print("Chaîne: " + s)
    case _ -> print("Autre")
}
```

### 3.4. BRAINSTORM : Syntaxe Spéciale pour Vecteurs ?

```normil
# Option 1 : Opérateurs dédiés
let v3 = v1 .+ v2        # Addition vectorielle
let v4 = v1 .* v2        # Produit élément par élément
let v5 = v1 @ v2         # Produit scalaire

# Option 2 : Fonctions explicites
let v3 = add(v1, v2)
let v4 = mul(v1, v2)
let v5 = dot(v1, v2)

# QUESTION : Quelle syntaxe préférez-vous ?
# - Option 1 : Plus concise, style NumPy
# - Option 2 : Plus explicite, meilleure lisibilité
```

---

## 4. Primitives

### 4.1. Primitives Mémoire

```normil
# Append à l'Episodic Log
primitive episodic_append(record: EpisodicRecord) -> uuid

# Query sur l'Episodic Log
primitive episodic_query(
    vec: Vec, 
    k: int = 10,
    filters: map<str, any> = {}
) -> list<EpisodicRecord>

# Working Memory
primitive wm_add(entry: WorkingMemoryEntry) -> void
primitive wm_get(id: uuid) -> optional<WorkingMemoryEntry>
primitive wm_query(vec: Vec, k: int) -> list<WorkingMemoryEntry>

# Semantic Store
primitive semantic_upsert(concept: Concept) -> void
primitive semantic_query(vec: Vec, k: int) -> list<Concept>
primitive semantic_merge(c1: Concept, c2: Concept) -> Concept
```

### 4.2. Primitives Vectorielles

```normil
# Création
primitive zeros(dim: int) -> Vec
primitive ones(dim: int) -> Vec
primitive fill(dim: int, value: float) -> Vec
primitive random(dim: int, mean: float = 0.0, std: float = 1.0) -> Vec

# Opérations
primitive dot(v1: Vec, v2: Vec) -> float
primitive norm(v: Vec) -> float
primitive normalize(v: Vec) -> Vec
primitive add(v1: Vec, v2: Vec) -> Vec
primitive sub(v1: Vec, v2: Vec) -> Vec
primitive mul(v1: Vec, v2: Vec) -> Vec  # Element-wise
primitive scale(v: Vec, scalar: float) -> Vec

# Transformations
primitive quantize(v: Vec, bits: int) -> Vec
primitive dequantize(v: Vec) -> Vec
```

### 4.3. Primitives Apprentissage

```normil
# Low-rank update (pour plasticité)
primitive lowrankupdate(
    W: Matrix,           # BRAINSTORM : Ajouter type Matrix ?
    u: Vec,
    v: Vec
) -> Matrix

# Online clustering
primitive onlinecluster_update(
    centroid: Vec,
    new_point: Vec,
    learning_rate: float = 0.01
) -> Vec

# Consolidation
primitive consolidate(
    episodes: list<EpisodicRecord>,
    method: str = "cluster"  # "cluster" ou "distill"
) -> list<Concept>
```

### 4.4. Primitives Audit

```normil
# Logger une action
primitive audit_log(
    action: str,
    data: map<str, any>,
    level: str = "info"
) -> void

# Vérifier l'intégrité
primitive audit_verify(
    from_timestamp: timestamp,
    to_timestamp: timestamp
) -> bool

# Créer un snapshot
primitive audit_snapshot(name: str) -> str  # Retourne hash
```

### 4.5. BRAINSTORM : Autres Primitives Utiles ?

```normil
# Primitives temps réel ?
primitive now() -> timestamp
primitive sleep(ms: int) -> void

# Primitives I/O ?
primitive read_file(path: str) -> str
primitive write_file(path: str, content: str) -> void

# Primitives réseau (pour fédération) ?
primitive send_to_peer(peer_id: str, data: any) -> void
primitive receive_from_peer() -> optional<any>

# QUESTION : Quelles primitives sont essentielles vs optionnelles ?
```

---

## 5. Annotations

### 5.1. Annotation @plastic

```normil
# Contrôle la plasticité (capacité d'apprentissage)
@plastic(
    rate: float = 0.001,
    mode: str = "lowrank",       # "lowrank", "full", "frozen"
    stability_threshold: float = 0.95
)
fn process_input(input: Vec) -> Vec {
    // Cette fonction peut s'adapter
}

# BRAINSTORM : Autres paramètres utiles ?
# - decay_rate: float  (pour oubli progressif)
# - max_updates: int   (limite d'adaptations)
# - context: str       (profil contextuel)
```

### 5.2. Annotation @audit

```normil
# Force l'audit d'une fonction
@audit(level: str = "full")
fn critical_operation(data: EpisodicRecord) -> void {
    // Toutes les opérations sont auditées
}

# Niveaux d'audit :
# - "none" : Pas d'audit
# - "minimal" : Entrée/sortie seulement
# - "full" : Chaque opération intermédiaire
```

### 5.3. Annotation @sandbox

```normil
# Exécution dans un sandbox isolé
@sandbox(
    allow_io: bool = false,
    allow_network: bool = false,
    max_memory_mb: int = 100,
    max_time_ms: int = 1000
)
fn untrusted_code(input: str) -> str {
    // Code non sûr exécuté en isolation
}
```

### 5.4. BRAINSTORM : Autres Annotations ?

```normil
# @cache pour mémoïsation ?
@cache(ttl_ms: int = 60000)
fn expensive_computation(x: int) -> int { }

# @parallel pour exécution parallèle ?
@parallel(threads: int = 4)
fn batch_process(items: list<Vec>) -> list<Vec> { }

# @profile pour analyse de performance ?
@profile
fn critical_path(data: any) -> any { }
```

---

## 6. Pattern Matching

### 6.1. Pattern Matching Temporel

```normil
# Détection de séquences dans l'historique
match sequence in episodic_log {
    pattern [e1, e2, e3] 
    where similarity(e1.vecs["image"], e3.vecs["image"]) > 0.8 
    and time_diff(e1, e3) < 60000  # moins d'une minute
    -> {
        print("Séquence répétitive détectée !")
        consolidate([e1, e2, e3])
    }
}

# BRAINSTORM : Syntaxe alternative ?
# Option 1 : Style regex
# pattern /A B* C/ where A.similarity(C) > 0.8

# Option 2 : Style SQL
# SELECT e1, e2, e3 FROM episodic_log
# WHERE similarity(e1, e3) > 0.8
# ORDER BY timestamp
```

### 6.2. Pattern Matching sur Types

```normil
match value {
    case EpisodicRecord(e) where e.trust > 0.8 -> {
        process_trusted(e)
    }
    case EpisodicRecord(e) -> {
        process_untrusted(e)
    }
    case _ -> {
        print("Type inconnu")
    }
}
```

### 6.3. BRAINSTORM : Pattern Matching Avancé ?

```normil
# Extraction de patterns dans les vecteurs ?
match vec {
    pattern high_activation where max(vec) > 0.9 -> {
        print("Activation forte détectée")
    }
    pattern sparse where count_nonzero(vec) < dim * 0.1 -> {
        print("Vecteur sparse")
    }
}

# Pattern sur graphes de mémoire ?
match memory_graph {
    pattern cycle(nodes) where len(nodes) > 2 -> {
        print("Cycle détecté : boucle de pensée")
    }
}
```

---

## 7. Transactions

### 7.1. Transactions de Base

```normil
# Transaction simple
transaction append_and_log(record: EpisodicRecord) {
    let id = episodic_append(record)
    audit_log("episode_appended", {"id": id})
}

# Transaction avec rollback
transaction safe_update(old_vec: Vec, new_vec: Vec) {
    try {
        let backup = old_vec
        // Opérations
        update_memory(new_vec)
        audit_log("memory_updated", {})
    } catch error {
        // Rollback automatique
        update_memory(backup)
        audit_log("rollback", {"error": error})
    }
}
```

### 7.2. Transactions Atomiques

```normil
# Garantit l'atomicité (tout ou rien)
atomic transaction consolidate_batch(episodes: list<EpisodicRecord>) {
    for episode in episodes {
        episodic_append(episode)
    }
    // Si une opération échoue, tout est annulé
}
```

### 7.3. BRAINSTORM : Transactions Avancées ?

```normil
# Transactions distribuées (pour fédération) ?
distributed transaction sync_with_peer(peer_id: str, data: list<EpisodicRecord>) {
    send_to_peer(peer_id, data)
    let ack = wait_for_ack(peer_id, timeout_ms=5000)
    if not ack {
        rollback()
    }
}

# Transactions avec compensation ?
compensating transaction process_with_undo(data: any) {
    let result = process(data)
  
    on_rollback {
        unprocess(result)  // Action de compensation
    }
}
```

---

## 8. Sandbox et Sécurité

### 8.1. Contrôle d'Accès

```normil
# Déclaration de permissions
permissions {
    allow read on episodic_log
    allow write on working_memory
    deny delete on episodic_log
}

# Vérification de permission
if has_permission("write", "semantic_store") {
    semantic_upsert(concept)
}
```

### 8.2. Signature de Modules

```normil
# Module signé (pour instincts par exemple)
@signed(
    public_key: str = "...",
    signature: str = "..."
)
module trusted_instinct {
    // Code vérifié et signé
}

# Validation automatique au chargement
```

### 8.3. BRAINSTORM : Autres Mécanismes de Sécurité ?

```normil
# Whitelist d'accès mémoire ?
@memory_access(
    allow: list<str> = ["working_memory", "episodic_log"],
    deny: list<str> = ["semantic_store"]
)

# Rate limiting ?
@rate_limit(max_calls_per_second: int = 100)

# Quota de ressources ?
@quota(max_memory_mb: int = 500, max_cpu_percent: int = 50)
```

---

## 9. Système d'Audit

### 9.1. Hooks d'Audit

```normil
# Hook avant transaction
before_transaction fn log_before(tx_name: str, args: map<str, any>) {
    audit_log("tx_start", {"name": tx_name, "args": args})
}

# Hook après transaction
after_transaction fn log_after(tx_name: str, result: any) {
    audit_log("tx_end", {"name": tx_name, "result": result})
}

# Hook sur erreur
on_error fn log_error(error: Error) {
    audit_log("error", {"message": error.message}, level="error")
}
```

### 9.2. Vérification d'Intégrité

```normil
# Vérifier le hash chain
fn verify_integrity() -> bool {
    let logs = audit_get_logs()
    for i in range(1, len(logs)) {
        if not verify_hash_chain(logs[i-1], logs[i]) {
            return false
        }
    }
    return true
}
```

### 9.3. BRAINSTORM : Fonctionnalités d'Audit Avancées ?

```normil
# Audit sélectif par niveau ?
audit_set_level("critical_operations", "full")
audit_set_level("routine_operations", "minimal")

# Export d'audit pour analyse ?
audit_export(
    from: timestamp,
    to: timestamp,
    format: str = "json",  # json, csv, parquet
    destination: str = "audit_report.json"
)

# Audit queries pour analyse ?
audit_query("SELECT * FROM audit_log WHERE level='error' AND timestamp > ?", [yesterday])
```

---

## 10. BRAINSTORM : Fonctionnalités Futures

### 10.1. Compilation ?

- Compiler NORMiL vers bytecode pour performance ?
- JIT compilation pour hot paths ?

### 10.2. Interopérabilité ?

- Appeler du Python depuis NORMiL ?
- Exporter des fonctions NORMiL pour Python ?

### 10.3. Debugging ?

- Debugger avec breakpoints ?
- Stepping et inspection de variables ?

### 10.4. IDE Support ?

- Syntax highlighting ?
- Autocomplétion ?
- Linting et formatage ?

---

## 📝 Prochaines Étapes

1. **Finaliser la grammaire** : Convertir cette spec en grammaire EBNF formelle
2. **Implémenter le lexer** : Tokenisation du code source
3. **Implémenter le parser** : Construction de l'AST
4. **Runtime minimal** : Exécution des primitives de base
5. **Tests** : Suite de tests pour chaque fonctionnalité

---

## 💡 Questions Ouvertes pour Brainstorming

1. **Syntaxe des opérations vectorielles** : Opérateurs spéciaux (.+, .*) ou fonctions explicites ?
2. **Généricité** : Support des types génériques `<T>` ?
3. **Vecteurs dynamiques** : Dimension fixe ou permettre dim=? ?
4. **Pattern matching avancé** : Sur vecteurs, graphes, séquences ?
5. **Transactions distribuées** : Pour fédération d'IAs ?
6. **Primitives I/O** : Lecture/écriture fichiers, réseau ?
7. **Compilation** : Interpréter ou compiler ?

**Vos retours et idées sont essentiels pour façonner NORMiL ! 🚀**
