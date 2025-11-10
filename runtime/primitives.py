"""
NORMiL Primitives
=================

Auteur : Diego Morales Magri

Implémentation des primitives natives NORMiL en Python.

Catégories:
- Primitives vectorielles: zeros, ones, random, dot, norm, add, sub, etc.
- Primitives mémoire: episodic_append, episodic_query, semantic_upsert, etc.
- Primitives apprentissage: lowrankupdate, quantize, consolidate
- Primitives audit: audit_log, audit_verify
- Primitives utilitaires: generate_uuid, now, print, str
"""

import numpy as np
from typing import List, Dict, Any, Optional
from datetime import datetime
import uuid as uuid_lib

try:
    from .normil_types import (Vec, EpisodicRecord, Concept, WorkingMemoryEntry, 
                               IndexEntry, SafetyGuardrail, AuditLogEntry, 
                               InstinctPackage, ProtoInstinct, ImageTensor, 
                               AudioSegment, ModalityFusion, Rule, generate_uuid, now)
except ImportError:
    from normil_types import (Vec, EpisodicRecord, Concept, WorkingMemoryEntry,
                              IndexEntry, SafetyGuardrail, AuditLogEntry,
                              InstinctPackage, ProtoInstinct, ImageTensor,
                              AudioSegment, ModalityFusion, Rule, generate_uuid, now)


# ============================================
# Store global (simulation)
# ============================================

class GlobalStore:
    """Store global pour simuler la persistance"""
    episodic_log: List[EpisodicRecord] = []
    semantic_store: List[Concept] = []
    working_memory: List[WorkingMemoryEntry] = []
    audit_logs: List[Dict[str, Any]] = []

_store = GlobalStore()


# ============================================
# Plasticity Mode Registry (Phase 7.6)
# ============================================

class PlasticityModeRegistry:
    """Registry pour les modes de plasticité personnalisables"""
    
    def __init__(self):
        self._modes = {}
        self._register_builtin_modes()
    
    def _register_builtin_modes(self):
        """Enregistre les modes built-in"""
        self._modes['hebbian'] = {
            'normalize': True,
            'description': 'Hebbian learning - cells that fire together wire together'
        }
        self._modes['stdp'] = {
            'normalize': True,
            'description': 'Spike-Timing Dependent Plasticity'
        }
        self._modes['anti_hebbian'] = {
            'normalize': True,
            'description': 'Anti-Hebbian learning - decorrelation'
        }
    
    def register_mode(self, name: str, normalize: bool = False, 
                     update_fn: Optional[callable] = None,
                     description: str = ""):
        """
        Enregistre un nouveau mode de plasticité.
        
        Args:
            name: Nom du mode
            normalize: Si True, applique normalisation automatique
            update_fn: Fonction optionnelle pour update personnalisé
            description: Description du mode
        """
        if name in self._modes:
            raise ValueError(f"Plasticity mode '{name}' already registered")
        
        self._modes[name] = {
            'normalize': normalize,
            'update_fn': update_fn,
            'description': description
        }
    
    def get_mode(self, name: str) -> Dict[str, Any]:
        """Récupère les paramètres d'un mode"""
        if name not in self._modes:
            # Mode inconnu = pas de normalisation automatique
            return {'normalize': False, 'update_fn': None, 'description': 'Custom mode'}
        return self._modes[name]
    
    def list_modes(self) -> List[str]:
        """Liste tous les modes enregistrés"""
        return list(self._modes.keys())
    
    def should_normalize(self, mode: str) -> bool:
        """Vérifie si un mode requiert normalisation"""
        return self.get_mode(mode).get('normalize', False)

# Instance globale du registry
_plasticity_modes = PlasticityModeRegistry()


# ============================================
# Primitives Vectorielles
# ============================================

def vec(dim: int, values: List[float]) -> Vec:
    """
    Crée un vecteur à partir d'une dimension et d'une liste de valeurs.
    
    Args:
        dim: Dimension du vecteur
        values: Liste de valeurs (doit avoir exactement dim éléments)
    
    Returns:
        Vec: Vecteur créé
    
    Example:
        v = vec(3, [1.0, 2.0, 3.0])
    """
    if len(values) != dim:
        raise ValueError(f"Expected {dim} values, got {len(values)}")
    data = np.array(values, dtype=np.float16)
    return Vec(data, dim)


def zeros(dim: int) -> Vec:
    """Crée un vecteur de zéros"""
    return Vec.zeros(dim)


def ones(dim: int) -> Vec:
    """Crée un vecteur de uns"""
    return Vec.ones(dim)


def fill(dim: int, value: float) -> Vec:
    """Crée un vecteur rempli d'une valeur"""
    data = np.full(dim, value, dtype=np.float16)
    return Vec(data, dim)


def random(dim: int, mean: float = 0.0, std: float = 1.0) -> Vec:
    """Crée un vecteur aléatoire (distribution normale)"""
    return Vec.random(dim, mean, std)


def dot(v1: Vec, v2: Vec) -> float:
    """Produit scalaire de deux vecteurs"""
    if v1.dim != v2.dim:
        raise ValueError(f"Dimension mismatch: {v1.dim} vs {v2.dim}")
    return float(np.dot(v1.data, v2.data))


def norm(v: Vec) -> float:
    """Norme L2 d'un vecteur"""
    return float(np.linalg.norm(v.data))


def normalize(v: Vec) -> Vec:
    """Normalise un vecteur (norme L2 = 1)"""
    n = norm(v)
    if n == 0:
        return v
    data = v.data / n
    return Vec(data, v.dim, v.quantization)


def vec_add(v1: Vec, v2: Vec) -> Vec:
    """Addition de deux vecteurs"""
    if v1.dim != v2.dim:
        raise ValueError(f"Dimension mismatch: {v1.dim} vs {v2.dim}")
    data = v1.data + v2.data
    return Vec(data, v1.dim, v1.quantization)


def vec_sub(v1: Vec, v2: Vec) -> Vec:
    """Soustraction de deux vecteurs"""
    if v1.dim != v2.dim:
        raise ValueError(f"Dimension mismatch: {v1.dim} vs {v2.dim}")
    data = v1.data - v2.data
    return Vec(data, v1.dim, v1.quantization)


def vec_mul(v1: Vec, v2: Vec) -> Vec:
    """Multiplication élément par élément"""
    if v1.dim != v2.dim:
        raise ValueError(f"Dimension mismatch: {v1.dim} vs {v2.dim}")
    data = v1.data * v2.data
    return Vec(data, v1.dim, v1.quantization)


def scale(v: Vec, scalar: float) -> Vec:
    """Multiplication par un scalaire"""
    data = v.data * scalar
    return Vec(data, v.dim, v.quantization)


# ============================================
# Primitives Mémoire - Episodic Log
# ============================================

def episodic_append(record: EpisodicRecord) -> str:
    """Ajoute un enregistrement au log épisodique"""
    _store.episodic_log.append(record)
    audit_log("episodic_append", {"id": record.id, "summary": record.summary})
    return record.id


def episodic_query(vec: Vec, k: int = 10, filters: Optional[Dict[str, Any]] = None) -> List[EpisodicRecord]:
    """
    Recherche les k enregistrements les plus similaires à vec.
    
    Args:
        vec: Vecteur de requête
        k: Nombre de résultats
        filters: Filtres optionnels (ex: {"trust": 0.7})
    
    Returns:
        Liste d'EpisodicRecord triés par similarité
    """
    if not _store.episodic_log:
        return []
    
    # Calculer similarités
    similarities = []
    for record in _store.episodic_log:
        # Utiliser le premier vecteur disponible
        if "default" in record.vecs:
            rec_vec = record.vecs["default"]
        else:
            rec_vec = list(record.vecs.values())[0]
        
        # Similarité cosine
        if vec.dim == rec_vec.dim:
            sim = dot(vec, rec_vec) / (norm(vec) * norm(rec_vec))
            
            # Appliquer filtres
            if filters:
                if "trust" in filters and record.trust < filters["trust"]:
                    continue
            
            similarities.append((record, sim))
    
    # Trier par similarité décroissante
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    # Retourner top-k
    return [rec for rec, sim in similarities[:k]]


def episodic_get(id: str) -> Optional[EpisodicRecord]:
    """Récupère un enregistrement par ID"""
    for record in _store.episodic_log:
        if record.id == id:
            return record
    return None


# ============================================
# Primitives Mémoire - Working Memory
# ============================================

def wm_add(entry: WorkingMemoryEntry) -> None:
    """Ajoute une entrée à la working memory"""
    _store.working_memory.append(entry)


def wm_get(id: str) -> Optional[WorkingMemoryEntry]:
    """Récupère une entrée de working memory par ID"""
    for entry in _store.working_memory:
        if entry.id == id:
            return entry
    return None


def wm_query(vec: Vec, k: int = 10) -> List[WorkingMemoryEntry]:
    """Recherche dans la working memory"""
    if not _store.working_memory:
        return []
    
    similarities = []
    for entry in _store.working_memory:
        if vec.dim == entry.vec_combined.dim:
            sim = dot(vec, entry.vec_combined) / (norm(vec) * norm(entry.vec_combined))
            similarities.append((entry, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [entry for entry, sim in similarities[:k]]


# ============================================
# Primitives Mémoire - Semantic Store
# ============================================

def semantic_upsert(concept: Concept) -> None:
    """Ajoute ou met à jour un concept dans le store sémantique"""
    # Vérifier si le concept existe déjà
    for i, existing in enumerate(_store.semantic_store):
        if existing.concept_id == concept.concept_id:
            _store.semantic_store[i] = concept
            audit_log("semantic_update", {"id": concept.concept_id})
            return
    
    # Ajouter nouveau concept
    _store.semantic_store.append(concept)
    audit_log("semantic_insert", {"id": concept.concept_id})


def semantic_query(vec: Vec, k: int = 10) -> List[Concept]:
    """Recherche dans le store sémantique"""
    if not _store.semantic_store:
        return []
    
    similarities = []
    for concept in _store.semantic_store:
        if vec.dim == concept.centroid_vec.dim:
            sim = dot(vec, concept.centroid_vec) / (norm(vec) * norm(concept.centroid_vec))
            similarities.append((concept, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    return [concept for concept, sim in similarities[:k]]


def semantic_merge(c1: Concept, c2: Concept) -> Concept:
    """Fusionne deux concepts"""
    # Moyenne pondérée des centroides
    total_docs = c1.doc_count + c2.doc_count
    weight1 = c1.doc_count / total_docs
    weight2 = c2.doc_count / total_docs
    
    merged_vec_data = c1.centroid_vec.data * weight1 + c2.centroid_vec.data * weight2
    merged_vec = Vec(merged_vec_data, c1.centroid_vec.dim)
    
    # Fusionner labels
    merged_labels = list(set(c1.labels + c2.labels))
    
    # Moyenne trust
    merged_trust = (c1.trust_score * weight1 + c2.trust_score * weight2)
    
    return Concept.create(
        centroid=merged_vec,
        labels=merged_labels,
        trust=merged_trust
    )


# ============================================
# Primitives Apprentissage
# ============================================

def lowrankupdate(W: np.ndarray, u: Vec, v: Vec) -> np.ndarray:
    """
    Low-rank update: W' = W + u ⊗ v
    
    Pour la plasticité avec faible coût computationnel.
    """
    # u ⊗ v = outer product
    outer = np.outer(u.data, v.data)
    return W + outer


def quantize(v: Vec, bits: int) -> Vec:
    """Quantise un vecteur sur n bits"""
    # Quantisation simple: mapping [-range, range] -> [0, 2^bits - 1]
    data = v.data.copy()
    v_min, v_max = data.min(), data.max()
    
    if v_min == v_max:
        return v
    
    # Normaliser [0, 1]
    normalized = (data - v_min) / (v_max - v_min)
    
    # Quantiser
    levels = 2 ** bits
    quantized = np.round(normalized * (levels - 1)) / (levels - 1)
    
    # Dé-normaliser
    dequantized = quantized * (v_max - v_min) + v_min
    
    return Vec(dequantized.astype(np.float16), v.dim, bits)


def onlinecluster_update(centroid: Vec, x: Vec, lr: float = 0.1) -> Vec:
    """
    Mise à jour incrémentale d'un centroïde (clustering en ligne).
    
    Formule: centroid' = (1 - lr) * centroid + lr * x
    
    Args:
        centroid: Centroïde actuel
        x: Nouveau point à intégrer
        lr: Taux d'apprentissage (learning rate), contrôle la vitesse d'adaptation
    
    Returns:
        Vec: Nouveau centroïde après mise à jour
    
    Example:
        ```normil
        let c = zeros(128)
        let x = random(128)
        let c_new = onlinecluster_update(c, x, 0.1)
        ```
    """
    if centroid.dim != x.dim:
        raise ValueError(f"Dimension mismatch: centroid.dim={centroid.dim}, x.dim={x.dim}")
    
    if not (0.0 <= lr <= 1.0):
        raise ValueError(f"Learning rate must be in [0, 1], got {lr}")
    
    # Mise à jour incrémentale
    new_data = (1.0 - lr) * centroid.data + lr * x.data
    
    return Vec(new_data.astype(np.float16), centroid.dim)


def normalize_plasticity(weights: Vec) -> Vec:
    """
    Normalise les poids pour stabiliser la plasticité synaptique.
    
    Applique une normalisation L2 pour maintenir la norme constante.
    
    Args:
        weights: Vecteur de poids synaptiques
    
    Returns:
        Vec: Poids normalisés
    
    Example:
        ```normil
        let w = vec(3, [1.0, 2.0, 3.0])
        let w_norm = normalize_plasticity(w)
        // w_norm aura une norme L2 de 1.0
        ```
    """
    norm_val = np.linalg.norm(weights.data)
    
    # Éviter division par zéro (seuil adapté à float16)
    if norm_val < 1e-4:
        return weights
    
    normalized = weights.data / norm_val
    return Vec(normalized.astype(np.float16), weights.dim)


def decay_learning_rate(lr: float, factor: float = 0.99) -> float:
    """
    Applique un decay exponentiel au learning rate.
    
    Formule: lr' = lr * factor
    
    Args:
        lr: Learning rate actuel
        factor: Facteur de decay (0 < factor < 1)
    
    Returns:
        float: Nouveau learning rate
    
    Example:
        ```normil
        let lr = 0.1
        lr = decay_learning_rate(lr, 0.95)  // lr = 0.095
        lr = decay_learning_rate(lr, 0.95)  // lr = 0.09025
        ```
    """
    if not (0.0 < factor <= 1.0):
        raise ValueError(f"Decay factor must be in (0, 1], got {factor}")
    
    if not (0.0 <= lr <= 1.0):
        raise ValueError(f"Learning rate must be in [0, 1], got {lr}")
    
    return lr * factor


def compute_stability(weights_old: Vec, weights_new: Vec, threshold: float = 0.01) -> bool:
    """
    Calcule si les poids ont atteint la stabilité.
    
    Compare la différence relative entre anciens et nouveaux poids.
    Retourne True si la différence est sous le seuil.
    
    Args:
        weights_old: Poids avant mise à jour
        weights_new: Poids après mise à jour
        threshold: Seuil de stabilité (différence relative maximale)
    
    Returns:
        bool: True si stable, False sinon
    
    Example:
        ```normil
        let w_old = vec(3, [1.0, 2.0, 3.0])
        let w_new = vec(3, [1.001, 2.001, 3.001])
        let is_stable = compute_stability(w_old, w_new, 0.01)
        // is_stable = True (changement < 1%)
        ```
    """
    if weights_old.dim != weights_new.dim:
        raise ValueError(f"Dimension mismatch: {weights_old.dim} vs {weights_new.dim}")
    
    # Vérifier les poids quasi-nuls AVANT tout calcul (seuil adapté à float16)
    norm_old = np.linalg.norm(weights_old.data)
    if norm_old < 1e-4:
        return True  # Poids quasi-nuls considérés comme stables
    
    # Calculer la différence relative (suppression warning car on a déjà vérifié)
    diff = np.abs(weights_new.data - weights_old.data)
    with np.errstate(divide='ignore', invalid='ignore'):
        relative_change = np.linalg.norm(diff) / norm_old
    
    # Retourner un bool Python natif (pas np.bool_)
    return bool(relative_change < threshold)


def register_plasticity_mode(name: str, normalize: bool = False, description: str = "") -> bool:
    """
    Enregistre un nouveau mode de plasticité personnalisé.
    
    Args:
        name: Nom du mode (unique)
        normalize: Si True, applique normalisation L2 automatique
        description: Description optionnelle du mode
    
    Returns:
        bool: True si enregistré, False si déjà existant
    
    Exemple:
        ```normil
        // Enregistrer un mode custom
        let success = register_plasticity_mode("oja", true, "Oja's rule")
        
        @plastic(rate: 0.01, mode: "oja")
        fn oja_learn(x: Vec) -> Vec {
            // Implémentation custom
            return x
        }
        ```
    """
    try:
        _plasticity_modes.register_mode(name, normalize=normalize, description=description)
        return True
    except ValueError:
        return False


def list_plasticity_modes() -> List[str]:
    """
    Liste tous les modes de plasticité disponibles.
    
    Returns:
        List[str]: Noms des modes enregistrés
    
    Exemple:
        ```normil
        let modes = list_plasticity_modes()
        // modes = ["hebbian", "stdp", "anti_hebbian", ...]
        ```
    """
    return _plasticity_modes.list_modes()


def compute_stability_window(weight_history: List[Vec], threshold: float = 0.01) -> bool:
    """
    Vérifie la stabilité sur une fenêtre d'itérations (convergence soutenue).
    
    Args:
        weight_history: Liste de vecteurs de poids sur N dernières itérations
        threshold: Seuil de stabilité (défaut: 0.01 = 1%)
    
    Returns:
        bool: True si tous les changements consécutifs sont < threshold
    
    Exemple:
        ```normil
        let history = [w1, w2, w3, w4, w5]
        let stable = compute_stability_window(history, 0.01)
        // stable = true si tous les changements w[i] -> w[i+1] < 1%
        ```
    """
    if len(weight_history) < 2:
        return True  # Pas assez d'historique
    
    # Vérifier tous les changements consécutifs
    for i in range(len(weight_history) - 1):
        w_old = weight_history[i]
        w_new = weight_history[i + 1]
        
        if not compute_stability(w_old, w_new, threshold):
            return False  # Un changement trop grand
    
    return True  # Tous les changements sont petits


def compute_weight_variance(weight_history: List[Vec]) -> float:
    """
    Calcule la variance des poids sur une fenêtre d'itérations.
    
    Args:
        weight_history: Liste de vecteurs de poids
    
    Returns:
        float: Variance moyenne des composantes
    
    Exemple:
        ```normil
        let history = [w1, w2, w3, w4, w5]
        let var = compute_weight_variance(history)
        // var faible = poids stables
        ```
    """
    if len(weight_history) < 2:
        return 0.0
    
    # Convertir en matrice (iterations x dimensions)
    data = np.array([w.data for w in weight_history], dtype=np.float16)
    
    # Variance par dimension, puis moyenne
    variances = np.var(data, axis=0)
    mean_variance = float(np.mean(variances))
    
    return mean_variance


def consolidate(episodes: List[EpisodicRecord], method: str = "cluster") -> List[Concept]:
    """
    Consolide des épisodes en concepts sémantiques.
    
    Args:
        episodes: Liste d'enregistrements épisodiques
        method: "cluster" (centroid) ou "distill" (model distillation)
    
    Returns:
        Liste de concepts
    """
    if not episodes:
        return []
    
    if method == "cluster":
        # Simple clustering: calculer centroïde
        vecs = []
        for ep in episodes:
            if "default" in ep.vecs:
                vecs.append(ep.vecs["default"])
        
        if not vecs:
            return []
        
        # Centroïde
        centroid_data = np.mean([v.data for v in vecs], axis=0)
        centroid = Vec(centroid_data.astype(np.float16), vecs[0].dim)
        
        # Labels communs
        all_labels = []
        for ep in episodes:
            all_labels.extend([l.label for l in ep.labels])
        unique_labels = list(set(all_labels))
        
        # Trust moyen
        avg_trust = np.mean([ep.trust for ep in episodes])
        
        concept = Concept.create(
            centroid=centroid,
            labels=unique_labels,
            trust=float(avg_trust)
        )
        concept.doc_count = len(episodes)
        
        return [concept]
    
    elif method == "distill":
        # TODO: Distillation avec petit modèle
        raise NotImplementedError("Distillation not yet implemented")
    
    else:
        raise ValueError(f"Unknown consolidation method: {method}")


# ============================================================================
# Phase 7.9: Learning Rate Scheduling
# ============================================================================

def lr_warmup_linear(current_step: int, warmup_steps: int, target_lr: float) -> float:
    """
    Calcule le learning rate avec warmup linéaire.
    
    Pendant les warmup_steps premières itérations, le LR augmente linéairement
    de 0 à target_lr. Ensuite, il reste constant à target_lr.
    
    Args:
        current_step: Étape actuelle (commence à 0)
        warmup_steps: Nombre d'étapes de warmup
        target_lr: Learning rate cible après warmup
    
    Returns:
        Learning rate pour l'étape actuelle
    
    Example:
        ```normil
        let step = 5
        let lr = lr_warmup_linear(step, 10, 0.001)  // lr = 0.0005
        ```
    """
    if warmup_steps <= 0:
        return target_lr
    
    if current_step >= warmup_steps:
        return target_lr
    
    # Interpolation linéaire de 0 à target_lr
    return target_lr * (current_step / warmup_steps)


def lr_cosine_annealing(current_step: int, total_steps: int, min_lr: float, max_lr: float) -> float:
    """
    Calcule le learning rate avec cosine annealing.
    
    Le LR décroît selon une courbe cosinus de max_lr à min_lr sur total_steps.
    Cela permet une décroissance douce du learning rate.
    
    Args:
        current_step: Étape actuelle (commence à 0)
        total_steps: Nombre total d'étapes
        min_lr: Learning rate minimal
        max_lr: Learning rate maximal
    
    Returns:
        Learning rate pour l'étape actuelle
    
    Example:
        ```normil
        let step = 50
        let lr = lr_cosine_annealing(step, 100, 0.0001, 0.01)
        ```
    """
    if total_steps <= 0:
        return max_lr
    
    if current_step >= total_steps:
        return min_lr
    
    # Cosine annealing: cos décroît de 1 à -1, donc (1 + cos) décroît de 2 à 0
    import math
    progress = current_step / total_steps
    cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
    
    # Interpoler entre min_lr et max_lr
    return min_lr + (max_lr - min_lr) * cosine_decay


def lr_step_decay(current_step: int, initial_lr: float, decay_rate: float, decay_steps: int) -> float:
    """
    Calcule le learning rate avec step decay.
    
    Le LR est multiplié par decay_rate tous les decay_steps.
    C'est un decay par paliers (escalier).
    
    Args:
        current_step: Étape actuelle (commence à 0)
        initial_lr: Learning rate initial
        decay_rate: Facteur de multiplication (ex: 0.5 pour diviser par 2)
        decay_steps: Nombre d'étapes entre chaque decay
    
    Returns:
        Learning rate pour l'étape actuelle
    
    Example:
        ```normil
        let step = 25
        let lr = lr_step_decay(step, 0.01, 0.5, 10)  // Decay à step 10, 20, 30...
        ```
    """
    if decay_steps <= 0:
        return initial_lr
    
    # Nombre de fois qu'on a decay
    num_decays = current_step // decay_steps
    
    return initial_lr * (decay_rate ** num_decays)


def lr_plateau_factor(loss_history: List[float], patience: int, factor: float, threshold: float) -> float:
    """
    Détermine le facteur de réduction du LR basé sur plateau detection.
    
    Si la loss n'améliore pas de plus de threshold pendant patience itérations,
    retourne factor (pour multiplier le LR). Sinon retourne 1.0 (pas de changement).
    
    Args:
        loss_history: Historique des losses (ordre chronologique)
        patience: Nombre d'itérations sans amélioration avant réduction
        factor: Facteur de réduction (ex: 0.5 pour diviser par 2)
        threshold: Amélioration minimale requise (ex: 0.01 pour 1%)
    
    Returns:
        1.0 si pas de plateau, factor si plateau détecté
    
    Example:
        ```normil
        let losses = [0.5, 0.49, 0.48, 0.48, 0.48, 0.48]
        let reduction = lr_plateau_factor(losses, 3, 0.5, 0.01)  // 0.5 (plateau)
        ```
    """
    if len(loss_history) < patience + 1:
        return 1.0  # Pas assez d'historique
    
    # Prendre les patience+1 dernières losses
    recent = loss_history[-(patience + 1):]
    
    # Meilleure loss dans la fenêtre (min)
    best_loss = min(recent[:-1])
    current_loss = recent[-1]
    
    # Amélioration relative
    if best_loss > 0:
        improvement = (best_loss - current_loss) / abs(best_loss)
    else:
        improvement = 0.0
    
    # Si amélioration < threshold, on a un plateau
    if improvement < threshold:
        return factor
    
    return 1.0


# ============================================
# Primitives Audit
# ============================================

def audit_log(action: str, data: Dict[str, Any], level: str = "info") -> None:
    """Enregistre une action dans l'audit log"""
    log_entry = {
        "timestamp": now(),
        "action": action,
        "data": data,
        "level": level
    }
    _store.audit_logs.append(log_entry)


def audit_verify(from_timestamp: float, to_timestamp: float) -> bool:
    """Vérifie l'intégrité des logs d'audit"""
    # Pour l'instant, simple vérification que les logs existent
    relevant_logs = [
        log for log in _store.audit_logs
        if from_timestamp <= log["timestamp"] <= to_timestamp
    ]
    return len(relevant_logs) > 0


def audit_snapshot(name: str) -> str:
    """Crée un snapshot de l'état actuel"""
    import hashlib
    import json
    
    state = {
        "name": name,
        "timestamp": now(),
        "episodic_count": len(_store.episodic_log),
        "semantic_count": len(_store.semantic_store),
        "wm_count": len(_store.working_memory)
    }
    
    # Hash du snapshot
    state_json = json.dumps(state, sort_keys=True)
    snapshot_hash = hashlib.sha256(state_json.encode()).hexdigest()
    
    audit_log("snapshot_created", {"name": name, "hash": snapshot_hash})
    return snapshot_hash


# ============================================
# Primitives Utilitaires
# ============================================

def normil_print(value: Any) -> None:
    """Print pour NORMiL"""
    print(value)


def normil_str(value: Any) -> str:
    """Convertit en string"""
    return str(value)


def normil_len(value: Any) -> int:
    """Retourne la longueur"""
    return len(value)


def normil_range(start: int, end: int, step: int = 1) -> List[int]:
    """Génère une range"""
    return list(range(start, end, step))


# ============================================
# Primitives String (Phase 3.3)
# ============================================

def string_length(s: str) -> int:
    """Retourne la longueur d'une string"""
    return len(s)


def string_upper(s: str) -> str:
    """Convertit en majuscules"""
    return s.upper()


def string_lower(s: str) -> str:
    """Convertit en minuscules"""
    return s.lower()


def string_substring(s: str, start: int, end: int) -> str:
    """Extrait une sous-chaîne (slicing)"""
    return s[start:end]


def string_split(s: str, delimiter: str = " ") -> List[str]:
    """Sépare une string selon un délimiteur"""
    return s.split(delimiter)


def string_join(parts: List[str], separator: str = "") -> str:
    """Joint une liste de strings"""
    return separator.join(parts)


def string_replace(s: str, old: str, new: str) -> str:
    """Remplace toutes les occurrences"""
    return s.replace(old, new)


def string_contains(s: str, substring: str) -> bool:
    """Vérifie si contient une sous-chaîne"""
    return substring in s


def string_startswith(s: str, prefix: str) -> bool:
    """Vérifie si commence par un préfixe"""
    return s.startswith(prefix)


def string_endswith(s: str, suffix: str) -> bool:
    """Vérifie si termine par un suffixe"""
    return s.endswith(suffix)


def string_trim(s: str) -> str:
    """Enlève les espaces au début et à la fin"""
    return s.strip()


def string_repeat(s: str, count: int) -> str:
    """Répète une string n fois"""
    return s * count


def string_char_at(s: str, index: int) -> str:
    """Retourne le caractère à l'index"""
    if 0 <= index < len(s):
        return s[index]
    raise IndexError(f"String index out of range: {index}")


def string_index_of(s: str, substring: str) -> int:
    """Retourne l'index de la première occurrence (-1 si absent)"""
    try:
        return s.index(substring)
    except ValueError:
        return -1


# ============================================
# Registry des primitives
# ============================================

PRIMITIVES = {
    # Vectorielles
    "vec": vec,
    "zeros": zeros,
    "ones": ones,
    "fill": fill,
    "random": random,
    "dot": dot,
    "norm": norm,
    "normalize": normalize,
    "vec_add": vec_add,
    "vec_sub": vec_sub,
    "vec_mul": vec_mul,
    "scale": scale,
    
    # Episodic
    "episodic_append": episodic_append,
    "episodic_query": episodic_query,
    "episodic_get": episodic_get,
    
    # Working Memory
    "wm_add": wm_add,
    "wm_get": wm_get,
    "wm_query": wm_query,
    
    # Semantic
    "semantic_upsert": semantic_upsert,
    "semantic_query": semantic_query,
    "semantic_merge": semantic_merge,
    
    # Apprentissage
    "lowrankupdate": lowrankupdate,
    "quantize": quantize,
    "onlinecluster_update": onlinecluster_update,
    "consolidate": consolidate,
    
    # Plasticité (Phase 7)
    "normalize_plasticity": normalize_plasticity,
    "decay_learning_rate": decay_learning_rate,
    "compute_stability": compute_stability,
    "register_plasticity_mode": register_plasticity_mode,  # Phase 7.6
    "list_plasticity_modes": list_plasticity_modes,        # Phase 7.6
    "compute_stability_window": compute_stability_window,  # Phase 7.8
    "compute_weight_variance": compute_weight_variance,    # Phase 7.8
    
    # Learning Rate Scheduling (Phase 7.9)
    "lr_warmup_linear": lr_warmup_linear,
    "lr_cosine_annealing": lr_cosine_annealing,
    "lr_step_decay": lr_step_decay,
    "lr_plateau_factor": lr_plateau_factor,
    
    # Audit
    "audit_log": audit_log,
    "audit_verify": audit_verify,
    "audit_snapshot": audit_snapshot,
    
    # Utilitaires
    "print": normil_print,
    "str": normil_str,
    "to_string": normil_str,  # Alias pour éviter conflit avec le type str
    "len": normil_len,
    "range": normil_range,
    "generate_uuid": generate_uuid,
    "now": now,
    
    # String operations (Phase 3.3)
    "string_length": string_length,
    "string_upper": string_upper,
    "string_lower": string_lower,
    "string_substring": string_substring,
    "string_split": string_split,
    "string_join": string_join,
    "string_replace": string_replace,
    "string_contains": string_contains,
    "string_startswith": string_startswith,
    "string_endswith": string_endswith,
    "string_trim": string_trim,
    "string_repeat": string_repeat,
    "string_char_at": string_char_at,
    "string_index_of": string_index_of,
}


# ============================================
# Primitives O-RedMind Phase 8
# ============================================

# Index & Retrieval
# ------------------

def fastindex_query(vec: Vec, k: int = 10, filters: Optional[Dict[str, str]] = None) -> List[IndexEntry]:
    """
    Top-k retrieval rapide avec filtres optionnels.
    
    Utilise une recherche combinée : cache RAM + HNSW disk.
    Complexité : O(log N) avec HNSW
    """
    from normil_types import IndexEntry
    
    # Pour l'instant, implémentation simple linéaire
    # TODO Phase 8.2 : Implémenter vrai HNSW
    candidates = []
    
    # Recherche dans tous les index entries du store
    if not hasattr(_store, 'index_entries'):
        _store.index_entries = []
    
    for entry in _store.index_entries:
        # Appliquer filtres
        if filters:
            match = all(
                entry.metadata.get(k) == v 
                for k, v in filters.items()
            )
            if not match:
                continue
        
        # Calculer distance
        dist = np.linalg.norm(vec.data - entry.vec.data)
        candidates.append((dist, entry))
    
    # Trier et retourner top-k
    candidates.sort(key=lambda x: x[0])
    return [entry for _, entry in candidates[:k]]


def hnsw_insert(vec: Vec, metadata: Optional[Dict[str, str]] = None, layer: int = 0) -> IndexEntry:
    """
    Insert un vecteur dans l'index HNSW.
    
    HNSW (Hierarchical Navigable Small World) est un graphe multi-couches
    pour recherche de plus proches voisins en temps logarithmique.
    """
    from normil_types import IndexEntry
    
    if not hasattr(_store, 'index_entries'):
        _store.index_entries = []
    
    # Créer nouvelle entrée
    entry = IndexEntry.create(vec, metadata or {}, layer)
    
    # Connecter aux voisins (pour l'instant, simple k-NN)
    # TODO Phase 8.2 : Vrai algorithme HNSW avec construction hiérarchique
    k_neighbors = 5
    distances = []
    
    for existing in _store.index_entries:
        if existing.layer == layer:
            dist = np.linalg.norm(vec.data - existing.vec.data)
            distances.append((dist, existing.id))
    
    # Connecter aux k plus proches
    distances.sort(key=lambda x: x[0])
    for dist, neighbor_id in distances[:k_neighbors]:
        entry.add_neighbor(neighbor_id, dist)
    
    # Ajouter au store
    _store.index_entries.append(entry)
    
    audit_log("hnsw_insert", {"entry_id": entry.id, "layer": layer})
    
    return entry


def bloom_contains(key: str, bloom_filter: Optional[Any] = None) -> bool:
    """
    Vérifie si une clé existe probablement dans un Bloom filter.
    
    Bloom filter = structure probabiliste pour test d'appartenance.
    Peut avoir des faux positifs, mais jamais de faux négatifs.
    """
    import hashlib
    
    if bloom_filter is None:
        # Utiliser un bloom filter global simple
        if not hasattr(_store, 'bloom_bits'):
            _store.bloom_bits = set()
            _store.bloom_size = 10000
        bloom_filter = _store.bloom_bits
    
    # Hash la clé avec plusieurs fonctions
    hash1 = int(hashlib.md5(key.encode()).hexdigest(), 16) % _store.bloom_size
    hash2 = int(hashlib.sha1(key.encode()).hexdigest(), 16) % _store.bloom_size
    hash3 = int(hashlib.sha256(key.encode()).hexdigest(), 16) % _store.bloom_size
    
    # Check si tous les bits sont présents
    return all(h in bloom_filter for h in [hash1, hash2, hash3])


def bloom_add(key: str, bloom_filter: Optional[Any] = None):
    """Ajoute une clé au Bloom filter"""
    import hashlib
    
    if bloom_filter is None:
        if not hasattr(_store, 'bloom_bits'):
            _store.bloom_bits = set()
            _store.bloom_size = 10000
        bloom_filter = _store.bloom_bits
    
    hash1 = int(hashlib.md5(key.encode()).hexdigest(), 16) % _store.bloom_size
    hash2 = int(hashlib.sha1(key.encode()).hexdigest(), 16) % _store.bloom_size
    hash3 = int(hashlib.sha256(key.encode()).hexdigest(), 16) % _store.bloom_size
    
    bloom_filter.update([hash1, hash2, hash3])


def lru_cache_get(key: str, cache: Optional[Dict] = None) -> Optional[Any]:
    """
    Récupère une valeur du cache LRU.
    
    LRU = Least Recently Used eviction policy.
    """
    if cache is None:
        if not hasattr(_store, 'lru_cache'):
            _store.lru_cache = {}
            _store.lru_order = []
            _store.lru_max_size = 1000
        cache = _store.lru_cache
    
    if key in cache:
        # Mettre à jour l'ordre d'accès
        if key in _store.lru_order:
            _store.lru_order.remove(key)
        _store.lru_order.append(key)
        
        return cache[key]
    
    return None


def lru_cache_put(key: str, value: Any, cache: Optional[Dict] = None):
    """Insère une valeur dans le cache LRU avec éviction si nécessaire"""
    if cache is None:
        if not hasattr(_store, 'lru_cache'):
            _store.lru_cache = {}
            _store.lru_order = []
            _store.lru_max_size = 1000
        cache = _store.lru_cache
    
    # Éviction si plein
    if len(cache) >= _store.lru_max_size and key not in cache:
        # Évince le moins récemment utilisé
        if _store.lru_order:
            lru_key = _store.lru_order.pop(0)
            del cache[lru_key]
    
    # Insertion
    cache[key] = value
    if key in _store.lru_order:
        _store.lru_order.remove(key)
    _store.lru_order.append(key)


def rerank_neural(candidates: List[IndexEntry], query: Vec, model: Optional[Any] = None) -> List[IndexEntry]:
    """
    Re-scoring neural des candidats pour améliorer la pertinence.
    
    Utilise un petit modèle neural pour affiner les distances.
    """
    # Pour l'instant, simple re-scoring par similarité cosinus
    # TODO Phase 8.3 : Vrai modèle neural de re-ranking
    
    scored = []
    query_norm = np.linalg.norm(query.data)
    
    for entry in candidates:
        entry_norm = np.linalg.norm(entry.vec.data)
        
        # Similarité cosinus
        if query_norm > 0 and entry_norm > 0:
            similarity = np.dot(query.data, entry.vec.data) / (query_norm * entry_norm)
        else:
            similarity = 0.0
        
        scored.append((similarity, entry))
    
    # Trier par similarité décroissante
    scored.sort(key=lambda x: x[0], reverse=True)
    
    return [entry for _, entry in scored]


# Safety & Governance
# --------------------

def check_guardrail(action: str, context: Dict[str, Any]) -> Optional[str]:
    """
    Vérifie si une action viole un guardrail de sécurité.
    
    Returns:
        None si OK, message d'erreur si violation
    """
    from normil_types import SafetyGuardrail
    
    if not hasattr(_store, 'guardrails'):
        _store.guardrails = []
    
    for guardrail in _store.guardrails:
        # Évaluer la condition (simple pour l'instant)
        # TODO Phase 8.2 : Parser d'expressions booléennes complet
        
        # Check simple : si action correspond
        if guardrail.action_blocked in action or guardrail.action_blocked == "*":
            audit_log("guardrail_check", {
                "action": action,
                "guardrail": guardrail.id,
                "blocked": True
            }, level="warning")
            
            if guardrail.require_consent:
                return f"Guardrail '{guardrail.id}' requires consent for action '{action}'"
            else:
                return f"Guardrail '{guardrail.id}' blocks action '{action}'"
    
    return None


def add_guardrail(guardrail: 'SafetyGuardrail'):
    """Ajoute un guardrail au système"""
    if not hasattr(_store, 'guardrails'):
        _store.guardrails = []
    
    _store.guardrails.append(guardrail)
    audit_log("guardrail_added", {"id": guardrail.id})


def require_consent(action: str, reason: str, data_accessed: List[str], 
                   expiry_ttl: int = 3600000) -> bool:
    """
    Requête de consentement utilisateur pour une action.
    
    Returns:
        True si consentement accordé, False sinon
    """
    from normil_types import ConsentRequest
    
    request = ConsentRequest(
        action=action,
        reason=reason,
        data_accessed=data_accessed,
        expiry_ttl=expiry_ttl
    )
    
    # Pour l'instant, log la requête
    # TODO Phase 8.3 : Vrai système de consentement avec UI
    audit_log("consent_request", {
        "action": action,
        "reason": reason,
        "data_count": len(data_accessed)
    }, level="info")
    
    # Retourner False par défaut (sécurité)
    return False


def audit_append(event_type: str, actor: str, action: str, data: Any) -> str:
    """
    Append une entrée au journal d'audit avec hash chaining.
    
    Returns:
        ID de l'entrée créée
    """
    from normil_types import AuditLogEntry
    
    if not hasattr(_store, 'audit_chain'):
        _store.audit_chain = []
    
    # Récupérer le hash précédent
    prev_hash = "0" * 64
    if _store.audit_chain:
        prev_hash = _store.audit_chain[-1].compute_hash()
    
    # Créer nouvelle entrée
    entry = AuditLogEntry.create(
        event_type=event_type,
        actor=actor,
        action=action,
        data=data,
        prev_hash=prev_hash
    )
    
    _store.audit_chain.append(entry)
    
    return entry.id


def hash_chain_verify() -> bool:
    """
    Vérifie l'intégrité de la chaîne d'audit.
    
    Returns:
        True si intègre, False si corruption détectée
    """
    if not hasattr(_store, 'audit_chain'):
        return True
    
    if len(_store.audit_chain) == 0:
        return True
    
    # Vérifier chaque lien de la chaîne
    for i in range(1, len(_store.audit_chain)):
        prev_entry = _store.audit_chain[i - 1]
        curr_entry = _store.audit_chain[i]
        
        expected_hash = prev_entry.compute_hash()
        if curr_entry.prev_hash != expected_hash:
            audit_log("chain_corruption", {
                "index": i,
                "expected": expected_hash,
                "actual": curr_entry.prev_hash
            }, level="error")
            return False
    
    return True


def rollback_to_snapshot(snapshot_hash: str) -> bool:
    """
    Rollback l'état du système à un snapshot précédent.
    
    Returns:
        True si succès, False si snapshot introuvable
    """
    # TODO Phase 8.3 : Implémenter vrai système de snapshots
    
    audit_log("rollback_attempted", {"snapshot_hash": snapshot_hash}, level="warning")
    
    # Pour l'instant, juste logger
    return False


# Instinct Core
# --------------

def score_prototypes(vec: Vec, prototypes: List['ProtoInstinct']) -> List[tuple]:
    """
    Score un vecteur contre une liste de prototypes d'instinct.
    
    Returns:
        Liste de (prototype_id, score) triée par score décroissant
    """
    scores = []
    
    query_norm = np.linalg.norm(vec.data)
    
    for proto in prototypes:
        proto_norm = np.linalg.norm(proto.vec_ref.data)
        
        # Similarité cosinus pondérée
        if query_norm > 0 and proto_norm > 0:
            similarity = np.dot(vec.data, proto.vec_ref.data) / (query_norm * proto_norm)
            weighted_score = similarity * proto.weight
        else:
            weighted_score = 0.0
        
        scores.append((proto.id, weighted_score))
    
    # Trier par score décroissant
    scores.sort(key=lambda x: x[1], reverse=True)
    
    return scores


def sign_package(package: 'InstinctPackage', private_key: str) -> 'InstinctPackage':
    """
    Signe cryptographiquement un package d'instinct.
    
    TODO Phase 8.2 : Vraie signature RSA/Ed25519
    """
    import hashlib
    
    # Pour l'instant, simple hash
    data = f"{package.package_id}{package.version}{private_key}".encode()
    signature = hashlib.sha256(data).hexdigest()
    
    package.signature = signature
    
    audit_log("package_signed", {
        "package_id": package.package_id,
        "version": package.version
    })
    
    return package


def verify_signature(package: 'InstinctPackage', public_key: str) -> bool:
    """
    Vérifie la signature d'un package.
    
    Returns:
        True si signature valide, False sinon
    """
    # TODO Phase 8.2 : Vraie vérification cryptographique
    
    # Pour l'instant, juste vérifier que signature existe
    is_valid = len(package.signature) == 64
    
    audit_log("signature_verified", {
        "package_id": package.package_id,
        "valid": is_valid
    })
    
    return is_valid


def validate_overlay(overlay: 'InstinctOverlay', tests: List[Any]) -> bool:
    """
    Valide un overlay d'instinct contre une batterie de tests.
    
    Returns:
        True si tous les tests passent, False sinon
    """
    # TODO Phase 8.3 : Vrai système de validation avec sandbox
    
    audit_log("overlay_validation", {
        "provenance": overlay.provenance,
        "test_count": len(tests)
    })
    
    # Pour l'instant, toujours True
    return True


# Consolidation
# --------------

def priority_sample(episodes: List['EpisodicRecord'], k: int, 
                   priority_fn: Optional[callable] = None) -> List['EpisodicRecord']:
    """
    Échantillonne k épisodes selon une fonction de priorité.
    
    Utilisé pour replay priorisé dans consolidation.
    """
    if priority_fn is None:
        # Priorité par défaut : récence + trust
        def default_priority(ep):
            recency = 1.0 / (now() - ep.timestamp + 1.0)
            return recency * ep.trust
        priority_fn = default_priority
    
    # Calculer priorités
    scored = [(priority_fn(ep), ep) for ep in episodes]
    
    # Softmax sampling (approximation)
    import random
    scored.sort(key=lambda x: x[0], reverse=True)
    
    # Prendre top-k
    return [ep for _, ep in scored[:k]]


def distill_to_semantic(episodes: List['EpisodicRecord']) -> 'Concept':
    """
    Distille plusieurs épisodes en un concept sémantique.
    
    Calcule le centroïde et crée un concept compressé.
    """
    from normil_types import Concept
    
    if not episodes:
        raise ValueError("Cannot distill empty episode list")
    
    # Récupérer tous les vecteurs
    vecs = []
    for ep in episodes:
        # Prendre le premier vecteur disponible
        if ep.vecs:
            first_key = list(ep.vecs.keys())[0]
            vecs.append(ep.vecs[first_key].data)
    
    if not vecs:
        raise ValueError("No vectors found in episodes")
    
    # Calculer centroïde
    centroid_data = np.mean(vecs, axis=0).astype(np.float16)
    centroid = Vec(centroid_data, len(centroid_data))
    
    # Collecter labels
    all_labels = []
    for ep in episodes:
        all_labels.extend([label.label for label in ep.labels])
    unique_labels = list(set(all_labels))
    
    # Trust moyen
    avg_trust = np.mean([ep.trust for ep in episodes])
    
    # Créer concept
    concept = Concept.create(
        centroid=centroid,
        labels=unique_labels,
        trust=float(avg_trust)
    )
    concept.doc_count = len(episodes)
    
    audit_log("distill_to_semantic", {
        "episode_count": len(episodes),
        "concept_id": concept.concept_id
    })
    
    return concept


def cluster_centroids(vecs: List[Vec], k: int, max_iter: int = 10) -> List[Vec]:
    """
    K-means clustering pour trouver k centroides.
    
    Version online pour consolidation incrémentale.
    """
    if len(vecs) < k:
        return vecs
    
    # Initialisation : k premiers vecteurs
    centroids = [Vec(vecs[i].data.copy(), vecs[i].dim) for i in range(k)]
    
    # K-means iterations
    for _ in range(max_iter):
        # Assignment
        clusters = [[] for _ in range(k)]
        
        for vec in vecs:
            # Trouver centroïde le plus proche
            distances = [np.linalg.norm(vec.data - c.data) for c in centroids]
            closest = np.argmin(distances)
            clusters[closest].append(vec)
        
        # Update centroids
        for i, cluster in enumerate(clusters):
            if cluster:
                new_centroid = np.mean([v.data for v in cluster], axis=0).astype(np.float16)
                centroids[i] = Vec(new_centroid, len(new_centroid))
    
    return centroids


def forgetting_policy(memory: 'EpisodicRecord', age: float, utility: float, 
                     threshold: float = 0.1) -> bool:
    """
    Décide si un souvenir doit être oublié.
    
    Args:
        memory: Enregistrement épisodique
        age: Âge en secondes
        utility: Score d'utilité [0, 1]
        threshold: Seuil en dessous duquel on oublie
    
    Returns:
        True si doit être oublié, False sinon
    """
    # Fonction de déclin temporel
    age_days = age / 86400.0
    decay = np.exp(-age_days / 30.0)  # Déclin exponentiel (30 jours half-life)
    
    # Score combiné
    retention_score = utility * decay * memory.trust
    
    should_forget = retention_score < threshold
    
    if should_forget:
        audit_log("forgetting_policy", {
            "memory_id": memory.id,
            "age_days": age_days,
            "utility": utility,
            "retention_score": retention_score,
            "decision": "forget"
        })
    
    return should_forget


# ============================================
# Phase 8.2 - Multimodal & Perception Primitives
# ============================================

def embed_image(img: ImageTensor, model: str = "mobilenet") -> Vec:
    """Encode une image en vecteur d'embedding
    
    Args:
        img: ImageTensor à encoder
        model: Modèle d'encodage ("mobilenet", "resnet", "vit")
    
    Returns:
        Vec d'embedding (dim dépend du modèle)
    """
    # Simulation: moyenne des pixels + projection
    # En production: appel à un vrai modèle CNN/ViT
    
    # Normalisation [0, 1]
    img_normalized = (img.data - img.data.min()) / (img.data.max() - img.data.min() + 1e-8)
    
    # Features simples: moyenne par canal + stats spatiales
    channel_means = np.mean(img_normalized, axis=(0, 1))  # (channels,)
    spatial_std = np.std(img_normalized, axis=2)  # (H, W)
    spatial_features = np.array([
        spatial_std.mean(),
        spatial_std.std(),
        spatial_std.max(),
        spatial_std.min()
    ])
    
    # Dimensions par modèle
    model_dims = {
        "mobilenet": 512,
        "resnet": 2048,
        "vit": 768
    }
    
    dim = model_dims.get(model, 512)
    
    # Combinaison features + padding/projection
    basic_features = np.concatenate([channel_means, spatial_features])
    
    # Projection vers dimension cible (simulation simple)
    if len(basic_features) < dim:
        # Padding + random projection
        embedding = np.zeros(dim)
        embedding[:len(basic_features)] = basic_features
        # Ajouter variance contrôlée pour simuler features CNN
        embedding[len(basic_features):] = np.random.randn(dim - len(basic_features)) * 0.1
    else:
        embedding = basic_features[:dim]
    
    audit_log("embed_image", {
        "model": model,
        "img_shape": f"{img.height}x{img.width}x{img.channels}",
        "embedding_dim": dim
    })
    
    return Vec.from_list(embedding.tolist())


def embed_audio(audio: AudioSegment, model: str = "wavenet") -> Vec:
    """Encode un segment audio en vecteur d'embedding
    
    Args:
        audio: AudioSegment à encoder
        model: Modèle d'encodage ("wavenet", "wav2vec", "hubert")
    
    Returns:
        Vec d'embedding
    """
    # Simulation: features audio basiques + projection
    # En production: appel à wav2vec2, HuBERT, etc.
    
    # Features temporelles
    samples = audio.samples
    
    # Stats basiques
    features = np.array([
        np.mean(samples),
        np.std(samples),
        np.max(samples),
        np.min(samples),
        np.median(samples),
    ])
    
    # Features spectrales (simulation FFT simple)
    fft = np.fft.fft(samples)
    fft_mag = np.abs(fft[:len(fft)//2])  # Magnitude spectrum
    
    # Bandes de fréquence (simulation mel-bands)
    num_bands = 20
    band_size = len(fft_mag) // num_bands
    mel_bands = np.array([
        np.mean(fft_mag[i*band_size:(i+1)*band_size]) 
        for i in range(num_bands)
    ])
    
    # Dimensions par modèle
    model_dims = {
        "wavenet": 512,
        "wav2vec": 768,
        "hubert": 1024
    }
    
    dim = model_dims.get(model, 512)
    
    # Combinaison
    basic_features = np.concatenate([features, mel_bands])
    
    # Projection
    if len(basic_features) < dim:
        embedding = np.zeros(dim)
        embedding[:len(basic_features)] = basic_features
        embedding[len(basic_features):] = np.random.randn(dim - len(basic_features)) * 0.1
    else:
        embedding = basic_features[:dim]
    
    audit_log("embed_audio", {
        "model": model,
        "duration": audio.duration,
        "sample_rate": audio.sample_rate,
        "embedding_dim": dim
    })
    
    return Vec.from_list(embedding.tolist())


def temporal_align(streams: Dict[str, Vec], window_ms: int = 500) -> ModalityFusion:
    """Aligne temporellement plusieurs flux de modalités
    
    Args:
        streams: Dict {"modality_name": Vec_embedding, ...}
        window_ms: Fenêtre temporelle d'alignement (millisecondes)
    
    Returns:
        ModalityFusion avec embeddings alignés
    """
    # Simulation: calcul de scores d'alignement basés sur similarité
    alignment_scores = {}
    
    # Reference: première modalité ou moyenne
    ref_vec = list(streams.values())[0] if streams else None
    
    for modality, vec in streams.items():
        if ref_vec is not None:
            # Score d'alignement = similarité cosine avec référence
            similarity = dot(vec, ref_vec) / (norm(vec) * norm(ref_vec) + 1e-8)
            # Clamper la similarité pour éviter dépassement dû aux erreurs d'arrondi
            similarity = max(-1.0, min(1.0, similarity))
            # Normaliser [0, 1]
            alignment_score = (similarity + 1.0) / 2.0
        else:
            alignment_score = 1.0
        
        alignment_scores[modality] = alignment_score
    
    audit_log("temporal_align", {
        "modalities": list(streams.keys()),
        "window_ms": window_ms,
        "alignment_scores": alignment_scores
    })
    
    return ModalityFusion.create(
        modalities=streams,
        alignment_scores=alignment_scores,
        metadata={"window_ms": window_ms}
    )


def cross_attention(query: Vec, key: Vec, value: Vec, num_heads: int = 8) -> Vec:
    """Applique un mécanisme de cross-attention entre modalités
    
    Args:
        query: Vec query (ex: embedding image)
        key: Vec key (ex: embedding audio)
        value: Vec value (même que key généralement)
        num_heads: Nombre de têtes d'attention
    
    Returns:
        Vec résultat de l'attention
    """
    # Simulation simple de multi-head attention
    # En production: vraie implémentation transformer
    
    # Assurer dimensions compatibles
    dim = min(query.dim, key.dim, value.dim)
    
    q = query.data[:dim]
    k = key.data[:dim]
    v = value.data[:dim]
    
    # Attention scores (softmax de similarités)
    # Score = softmax(Q·K^T / sqrt(d_k))
    attention_score = np.dot(q, k) / np.sqrt(dim)
    attention_weight = 1.0 / (1.0 + np.exp(-attention_score))  # Sigmoid (simplification)
    
    # Attention output = attention_weight * V
    output = attention_weight * v
    
    # Multi-head simulation: moyenne de plusieurs projections
    # (simplification extrême)
    for _ in range(num_heads - 1):
        # Projections aléatoires
        random_proj = np.random.randn(dim, dim) * 0.1
        q_proj = np.dot(random_proj, q)
        k_proj = np.dot(random_proj, k)
        v_proj = np.dot(random_proj, v)
        
        score = np.dot(q_proj, k_proj) / np.sqrt(dim)
        weight = 1.0 / (1.0 + np.exp(-score))
        output += weight * v_proj
    
    output = output / num_heads  # Moyenne
    
    audit_log("cross_attention", {
        "query_dim": query.dim,
        "key_dim": key.dim,
        "num_heads": num_heads,
        "attention_weight": float(attention_weight)
    })
    
    return Vec.from_list(output.tolist())


def fusion_concat(vecs: List[Vec]) -> Vec:
    """Fusionne plusieurs vecteurs par concaténation
    
    Args:
        vecs: Liste de Vec à fusionner
    
    Returns:
        Vec concaténé
    """
    if not vecs:
        return Vec.zeros(0)
    
    concatenated = np.concatenate([v.data for v in vecs])
    
    audit_log("fusion_concat", {
        "num_vecs": len(vecs),
        "dims": [v.dim for v in vecs],
        "output_dim": len(concatenated)
    })
    
    return Vec.from_list(concatenated.tolist())


def fusion_weighted(vecs: List[Vec], weights: Vec) -> Vec:
    """Fusionne plusieurs vecteurs par somme pondérée
    
    Args:
        vecs: Liste de Vec à fusionner
        weights: Vec de poids (doit avoir dim = len(vecs))
    
    Returns:
        Vec fusionné
    """
    if not vecs:
        return Vec.zeros(0)
    
    if weights.dim != len(vecs):
        raise ValueError(f"Weights dim ({weights.dim}) must equal number of vecs ({len(vecs)})")
    
    # Normaliser les poids
    w = weights.data / (np.sum(weights.data) + 1e-8)
    
    # Assurer que tous les vecs ont même dimension
    target_dim = vecs[0].dim
    
    # Somme pondérée
    result = np.zeros(target_dim)
    for i, vec in enumerate(vecs):
        if vec.dim != target_dim:
            raise ValueError(f"All vecs must have same dim, got {vec.dim} and {target_dim}")
        result += w[i] * vec.data
    
    audit_log("fusion_weighted", {
        "num_vecs": len(vecs),
        "weights": w.tolist(),
        "output_dim": target_dim
    })
    
    return Vec.from_list(result.tolist())


def vision_patch_extract(img: ImageTensor, patch_size: int = 16) -> List[Vec]:
    """Extrait des patches d'une image (Vision Transformer style)
    
    Args:
        img: ImageTensor source
        patch_size: Taille des patches (carrés)
    
    Returns:
        Liste de Vec, chaque Vec = un patch aplati
    """
    patches = []
    
    # Nombre de patches
    num_patches_h = img.height // patch_size
    num_patches_w = img.width // patch_size
    
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            # Extraire patch
            patch = img.data[
                i*patch_size:(i+1)*patch_size,
                j*patch_size:(j+1)*patch_size,
                :
            ]
            
            # Aplatir (flatten)
            patch_flat = patch.flatten()
            patches.append(Vec.from_list(patch_flat.tolist()))
    
    audit_log("vision_patch_extract", {
        "img_shape": f"{img.height}x{img.width}x{img.channels}",
        "patch_size": patch_size,
        "num_patches": len(patches),
        "patch_dim": patches[0].dim if patches else 0
    })
    
    return patches


def audio_spectrogram(audio: AudioSegment, n_fft: int = 512, hop_length: int = 256) -> ImageTensor:
    """Calcule un spectrogramme (représentation temps-fréquence)
    
    Args:
        audio: AudioSegment source
        n_fft: Taille de la FFT
        hop_length: Pas de déplacement de la fenêtre
    
    Returns:
        ImageTensor représentant le spectrogramme
    """
    samples = audio.samples
    
    # Nombre de fenêtres
    num_frames = (len(samples) - n_fft) // hop_length + 1
    
    # Calculer FFT pour chaque fenêtre
    spectrogram = np.zeros((n_fft // 2, num_frames))
    
    for i in range(num_frames):
        start = i * hop_length
        end = start + n_fft
        
        if end > len(samples):
            break
        
        # Fenêtre de signal
        window = samples[start:end]
        
        # FFT
        fft = np.fft.fft(window)
        magnitude = np.abs(fft[:n_fft//2])
        
        spectrogram[:, i] = magnitude
    
    # Log scale (dB)
    spectrogram_db = 20 * np.log10(spectrogram + 1e-8)
    
    # Normaliser [0, 1]
    spectrogram_norm = (spectrogram_db - spectrogram_db.min()) / (spectrogram_db.max() - spectrogram_db.min() + 1e-8)
    
    # Convertir en ImageTensor (grayscale)
    # Transposer pour avoir (time, freq) -> (height=freq, width=time)
    img_data = spectrogram_norm[:, :, np.newaxis]  # Ajouter dimension channel
    
    audit_log("audio_spectrogram", {
        "duration": audio.duration,
        "n_fft": n_fft,
        "hop_length": hop_length,
        "spectrogram_shape": f"{img_data.shape[0]}x{img_data.shape[1]}"
    })
    
    return ImageTensor.from_array(img_data, metadata={
        "type": "spectrogram",
        "n_fft": n_fft,
        "hop_length": hop_length,
        "sample_rate": audio.sample_rate
    })


# ============================================
# Phase 8.3 - Reasoner Hybride
# ============================================

def symbolic_match(context: Dict[str, Any], rules: List[Any]) -> List[Any]:
    """Pattern matching symbolique sur contexte
    
    Évalue des règles symboliques contre un contexte et retourne
    les règles qui matchent.
    
    Args:
        context: Dict de variables et valeurs
        rules: Liste de règles avec conditions
    
    Returns:
        Liste de règles qui matchent (triées par priorité)
    """
    matched_rules = []
    
    for rule in rules:
        # Règle peut être un objet Rule ou un dict
        if hasattr(rule, 'condition'):
            condition = rule.condition
            priority = getattr(rule, 'priority', 0)
            rule_id = getattr(rule, 'id', 'unknown')
        elif isinstance(rule, dict):
            condition = rule.get('condition', 'True')
            priority = rule.get('priority', 0)
            rule_id = rule.get('id', 'unknown')
        else:
            continue
        
        # Évaluation sécurisée de la condition
        try:
            # Variables disponibles dans la condition
            eval_context = {
                **context,
                'Vec': Vec,
                'len': len,
                'abs': abs,
                'min': min,
                'max': max,
                'sum': sum,
            }
            
            # Évaluer condition (sandbox simple)
            result = eval(condition, {"__builtins__": {}}, eval_context)
            
            if result:
                matched_rules.append({
                    'rule': rule,
                    'priority': priority,
                    'id': rule_id
                })
        except Exception as e:
            # Condition invalide, skip
            audit_log("symbolic_match_error", {
                "rule_id": rule_id,
                "error": str(e)
            })
            continue
    
    # Trier par priorité (descendant)
    matched_rules.sort(key=lambda x: x['priority'], reverse=True)
    
    audit_log("symbolic_match", {
        "num_rules": len(rules),
        "num_matched": len(matched_rules),
        "top_match": matched_rules[0]['id'] if matched_rules else None
    })
    
    # Retourner les règles matchées
    return [m['rule'] for m in matched_rules]


def neural_shortpass(vec: Vec, model: str, context_vec: Vec) -> tuple:
    """Inférence rapide (neural shortpass)
    
    Traitement neural rapide pour décisions simples.
    Latence faible (~10-50ms), précision moyenne.
    
    Args:
        vec: Vecteur d'entrée
        model: Modèle à utiliser ("tinynet", "mobilenet")
        context_vec: Contexte récent (working memory)
    
    Returns:
        (output_vec, confidence_score)
    """
    # Simulation: combinaison simple input + context
    # En production: vraie inférence avec modèle léger
    
    # Assurer dimensions compatibles
    dim = min(vec.dim, context_vec.dim)
    
    # Combinaison pondérée
    alpha = 0.7  # Poids pour input
    beta = 0.3   # Poids pour context
    
    combined = np.zeros(dim)
    combined += alpha * vec.data[:dim]
    combined += beta * context_vec.data[:dim]
    
    # Transformation simple (simulation layer linéaire + activation)
    # W = random matrix (simulé par hash du model name)
    np.random.seed(hash(model) % (2**32))
    W = np.random.randn(dim, dim) * 0.1
    
    output = np.tanh(np.dot(W, combined))
    
    # Confidence = cosine similarity avec input (heuristique)
    similarity = np.dot(output, vec.data[:dim]) / (np.linalg.norm(output) * np.linalg.norm(vec.data[:dim]) + 1e-8)
    confidence = (similarity + 1.0) / 2.0  # Normaliser [0, 1]
    
    audit_log("neural_shortpass", {
        "model": model,
        "input_dim": vec.dim,
        "confidence": float(confidence),
        "latency_ms": 15  # Simulé
    })
    
    return (Vec.from_list(output.tolist()), float(confidence))


def neural_longpass(vec: Vec, model: str, retrieved: List[Any]) -> tuple:
    """Inférence profonde avec retrieval (neural longpass)
    
    Traitement neural profond avec accès à mémoire épisodique.
    Latence plus élevée (~100-500ms), précision élevée.
    
    Args:
        vec: Vecteur d'entrée
        model: Modèle à utiliser ("deepnet", "transformer")
        retrieved: Records épisodiques récupérés
    
    Returns:
        (output_vec, trace_log: Dict)
    """
    # Simulation: agrégation des retrieved + processing profond
    # En production: vraie inférence transformer avec retrieved context
    
    # Extraire vecteurs des records
    context_vecs = []
    for record in retrieved:
        if hasattr(record, 'vec'):
            context_vecs.append(record.vec)
        elif hasattr(record, 'centroid_vec'):
            context_vecs.append(record.centroid_vec)
    
    # Mean pooling des context
    if context_vecs:
        dim = min(vec.dim, *[v.dim for v in context_vecs])
        context_pooled = np.mean([v.data[:dim] for v in context_vecs], axis=0)
    else:
        dim = vec.dim
        context_pooled = np.zeros(dim)
    
    # Simulation layers profondes
    # Layer 1
    np.random.seed(hash(model + "_layer1") % (2**32))
    W1 = np.random.randn(dim, dim) * 0.1
    h1 = np.tanh(np.dot(W1, vec.data[:dim]) + 0.2 * context_pooled)
    
    # Layer 2
    np.random.seed(hash(model + "_layer2") % (2**32))
    W2 = np.random.randn(dim, dim) * 0.1
    h2 = np.tanh(np.dot(W2, h1))
    
    # Layer 3 (output)
    np.random.seed(hash(model + "_layer3") % (2**32))
    W3 = np.random.randn(dim, dim) * 0.1
    output = np.tanh(np.dot(W3, h2))
    
    # Trace log pour auditabilité
    trace_log = {
        "model": model,
        "input_dim": vec.dim,
        "num_retrieved": len(retrieved),
        "layers": ["input", "dense_1", "dense_2", "output"],
        "latency_ms": 150,  # Simulé
        "activation_norms": [
            float(np.linalg.norm(h1)),
            float(np.linalg.norm(h2)),
            float(np.linalg.norm(output))
        ]
    }
    
    audit_log("neural_longpass", trace_log)
    
    return (Vec.from_list(output.tolist()), trace_log)


def meta_controller_decide(vec: Vec, cost_budget: float, latency_target_ms: int) -> str:
    """Meta-controller: décide entre shortpass et longpass
    
    Orchestration du reasoner hybride basée sur:
    - Budget computationnel
    - Contrainte de latence
    - Complexité estimée de l'input
    
    Args:
        vec: Vecteur d'entrée
        cost_budget: Budget compute [0, 1] (0=minimal, 1=maximal)
        latency_target_ms: Latence cible en millisecondes
    
    Returns:
        "shortpass" ou "longpass"
    """
    # Heuristiques pour décision
    
    # 1. Complexité de l'input (variance des composantes)
    variance = float(np.var(vec.data))
    complexity_score = min(1.0, variance / 2.0)  # Normaliser
    
    # 2. Contrainte latence
    if latency_target_ms < 50:
        latency_pressure = 1.0  # Force shortpass
    elif latency_target_ms > 200:
        latency_pressure = 0.0  # Permet longpass
    else:
        latency_pressure = (200 - latency_target_ms) / 150.0
    
    # 3. Budget disponible
    budget_pressure = 1.0 - cost_budget
    
    # Score combiné (weighted sum)
    shortpass_score = (
        0.4 * latency_pressure +
        0.3 * budget_pressure +
        0.3 * (1.0 - complexity_score)
    )
    
    # Décision avec seuil
    decision = "shortpass" if shortpass_score > 0.5 else "longpass"
    
    audit_log("meta_controller_decide", {
        "decision": decision,
        "complexity_score": complexity_score,
        "latency_pressure": latency_pressure,
        "budget_pressure": budget_pressure,
        "shortpass_score": shortpass_score,
        "cost_budget": cost_budget,
        "latency_target_ms": latency_target_ms
    })
    
    return decision


# ============================================
# Phase 8.4 - Dev Tools & Introspection
# ============================================

def introspect_type(obj: Any) -> Dict[str, Any]:
    """
    Introspect un objet NORMiL et retourne ses métadonnées complètes.
    
    Args:
        obj: Objet NORMiL (Vec, EpisodicRecord, Concept, etc.)
        
    Returns:
        Dict contenant type_name, fields, methods, metadata
        
    Exemples:
        >>> v = Vec([1.0, 2.0, 3.0])
        >>> info = introspect_type(v)
        >>> info["type_name"]  # "Vec"
        >>> info["dimension"]  # 3
    """
    
    info = {
        "type_name": type(obj).__name__,
        "module": type(obj).__module__,
        "fields": {},
        "methods": [],
        "metadata": {}
    }
    
    # Introspection selon le type
    if isinstance(obj, Vec):
        info["fields"] = {
            "data": f"List[{len(obj.data)}]",
            "dimension": len(obj.data)
        }
        info["metadata"] = {
            "norm": norm(obj),
            "mean": sum(obj.data) / len(obj.data) if len(obj.data) > 0 else 0.0,
            "min": min(obj.data) if len(obj.data) > 0 else 0.0,
            "max": max(obj.data) if len(obj.data) > 0 else 0.0
        }
        info["methods"] = ["__add__", "__mul__", "__getitem__", "to_list"]
        
    elif isinstance(obj, EpisodicRecord):
        first_vec = list(obj.vecs.values())[0] if obj.vecs and len(obj.vecs) > 0 else None
        info["fields"] = {
            "id": obj.id,
            "timestamp": obj.timestamp,
            "summary": obj.summary[:50] + "..." if len(obj.summary) > 50 else obj.summary,
            "vec_dim": len(first_vec.data) if first_vec else 0,
            "trust": obj.trust
        }
        info["metadata"] = {
            "has_provenance": bool(obj.provenance),
            "has_labels": len(obj.labels) > 0 if obj.labels else False,
            "num_vecs": len(obj.vecs) if obj.vecs else 0,
            "vec_keys": list(obj.vecs.keys()) if obj.vecs else [],
            "sources": obj.sources if obj.sources else []
        }
        info["methods"] = ["create", "from_dict", "to_dict"]
        
    elif isinstance(obj, Concept):
        info["fields"] = {
            "concept_id": obj.concept_id,
            "centroid_dim": len(obj.centroid_vec.data) if obj.centroid_vec else 0,
            "labels": obj.labels,
            "trust": obj.trust_score,
            "count": obj.doc_count
        }
        info["metadata"] = {
            "has_provenance": len(obj.provenance_versions) > 0 if obj.provenance_versions else False,
            "centroid_norm": norm(obj.centroid_vec) if obj.centroid_vec else 0.0,
            "num_labels": len(obj.labels) if obj.labels else 0
        }
        info["methods"] = ["create", "from_dict", "to_dict"]
        
    elif isinstance(obj, Rule):
        info["fields"] = {
            "id": obj.id,
            "condition": obj.condition[:50] + "..." if obj.condition and len(obj.condition) > 50 else obj.condition,
            "action": obj.action[:50] + "..." if len(obj.action) > 50 else obj.action,
            "priority": obj.priority
        }
        info["metadata"] = {
            "has_condition": bool(obj.condition),
            "condition_length": len(obj.condition) if obj.condition else 0
        }
        info["methods"] = []
        
    elif hasattr(obj, '__dict__'):
        # Objet générique avec __dict__
        info["fields"] = {k: type(v).__name__ if not isinstance(v, (str, int, float, bool, type(None))) else v 
                         for k, v in obj.__dict__.items()}
        info["metadata"] = {
            "num_fields": len(obj.__dict__),
            "field_types": {k: type(v).__name__ for k, v in obj.__dict__.items()}
        }
        info["methods"] = [m for m in dir(obj) if callable(getattr(obj, m)) and not m.startswith('_')]
        
    return info


def trace_execution(code: str, context: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    Trace l'exécution d'un morceau de code NORMiL et retourne le log détaillé.
    
    Args:
        code: Code NORMiL à tracer
        context: Variables contextuelles optionnelles
        
    Returns:
        Dict contenant result, trace_log, execution_time_ms, calls
        
    Exemples:
        >>> trace = trace_execution("v = random(10); norm(v)")
        >>> trace["execution_time_ms"]  # ~0.5
        >>> len(trace["calls"])  # 2 (random + norm)
    """
    import time
    
    trace_log = []
    start_time = time.time()
    
    ctx = context or {}
    
    # Hook pour tracker les appels
    original_primitives = {}
    
    def make_tracer(name: str, func):
        def traced(*args, **kwargs):
            call_start = time.time()
            result = func(*args, **kwargs)
            call_time = (time.time() - call_start) * 1000
            trace_log.append({
                "function": name,
                "args_types": [type(a).__name__ for a in args],
                "time_ms": round(call_time, 3),
                "result_type": type(result).__name__
            })
            return result
        return traced
    
    # Wrapper temporaire des primitives
    traced_primitives = {}
    for name, func in PRIMITIVES.items():
        traced_primitives[name] = make_tracer(name, func)
    
    # Contexte d'exécution avec primitives tracées
    exec_context = {**traced_primitives, **ctx}
    
    try:
        # Exécution
        result = eval(code, exec_context)
        execution_time = (time.time() - start_time) * 1000
        
        return {
            "result": result,
            "result_type": type(result).__name__,
            "trace_log": trace_log,
            "execution_time_ms": round(execution_time, 3),
            "calls": len(trace_log),
            "success": True
        }
    except Exception as e:
        execution_time = (time.time() - start_time) * 1000
        return {
            "result": None,
            "result_type": "Error",
            "trace_log": trace_log,
            "execution_time_ms": round(execution_time, 3),
            "calls": len(trace_log),
            "success": False,
            "error": str(e)
        }


def get_signature(primitive_name: str) -> Dict[str, Any]:
    """
    Retourne la signature d'une primitive (args, return type, documentation).
    
    Args:
        primitive_name: Nom de la primitive
        
    Returns:
        Dict contenant name, args, return_type, doc, category
        
    Exemples:
        >>> sig = get_signature("dot")
        >>> sig["args"]  # ["a: Vec", "b: Vec"]
        >>> sig["return_type"]  # "Float"
    """
    if primitive_name not in PRIMITIVES:
        return {
            "name": primitive_name,
            "found": False,
            "error": f"Primitive '{primitive_name}' not found"
        }
    
    func = PRIMITIVES[primitive_name]
    
    # Extraction de la signature depuis la docstring
    doc = func.__doc__ or ""
    lines = [l.strip() for l in doc.split('\n') if l.strip()]
    
    # Parse Args et Returns depuis docstring
    args_section = []
    return_section = []
    in_args = False
    in_returns = False
    
    for line in lines:
        if line.startswith("Args:"):
            in_args = True
            in_returns = False
            continue
        elif line.startswith("Returns:"):
            in_returns = True
            in_args = False
            continue
        elif line.startswith("Exemples:") or line.startswith("Examples:"):
            break
            
        if in_args and line and not line.startswith(">>>"):
            args_section.append(line)
        elif in_returns and line and not line.startswith(">>>"):
            return_section.append(line)
    
    # Déterminer la catégorie
    category = "unknown"
    if "episodic" in primitive_name:
        category = "episodic"
    elif "semantic" in primitive_name:
        category = "semantic"
    elif "consolidate" in primitive_name or "cluster" in primitive_name:
        category = "consolidation"
    elif "embed" in primitive_name or "fusion" in primitive_name or "vision" in primitive_name or "audio" in primitive_name:
        category = "multimodal"
    elif "symbolic" in primitive_name or "neural" in primitive_name or "meta_controller" in primitive_name:
        category = "reasoner"
    elif "hnsw" in primitive_name or "index" in primitive_name or "fastindex" in primitive_name or "bloom" in primitive_name or "lru" in primitive_name or "rerank" in primitive_name:
        category = "indexing"
    elif primitive_name in ["dot", "norm", "distance", "similarity", "zeros", "ones", "random"]:
        category = "vector"
    elif "lowrank" in primitive_name or "quantize" in primitive_name:
        category = "neural"
    elif "normalize" in primitive_name or "decay" in primitive_name or "stability" in primitive_name:
        category = "plasticity"
    elif "guardrail" in primitive_name or "consent" in primitive_name or "audit" in primitive_name or "rollback" in primitive_name or "hash_chain" in primitive_name:
        category = "safety"
    elif "prototypes" in primitive_name or "sign_package" in primitive_name or "verify_package" in primitive_name:
        category = "instinct"
    elif "introspect" in primitive_name or "trace" in primitive_name or "signature" in primitive_name or "list_primitives" in primitive_name or "viz" in primitive_name:
        category = "devtools"
    
    return {
        "name": primitive_name,
        "found": True,
        "args": args_section,
        "return_type": ' '.join(return_section) if return_section else "Unknown",
        "doc": lines[0] if lines else "",
        "category": category,
        "full_doc": doc
    }


def list_primitives(category: str = None) -> List[str]:
    """
    Liste toutes les primitives disponibles, optionnellement filtrées par catégorie.
    
    Args:
        category: Catégorie optionnelle ("vector", "episodic", "semantic", 
                  "consolidation", "multimodal", "reasoner", "indexing", 
                  "neural", "plasticity", "safety", "instinct", "devtools")
        
    Returns:
        Liste des noms de primitives
        
    Exemples:
        >>> all_prims = list_primitives()
        >>> len(all_prims)  # ~85+
        >>> vec_prims = list_primitives("vector")
        >>> "dot" in vec_prims  # True
    """
    if category is None:
        return sorted(PRIMITIVES.keys())
    
    # Filtrer par catégorie
    filtered = []
    for name in PRIMITIVES.keys():
        sig = get_signature(name)
        if sig.get("category") == category:
            filtered.append(name)
    
    return sorted(filtered)


def viz_vec_space(vectors: List[Vec], labels: List[str] = None, method: str = "pca") -> Dict[str, Any]:
    """
    Visualise un espace vectoriel en 2D/3D (PCA ou t-SNE).
    
    Args:
        vectors: Liste de vecteurs à visualiser
        labels: Labels optionnels pour chaque vecteur
        method: Méthode de réduction ("pca" ou "tsne")
        
    Returns:
        Dict contenant coordinates_2d, labels, method, explained_variance
        
    Exemples:
        >>> vecs = [random(128) for _ in range(50)]
        >>> viz = viz_vec_space(vecs, method="pca")
        >>> len(viz["coordinates_2d"])  # 50
    """
    if not vectors:
        return {"error": "No vectors provided", "coordinates_2d": []}
    
    # Conversion en matrice numpy-like
    dim = len(vectors[0].data)
    n = len(vectors)
    
    # PCA simple (2 composantes principales)
    if method == "pca":
        # Centre les données
        mean_vec = [sum(vectors[i].data[j] for i in range(n)) / n for j in range(dim)]
        centered = [[vectors[i].data[j] - mean_vec[j] for j in range(dim)] for i in range(n)]
        
        # Calcul simplifié : projection sur 2 axes principaux (variance max)
        # Axe 1 : direction de variance maximale
        coords_2d = []
        for i in range(n):
            # Projection simplifiée (sum et variance comme proxy)
            x = sum(centered[i][:dim//2]) / (dim//2)
            y = sum(centered[i][dim//2:]) / (dim - dim//2)
            coords_2d.append([x, y])
        
        # Variance expliquée (approximation)
        var_x = sum(c[0]**2 for c in coords_2d) / n
        var_y = sum(c[1]**2 for c in coords_2d) / n
        total_var = sum(sum(v**2 for v in row) for row in centered) / n
        explained = (var_x + var_y) / total_var if total_var > 0 else 0.0
        
        return {
            "coordinates_2d": coords_2d,
            "labels": labels or [f"vec_{i}" for i in range(n)],
            "method": "pca",
            "explained_variance": round(explained, 3),
            "n_vectors": n,
            "dimension": dim
        }
    
    elif method == "tsne":
        # t-SNE simplifié : utilise les distances pour positionner
        import random as py_random
        
        # Initialisation aléatoire
        coords_2d = [[py_random.gauss(0, 0.1), py_random.gauss(0, 0.1)] for _ in range(n)]
        
        # Quelques itérations de gradient descent simplifié
        for iteration in range(20):
            for i in range(n):
                grad_x, grad_y = 0.0, 0.0
                for j in range(n):
                    if i == j:
                        continue
                    # Distance haute dimension (euclidienne)
                    diff = [vectors[i].data[k] - vectors[j].data[k] for k in range(dim)]
                    dist_high = sum(d**2 for d in diff)**0.5
                    # Distance basse dimension
                    dx = coords_2d[i][0] - coords_2d[j][0]
                    dy = coords_2d[i][1] - coords_2d[j][1]
                    dist_low = (dx**2 + dy**2)**0.5 + 1e-8
                    
                    # Gradient simplifié
                    factor = (dist_high - dist_low) / dist_low
                    grad_x += factor * dx
                    grad_y += factor * dy
                
                # Update
                lr = 0.1 / (1 + iteration * 0.1)
                coords_2d[i][0] -= lr * grad_x / n
                coords_2d[i][1] -= lr * grad_y / n
        
        return {
            "coordinates_2d": coords_2d,
            "labels": labels or [f"vec_{i}" for i in range(n)],
            "method": "tsne",
            "iterations": 20,
            "n_vectors": n,
            "dimension": dim
        }
    
    else:
        return {"error": f"Unknown method: {method}", "coordinates_2d": []}


def viz_attention(query: Vec, keys: List[Vec], values: List[Vec] = None, num_heads: int = 1) -> Dict[str, Any]:
    """
    Visualise les poids d'attention entre query et keys.
    
    Args:
        query: Vecteur query
        keys: Liste de vecteurs keys
        values: Vecteurs values optionnels (sinon keys=values)
        num_heads: Nombre de têtes d'attention
        
    Returns:
        Dict contenant attention_weights, output, head_contributions
        
    Exemples:
        >>> q = random(64)
        >>> ks = [random(64) for _ in range(10)]
        >>> viz = viz_attention(q, ks)
        >>> sum(viz["attention_weights"])  # ~1.0
    """
    import math
    
    if values is None:
        values = keys
    
    n_keys = len(keys)
    dim = len(query.data)
    
    if n_keys == 0:
        return {"error": "No keys provided", "attention_weights": []}
    
    # Calcul des scores d'attention (dot product)
    scores = [dot(query, k) for k in keys]
    
    # Softmax
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    attention_weights = [e / sum_exp for e in exp_scores]
    
    # Output pondéré
    output_data = [0.0] * dim
    for i, weight in enumerate(attention_weights):
        for j in range(dim):
            output_data[j] += weight * values[i].data[j]
    output = Vec.from_list(output_data)
    
    # Contributions par tête (si multi-head)
    head_dim = dim // num_heads
    head_contributions = []
    
    if num_heads > 1:
        for h in range(num_heads):
            start = h * head_dim
            end = start + head_dim
            head_scores = [sum(query.data[start:end][i] * keys[k].data[start:end][i] 
                              for i in range(head_dim)) 
                          for k in range(n_keys)]
            head_max = max(head_scores)
            head_exp = [math.exp(s - head_max) for s in head_scores]
            head_sum = sum(head_exp)
            head_weights = [e / head_sum for e in head_exp]
            head_contributions.append({
                "head": h,
                "weights": head_weights,
                "entropy": -sum(w * math.log(w + 1e-10) for w in head_weights)
            })
    
    return {
        "attention_weights": attention_weights,
        "output": output,
        "num_keys": n_keys,
        "num_heads": num_heads,
        "head_contributions": head_contributions,
        "entropy": -sum(w * math.log(w + 1e-10) for w in attention_weights),
        "max_weight": max(attention_weights),
        "max_index": attention_weights.index(max(attention_weights))
    }


def viz_trace(trace_log: List[Dict[str, Any]]) -> str:
    """
    Formate un trace log en string lisible (format arbre).
    
    Args:
        trace_log: Log de trace (depuis trace_execution)
        
    Returns:
        String formatée du trace
        
    Exemples:
        >>> trace = trace_execution("norm(random(10))")
        >>> print(viz_trace(trace["trace_log"]))
        # Affiche:
        # └─ random [0.123ms] -> Vec
        # └─ norm [0.045ms] -> float
    """
    if not trace_log:
        return "(empty trace)"
    
    lines = []
    total_time = sum(entry.get("time_ms", 0) for entry in trace_log)
    
    lines.append(f"Trace ({len(trace_log)} calls, {total_time:.3f}ms total)")
    lines.append("=" * 60)
    
    for i, entry in enumerate(trace_log):
        func_name = entry.get("function", "unknown")
        time_ms = entry.get("time_ms", 0)
        result_type = entry.get("result_type", "?")
        args_types = entry.get("args_types", [])
        
        # Symbole
        symbol = "+-" if i == len(trace_log) - 1 else "|-"
        
        # Formatage
        args_str = ", ".join(args_types) if args_types else ""
        time_pct = (time_ms / total_time * 100) if total_time > 0 else 0
        
        line = f"{symbol} {func_name}({args_str}) [{time_ms:.3f}ms, {time_pct:.1f}%] -> {result_type}"
        lines.append(line)
    
    return "\n".join(lines)


# Enregistrer toutes les nouvelles primitives
PRIMITIVES.update({
    # Index & Retrieval
    "fastindex_query": fastindex_query,
    "hnsw_insert": hnsw_insert,
    "bloom_contains": bloom_contains,
    "bloom_add": bloom_add,
    "lru_cache_get": lru_cache_get,
    "lru_cache_put": lru_cache_put,
    "rerank_neural": rerank_neural,
    
    # Safety & Governance
    "check_guardrail": check_guardrail,
    "add_guardrail": add_guardrail,
    "require_consent": require_consent,
    "audit_append": audit_append,
    "hash_chain_verify": hash_chain_verify,
    "rollback_to_snapshot": rollback_to_snapshot,
    
    # Instinct Core
    "score_prototypes": score_prototypes,
    "sign_package": sign_package,
    "verify_signature": verify_signature,
    "validate_overlay": validate_overlay,
    
    # Consolidation
    "priority_sample": priority_sample,
    "distill_to_semantic": distill_to_semantic,
    "cluster_centroids": cluster_centroids,
    "forgetting_policy": forgetting_policy,
    
    # Phase 8.2 - Multimodal & Perception
    "embed_image": embed_image,
    "embed_audio": embed_audio,
    "temporal_align": temporal_align,
    "cross_attention": cross_attention,
    "fusion_concat": fusion_concat,
    "fusion_weighted": fusion_weighted,
    "vision_patch_extract": vision_patch_extract,
    "audio_spectrogram": audio_spectrogram,
    
    # Phase 8.3 - Reasoner Hybride
    "symbolic_match": symbolic_match,
    "neural_shortpass": neural_shortpass,
    "neural_longpass": neural_longpass,
    "meta_controller_decide": meta_controller_decide,
    
    # Phase 8.4 - Dev Tools
    "introspect_type": introspect_type,
    "trace_execution": trace_execution,
    "get_signature": get_signature,
    "list_primitives": list_primitives,
    "viz_vec_space": viz_vec_space,
    "viz_attention": viz_attention,
    "viz_trace": viz_trace,
})


# ============================================
# Tests
# ============================================

if __name__ == '__main__':
    print("=== Test NORMiL Primitives ===\n")
    
    # Test vectorielles
    print("1. Primitives Vectorielles:")
    v1 = zeros(128)
    v2 = random(128, mean=0.0, std=1.0)
    v3 = v1 + v2  # Utilise Vec.__add__
    print(f"   zeros(128): {v1}")
    print(f"   random(128): {v2}")
    print(f"   v1 + v2: {v3}")
    print(f"   dot(v2, v2): {dot(v2, v2):.3f}")
    print(f"   norm(v2): {norm(v2):.3f}\n")
    
    # Test episodic
    print("2. Primitives Episodic:")
    vec = random(128)
    record = EpisodicRecord.create("Test memory", vec, trust=0.9)
    id1 = episodic_append(record)
    print(f"   Appended: {id1}")
    
    results = episodic_query(vec, k=5)
    print(f"   Query results: {len(results)} records")
    print(f"   First result: {results[0] if results else 'None'}\n")
    
    # Test semantic
    print("3. Primitives Semantic:")
    centroid = random(128)
    concept = Concept.create(centroid, labels=["test"], trust=0.85)
    semantic_upsert(concept)
    print(f"   Upserted concept: {concept.concept_id[:8]}...")
    
    concepts = semantic_query(centroid, k=5)
    print(f"   Query results: {len(concepts)} concepts\n")
    
    # Test consolidation
    print("4. Consolidation:")
    episodes = [
        EpisodicRecord.create(f"Memory {i}", random(128), trust=0.8 + i*0.05)
        for i in range(3)
    ]
    for ep in episodes:
        episodic_append(ep)
    
    consolidated = consolidate(episodes, method="cluster")
    print(f"   Consolidated {len(episodes)} episodes into {len(consolidated)} concepts")
    if consolidated:
        print(f"   Concept: {consolidated[0]}\n")
    
    # Test audit
    print("5. Audit:")
    audit_log("test_action", {"key": "value"}, level="info")
    snapshot_hash = audit_snapshot("test_snapshot")
    print(f"   Snapshot hash: {snapshot_hash[:16]}...")
    print(f"   Total audit logs: {len(_store.audit_logs)}\n")
    
    # Test Phase 8 : Index & Retrieval
    print("6. Phase 8 - Index & Retrieval:")
    from normil_types import IndexEntry
    
    # HNSW insert
    vec1 = random(128)
    entry1 = hnsw_insert(vec1, {"type": "test", "id": "1"}, layer=0)
    print(f"   HNSW inserted: {entry1}")
    
    # Insert more entries
    for i in range(5):
        v = random(128)
        hnsw_insert(v, {"type": "test", "id": str(i+2)}, layer=0)
    
    # FastIndex query
    query_vec = random(128)
    results = fastindex_query(query_vec, k=3, filters={"type": "test"})
    print(f"   Query results: {len(results)} entries")
    
    # LRU cache
    lru_cache_put("key1", "value1")
    lru_cache_put("key2", "value2")
    cached = lru_cache_get("key1")
    print(f"   LRU cache get: {cached}")
    
    # Bloom filter
    bloom_add("test_key")
    exists = bloom_contains("test_key")
    not_exists = bloom_contains("nonexistent")
    print(f"   Bloom filter: exists={exists}, not_exists={not_exists}\n")
    
    # Test Phase 8 : Safety & Governance
    print("7. Phase 8 - Safety & Governance:")
    from normil_types import SafetyGuardrail
    
    # Add guardrail
    guardrail = SafetyGuardrail.create(
        id="no_delete",
        condition="action == 'delete'",
        action_blocked="delete",
        require_consent=True
    )
    add_guardrail(guardrail)
    
    # Check guardrail
    violation = check_guardrail("delete_file", {"file": "test.txt"})
    print(f"   Guardrail check: {violation}")
    
    # Audit chain
    id1 = audit_append("test_event", "system", "test_action", {"key": "value"})
    id2 = audit_append("test_event_2", "user", "read", {"file": "data.txt"})
    chain_valid = hash_chain_verify()
    print(f"   Audit chain valid: {chain_valid}")
    print(f"   Audit entries: {len(_store.audit_chain) if hasattr(_store, 'audit_chain') else 0}\n")
    
    # Test Phase 8 : Instinct Core
    print("8. Phase 8 - Instinct Core:")
    from normil_types import ProtoInstinct, InstinctPackage, MetaParams, InstinctCore, InstinctOverlay, ValidationManifest
    
    # Create prototypes
    proto1 = ProtoInstinct.create("safety", random(128), weight=1.5)
    proto2 = ProtoInstinct.create("attention", random(128), weight=1.0)
    
    # Score prototypes
    test_vec = random(128)
    scores = score_prototypes(test_vec, [proto1, proto2])
    print(f"   Prototype scores: {[(id, f'{score:.3f}') for id, score in scores]}")
    
    # Create and sign package
    meta = MetaParams({"visual": 0.6}, 0.001, 0.95)
    core = InstinctCore([proto1], [], meta)
    overlay = InstinctOverlay([], [], "validator", "sig123")
    manifest = ValidationManifest(["test1"], {}, {}, ["v1"], now())
    
    package = InstinctPackage.create("test_pkg", "1.0.0", core, overlay, manifest)
    signed = sign_package(package, "private_key_123")
    valid = verify_signature(signed, "public_key_123")
    print(f"   Package signed: {signed.package_id}")
    print(f"   Signature valid: {valid}\n")
    
    # Test Phase 8 : Consolidation
    print("9. Phase 8 - Consolidation:")
    
    # Priority sampling
    test_episodes = [
        EpisodicRecord.create(f"Memory {i}", random(128), trust=0.7 + i*0.1)
        for i in range(5)
    ]
    sampled = priority_sample(test_episodes, k=3)
    print(f"   Priority sampled: {len(sampled)} from {len(test_episodes)} episodes")
    
    # Distillation
    concept_distilled = distill_to_semantic(test_episodes[:3])
    print(f"   Distilled concept: {concept_distilled.concept_id[:8]}...")
    print(f"   Doc count: {concept_distilled.doc_count}")
    
    # Clustering
    test_vecs = [random(64) for _ in range(10)]
    centroids = cluster_centroids(test_vecs, k=3, max_iter=5)
    print(f"   K-means centroids: {len(centroids)} clusters")
    
    # Forgetting policy
    old_memory = EpisodicRecord.create("Old memory", random(128), trust=0.5)
    should_forget = forgetting_policy(old_memory, age=86400*60, utility=0.2, threshold=0.1)
    print(f"   Should forget old memory: {should_forget}\n")
    
    # Test Phase 8.2 : Multimodal & Perception
    print("10. Phase 8.2 - Multimodal Embedding:")
    
    # Image embedding
    img = ImageTensor.create(224, 224, 3, fill_value=0.5)
    img_emb = embed_image(img, model="mobilenet")
    print(f"   Image embedded: {img.height}x{img.width} -> Vec(dim={img_emb.dim})")
    
    # Audio embedding
    audio = AudioSegment.create(duration_sec=1.0, sample_rate=16000)
    audio_emb = embed_audio(audio, model="wavenet")
    print(f"   Audio embedded: {audio.duration}s -> Vec(dim={audio_emb.dim})\n")
    
    print("11. Phase 8.2 - Temporal Alignment:")
    
    fusion = temporal_align({
        "image": img_emb,
        "audio": audio_emb
    }, window_ms=500)
    print(f"   Modalities aligned: {list(fusion.modalities.keys())}")
    print(f"   Alignment scores: {fusion.alignment_scores}\n")
    
    print("12. Phase 8.2 - Cross-Attention & Fusion:")
    
    # Cross-attention
    attended = cross_attention(img_emb, audio_emb, audio_emb, num_heads=4)
    print(f"   Cross-attention output: Vec(dim={attended.dim})")
    
    # Concat fusion
    fused_concat = fusion_concat([img_emb, audio_emb])
    print(f"   Concat fusion: {img_emb.dim} + {audio_emb.dim} = {fused_concat.dim}")
    
    # Weighted fusion
    weights_vec = Vec.from_list([0.6, 0.4])
    fused_weighted = fusion_weighted([img_emb, audio_emb], weights_vec)
    print(f"   Weighted fusion: Vec(dim={fused_weighted.dim}), weights=[0.6, 0.4]\n")
    
    print("13. Phase 8.2 - Vision & Audio Processing:")
    
    # Vision patches
    patches = vision_patch_extract(img, patch_size=16)
    print(f"   Vision patches: {len(patches)} patches of dim {patches[0].dim if patches else 0}")
    
    # Audio spectrogram
    spectrogram = audio_spectrogram(audio, n_fft=512, hop_length=256)
    print(f"   Spectrogram: {spectrogram.height}x{spectrogram.width} (freq x time)\n")
    
    # Test Phase 8.3 : Reasoner Hybride
    print("14. Phase 8.3 - Symbolic Matching:")
    
    # Créer quelques règles
    from normil_types import Rule
    rules = [
        Rule(id="high_confidence", condition="confidence > 0.8", action="accept", priority=100),
        Rule(id="low_confidence", condition="confidence < 0.3", action="reject", priority=50),
        Rule(id="medium_range", condition="confidence >= 0.3 and confidence <= 0.8", action="review", priority=75)
    ]
    
    # Test avec différents contextes
    context1 = {"confidence": 0.9, "source": "camera"}
    matched1 = symbolic_match(context1, rules)
    print(f"   Context confidence=0.9: {len(matched1)} rules matched")
    print(f"   Top match: {matched1[0].id if matched1 else 'None'}")
    
    context2 = {"confidence": 0.5}
    matched2 = symbolic_match(context2, rules)
    print(f"   Context confidence=0.5: {len(matched2)} rules matched")
    print(f"   Top match: {matched2[0].id if matched2 else 'None'}\n")
    
    print("15. Phase 8.3 - Neural Shortpass:")
    
    input_vec = random(512)
    context_vec = random(512)
    
    output_short, confidence = neural_shortpass(input_vec, model="tinynet", context_vec=context_vec)
    print(f"   Shortpass output: Vec(dim={output_short.dim})")
    print(f"   Confidence: {confidence:.3f}\n")
    
    print("16. Phase 8.3 - Neural Longpass:")
    
    # Créer quelques records pour retrieval
    retrieved_records = [
        EpisodicRecord.create(f"Memory {i}", random(512), trust=0.8)
        for i in range(3)
    ]
    
    output_long, trace = neural_longpass(input_vec, model="deepnet", retrieved=retrieved_records)
    print(f"   Longpass output: Vec(dim={output_long.dim})")
    print(f"   Trace layers: {trace['layers']}")
    print(f"   Retrieved: {trace['num_retrieved']} records")
    print(f"   Latency: {trace['latency_ms']}ms\n")
    
    print("17. Phase 8.3 - Meta-Controller:")
    
    test_vec = random(256)
    
    # Décision avec contrainte latence forte
    decision1 = meta_controller_decide(test_vec, cost_budget=0.5, latency_target_ms=30)
    print(f"   Latency target=30ms: {decision1}")
    
    # Décision avec budget élevé
    decision2 = meta_controller_decide(test_vec, cost_budget=0.9, latency_target_ms=300)
    print(f"   Latency target=300ms, budget=0.9: {decision2}")
    
    # Décision équilibrée
    decision3 = meta_controller_decide(test_vec, cost_budget=0.5, latency_target_ms=100)
    print(f"   Latency target=100ms, budget=0.5: {decision3}\n")
    
    # Test Phase 8.4 : Dev Tools
    print("18. Phase 8.4 - Introspection:")
    
    # Introspect différents types
    vec_info = introspect_type(test_vec)
    print(f"   Vec introspection: type={vec_info['type_name']}, dim={vec_info['fields']['dimension']}")
    print(f"   Vec metadata: norm={vec_info['metadata']['norm']:.3f}, mean={vec_info['metadata']['mean']:.3f}")
    
    record_info = introspect_type(retrieved_records[0])
    print(f"   EpisodicRecord: id={record_info['fields']['id'][:8]}..., trust={record_info['fields']['trust']}")
    
    concept_vec = random(128)
    concept = Concept.create(concept_vec, labels=["test"], trust=0.9)
    concept_info = introspect_type(concept)
    print(f"   Concept: id={concept_info['fields']['concept_id'][:8]}..., labels={concept_info['fields']['labels']}\n")
    
    print("19. Phase 8.4 - Trace Execution:")
    
    # Trace simple execution
    trace = trace_execution("norm(random(64))")
    print(f"   Traced: {trace['calls']} calls in {trace['execution_time_ms']:.3f}ms")
    print(f"   Success: {trace['success']}, result type: {trace['result_type']}")
    if trace['trace_log']:
        print(f"   First call: {trace['trace_log'][0]['function']}")
    
    # Trace plus complexe
    trace2 = trace_execution("dot(random(32), random(32))")
    print(f"   Complex trace: {trace2['calls']} calls in {trace2['execution_time_ms']:.3f}ms\n")
    
    print("20. Phase 8.4 - Signature & Discovery:")
    
    # Get signature
    sig = get_signature("dot")
    print(f"   Signature 'dot': {sig['doc']}")
    print(f"   Category: {sig['category']}")
    
    sig2 = get_signature("embed_image")
    print(f"   Signature 'embed_image': {sig2['doc']}")
    print(f"   Category: {sig2['category']}")
    
    # List primitives
    all_prims = list_primitives()
    print(f"   Total primitives: {len(all_prims)}")
    
    vec_prims = list_primitives("vector")
    print(f"   Vector primitives: {len(vec_prims)} - {vec_prims[:5]}")
    
    multimodal_prims = list_primitives("multimodal")
    print(f"   Multimodal primitives: {len(multimodal_prims)}")
    
    devtools_prims = list_primitives("devtools")
    print(f"   DevTools primitives: {len(devtools_prims)} - {devtools_prims}\n")
    
    print("21. Phase 8.4 - Visualisation:")
    
    # Viz vec space
    vecs = [random(32) for _ in range(20)]
    labels = [f"vec_{i}" for i in range(20)]
    viz_pca = viz_vec_space(vecs, labels, method="pca")
    print(f"   PCA visualization: {viz_pca['n_vectors']} vectors")
    print(f"   Explained variance: {viz_pca['explained_variance']:.3f}")
    print(f"   First coordinate: [{viz_pca['coordinates_2d'][0][0]:.3f}, {viz_pca['coordinates_2d'][0][1]:.3f}]")
    
    viz_tsne = viz_vec_space(vecs[:10], method="tsne")
    print(f"   t-SNE visualization: {viz_tsne['n_vectors']} vectors, {viz_tsne['iterations']} iterations")
    
    # Viz attention
    query_vec = random(64)
    key_vecs = [random(64) for _ in range(8)]
    viz_att = viz_attention(query_vec, key_vecs, num_heads=4)
    print(f"   Attention weights: max={viz_att['max_weight']:.3f} at index {viz_att['max_index']}")
    print(f"   Entropy: {viz_att['entropy']:.3f}")
    print(f"   Head contributions: {len(viz_att['head_contributions'])} heads")
    
    # Viz trace
    trace_str = viz_trace(trace['trace_log'])
    print(f"   Trace formatted: {len(trace_str)} chars")
    print("   Preview:")
    for line in trace_str.split('\n')[:3]:
        print(f"     {line}")
    
    print("\n==> All primitives tests passed (Phase 8.1 + 8.2 + 8.3 + 8.4)!")
