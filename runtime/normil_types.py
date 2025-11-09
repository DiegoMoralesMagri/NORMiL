"""
NORMiL Runtime Types
====================

Implémentation des types natifs NORMiL en Python.

Types principaux:
- Vec: Vecteur avec dimension et quantisation
- EpisodicRecord: Enregistrement de mémoire épisodique
- Concept: Concept de mémoire sémantique
- WorkingMemoryEntry: Entrée de mémoire de travail
- ProtoInstinct: Prototype d'instinct
- Label, Provenance, Rule, Policy
"""

import numpy as np
import uuid as uuid_lib
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
from datetime import datetime


# ============================================
# Types Vectoriels
# ============================================

@dataclass
class SparseVec:
    """
    Vecteur creux (sparse) optimisé pour stockage.
    
    Stocke seulement les indices non-nuls et leurs valeurs.
    Efficace quand la plupart des valeurs sont à zéro.
    
    Attributs:
        indices: Indices des valeurs non-nulles
        values: Valeurs correspondantes
        dim: Dimension totale du vecteur
    """
    indices: List[int]
    values: List[float]
    dim: int
    
    def __post_init__(self):
        """Valide la cohérence"""
        if len(self.indices) != len(self.values):
            raise ValueError(f"SparseVec: indices and values must have same length")
        if self.indices and max(self.indices) >= self.dim:
            raise ValueError(f"SparseVec: index {max(self.indices)} out of bounds (dim={self.dim})")
    
    def __repr__(self) -> str:
        nnz = len(self.indices)
        sparsity = (1 - nnz/self.dim) * 100 if self.dim > 0 else 0
        return f"SparseVec(dim={self.dim}, nnz={nnz}, sparsity={sparsity:.1f}%)"
    
    def __len__(self) -> int:
        return self.dim
    
    def to_dense(self) -> 'Vec':
        """Convertit en vecteur dense"""
        data = np.zeros(self.dim, dtype=np.float16)
        for idx, val in zip(self.indices, self.values):
            data[idx] = val
        return Vec(data, self.dim)
    
    @classmethod
    def from_dense(cls, vec: 'Vec', threshold: float = 1e-6) -> 'SparseVec':
        """Crée un vecteur creux depuis un vecteur dense"""
        indices = []
        values = []
        for i, v in enumerate(vec.data):
            if abs(v) > threshold:
                indices.append(i)
                values.append(float(v))
        return cls(indices, values, vec.dim)
    
    @classmethod
    def from_lists(cls, indices: List[int], values: List[float], dim: int) -> 'SparseVec':
        """Crée un vecteur creux depuis des listes"""
        return cls(indices, values, dim)


@dataclass
class Vec:
    """
    Vecteur NORMiL avec dimension et quantisation.
    
    Attributs:
        data: Tableau numpy (float16 ou quantisé)
        dim: Dimension du vecteur
        quantization: Bits de quantisation (None pour float16)
    """
    data: np.ndarray
    dim: int
    quantization: Optional[int] = None
    
    def __post_init__(self):
        """Valide la dimension"""
        if len(self.data) != self.dim:
            raise ValueError(f"Vec dimension mismatch: expected {self.dim}, got {len(self.data)}")
        
        # Convertir en float16 si pas de quantisation
        if self.quantization is None and self.data.dtype != np.float16:
            self.data = self.data.astype(np.float16)
    
    def __repr__(self) -> str:
        q_str = f", q={self.quantization}" if self.quantization else ""
        return f"Vec(dim={self.dim}{q_str}, data={self.data[:3]}...)"
    
    def __len__(self) -> int:
        return self.dim
    
    def __getitem__(self, index):
        return self.data[index]
    
    def __setitem__(self, index, value):
        self.data[index] = value
    
    def __add__(self, other):
        """Addition de vecteurs"""
        if isinstance(other, Vec):
            if self.dim != other.dim:
                raise ValueError(f"Vec dimension mismatch: {self.dim} vs {other.dim}")
            result_data = self.data + other.data
            return Vec(result_data, self.dim, self.quantization)
        else:
            raise TypeError(f"Cannot add Vec and {type(other)}")
    
    def __sub__(self, other):
        """Soustraction de vecteurs"""
        if isinstance(other, Vec):
            if self.dim != other.dim:
                raise ValueError(f"Vec dimension mismatch: {self.dim} vs {other.dim}")
            result_data = self.data - other.data
            return Vec(result_data, self.dim, self.quantization)
        else:
            raise TypeError(f"Cannot subtract {type(other)} from Vec")
    
    def __mul__(self, scalar):
        """Multiplication par un scalaire"""
        if isinstance(scalar, (int, float, np.number)):
            result_data = self.data * scalar
            return Vec(result_data, self.dim, self.quantization)
        else:
            raise TypeError(f"Cannot multiply Vec by {type(scalar)}")
    
    def __rmul__(self, scalar):
        """Multiplication par un scalaire (inverse)"""
        return self.__mul__(scalar)
    
    def to_list(self) -> List[float]:
        """Convertit en liste Python"""
        return self.data.tolist()
    
    @classmethod
    def zeros(cls, dim: int, quantization: Optional[int] = None) -> 'Vec':
        """Crée un vecteur de zéros"""
        data = np.zeros(dim, dtype=np.float16)
        return cls(data, dim, quantization)
    
    @classmethod
    def ones(cls, dim: int, quantization: Optional[int] = None) -> 'Vec':
        """Crée un vecteur de uns"""
        data = np.ones(dim, dtype=np.float16)
        return cls(data, dim, quantization)
    
    @classmethod
    def random(cls, dim: int, mean: float = 0.0, std: float = 1.0, 
               quantization: Optional[int] = None) -> 'Vec':
        """Crée un vecteur aléatoire (distribution normale)"""
        data = np.random.normal(mean, std, dim).astype(np.float16)
        return cls(data, dim, quantization)
    
    @classmethod
    def from_list(cls, values: List[float], quantization: Optional[int] = None) -> 'Vec':
        """Crée un vecteur depuis une liste"""
        data = np.array(values, dtype=np.float16)
        return cls(data, len(values), quantization)


# ============================================
# Types Mémoire
# ============================================

@dataclass
class Label:
    """Label avec score de confiance"""
    label: str
    score: float
    
    def __repr__(self) -> str:
        return f"Label({self.label}, {self.score:.3f})"


@dataclass
class Provenance:
    """Information de provenance (traçabilité)"""
    device_id: str
    signature: str
    timestamp: float
    
    def __repr__(self) -> str:
        return f"Provenance({self.device_id}, {self.signature[:8]}...)"
    
    @classmethod
    def create(cls, device_id: str = "default") -> 'Provenance':
        """Crée une provenance avec timestamp actuel"""
        import hashlib
        ts = datetime.now().timestamp()
        data = f"{device_id}{ts}".encode()
        sig = hashlib.sha256(data).hexdigest()
        return cls(device_id, sig, ts)


@dataclass
class EpisodicRecord:
    """
    Enregistrement de mémoire épisodique.
    
    Stocke un événement brut horodaté avec vecteurs multimodaux.
    """
    id: str
    timestamp: float
    sources: List[str]
    vecs: Dict[str, Vec]
    summary: str
    labels: List[Label]
    trust: float
    provenance: Provenance
    outcome: Optional[str] = None
    
    def __repr__(self) -> str:
        return (f"EpisodicRecord(id={self.id[:8]}..., "
                f"timestamp={self.timestamp:.2f}, "
                f"sources={self.sources}, "
                f"trust={self.trust:.2f})")
    
    @classmethod
    def create(cls, summary: str, vec: Vec, trust: float = 0.9, 
               source: str = "default") -> 'EpisodicRecord':
        """Crée un enregistrement épisodique simple"""
        return cls(
            id=str(uuid_lib.uuid4()),
            timestamp=datetime.now().timestamp(),
            sources=[source],
            vecs={"default": vec},
            summary=summary,
            labels=[],
            trust=trust,
            provenance=Provenance.create(),
            outcome=None
        )


@dataclass
class WorkingMemoryEntry:
    """Entrée dans la mémoire de travail (cache volatile)"""
    id: str
    vec_combined: Vec
    last_access_ms: float
    relevance_score: float
    expire_ttl: int  # millisecondes
    refs_to_episodic_ids: List[str]
    
    def __repr__(self) -> str:
        return (f"WorkingMemoryEntry(id={self.id[:8]}..., "
                f"relevance={self.relevance_score:.2f}, "
                f"ttl={self.expire_ttl}ms)")
    
    @classmethod
    def create(cls, vec: Vec, relevance: float = 1.0, 
               ttl_ms: int = 60000) -> 'WorkingMemoryEntry':
        """Crée une entrée de working memory"""
        return cls(
            id=str(uuid_lib.uuid4()),
            vec_combined=vec,
            last_access_ms=datetime.now().timestamp() * 1000,
            relevance_score=relevance,
            expire_ttl=ttl_ms,
            refs_to_episodic_ids=[]
        )


@dataclass
class Concept:
    """Concept dans la mémoire sémantique (knowledge compressé)"""
    concept_id: str
    centroid_vec: Vec
    doc_count: int
    provenance_versions: List[str]
    trust_score: float
    labels: List[str]
    
    def __repr__(self) -> str:
        return (f"Concept(id={self.concept_id[:8]}..., "
                f"doc_count={self.doc_count}, "
                f"trust={self.trust_score:.2f}, "
                f"labels={self.labels})")
    
    @classmethod
    def create(cls, centroid: Vec, labels: List[str] = None, 
               trust: float = 0.8) -> 'Concept':
        """Crée un concept"""
        return cls(
            concept_id=str(uuid_lib.uuid4()),
            centroid_vec=centroid,
            doc_count=1,
            provenance_versions=["v1"],
            trust_score=trust,
            labels=labels or []
        )


# ============================================
# Types Instinct
# ============================================

@dataclass
class Rule:
    """Règle symbolique pour instinct"""
    id: str
    condition: str  # Expression booléenne
    action: str
    priority: int
    
    def __repr__(self) -> str:
        return f"Rule({self.id}, priority={self.priority})"


@dataclass
class ProtoInstinct:
    """Prototype d'instinct (vecteur de référence + règle)"""
    id: str
    vec_ref: Vec
    rule: Optional[Rule]
    weight: float
    
    def __repr__(self) -> str:
        return f"ProtoInstinct({self.id}, weight={self.weight:.2f})"
    
    @classmethod
    def create(cls, id: str, vec: Vec, weight: float = 1.0, 
               rule: Optional[Rule] = None) -> 'ProtoInstinct':
        """Crée un proto-instinct"""
        return cls(id, vec, rule, weight)


@dataclass
class Policy:
    """Politique (meta-règle)"""
    name: str
    rules: List[Rule]
    activation_threshold: float
    
    def __repr__(self) -> str:
        return f"Policy({self.name}, {len(self.rules)} rules)"


# ============================================
# Types O-RedMind Phase 8
# ============================================

@dataclass
class MetaParams:
    """Paramètres méta pour instinct core"""
    attention_weights: Dict[str, float]
    base_plastic_rate: float
    safety_threshold: float
    
    def __repr__(self) -> str:
        return f"MetaParams(base_rate={self.base_plastic_rate:.4f}, safety={self.safety_threshold:.2f})"


@dataclass
class ValidationManifest:
    """Manifest de validation pour overlay d'instinct"""
    tests_passed: List[str]
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    validators: List[str]
    timestamp: float
    
    def __repr__(self) -> str:
        return f"ValidationManifest({len(self.tests_passed)} tests, {len(self.validators)} validators)"


@dataclass
class InstinctCore:
    """Core immuable de l'instinct"""
    prototypes: List[ProtoInstinct]
    rules: List[Rule]
    meta_params: MetaParams
    
    def __repr__(self) -> str:
        return f"InstinctCore({len(self.prototypes)} protos, {len(self.rules)} rules)"


@dataclass
class InstinctOverlay:
    """Overlay modifiable validé pour l'instinct"""
    prototypes: List[ProtoInstinct]
    rules: List[Rule]
    provenance: str
    validation_signature: str
    
    def __repr__(self) -> str:
        return f"InstinctOverlay({len(self.prototypes)} protos, validated)"


@dataclass
class InstinctPackage:
    """
    Package complet d'instinct avec core + overlay.
    
    Utilisé pour gouvernance et versioning des instincts O-RedMind.
    Le core est signé et immuable, l'overlay est validé et opt-in.
    """
    package_id: str
    version: str
    signature: str
    timestamp: float
    core: InstinctCore
    overlay: InstinctOverlay
    validation_manifest: ValidationManifest
    
    def __repr__(self) -> str:
        return f"InstinctPackage({self.package_id}, v{self.version})"
    
    @classmethod
    def create(cls, package_id: str, version: str, 
               core: InstinctCore, overlay: InstinctOverlay,
               manifest: ValidationManifest) -> 'InstinctPackage':
        """Crée un package d'instinct"""
        import hashlib
        
        # Signature du package
        data = f"{package_id}{version}{core}{overlay}".encode()
        signature = hashlib.sha256(data).hexdigest()
        
        return cls(
            package_id=package_id,
            version=version,
            signature=signature,
            timestamp=datetime.now().timestamp(),
            core=core,
            overlay=overlay,
            validation_manifest=manifest
        )


@dataclass
class SafetyGuardrail:
    """
    Guardrail de sécurité pour O-RedMind.
    
    Définit une règle de sécurité qui bloque certaines actions
    et peut nécessiter un consentement utilisateur explicite.
    """
    id: str
    condition: str              # Expression booléenne
    action_blocked: str         # Action à bloquer
    require_consent: bool       # Nécessite consentement humain
    override_level: int         # Niveau privilège requis (0-10)
    description: str = ""
    
    def __repr__(self) -> str:
        consent = "🔒" if self.require_consent else ""
        return f"SafetyGuardrail({self.id}, level={self.override_level} {consent})"
    
    @classmethod
    def create(cls, id: str, condition: str, action_blocked: str,
               require_consent: bool = True, override_level: int = 10,
               description: str = "") -> 'SafetyGuardrail':
        """Crée un guardrail de sécurité"""
        return cls(
            id=id,
            condition=condition,
            action_blocked=action_blocked,
            require_consent=require_consent,
            override_level=override_level,
            description=description
        )


@dataclass
class ConsentRequest:
    """Requête de consentement utilisateur"""
    action: str
    reason: str
    data_accessed: List[str]
    expiry_ttl: int  # millisecondes
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    
    def __repr__(self) -> str:
        return f"ConsentRequest({self.action}, {len(self.data_accessed)} data points)"


@dataclass
class AuditLogEntry:
    """
    Entrée dans le journal d'audit append-only.
    
    Utilise hash chaining pour garantir l'intégrité :
    chaque entrée contient le hash de l'entrée précédente.
    """
    id: str
    timestamp: float
    event_type: str
    actor: str
    action: str
    data_hash: str
    prev_hash: str              # Hash chaining
    signature: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"AuditLogEntry({self.event_type}, actor={self.actor}, ts={self.timestamp:.2f})"
    
    @classmethod
    def create(cls, event_type: str, actor: str, action: str,
               data: Any, prev_hash: str = "0" * 64) -> 'AuditLogEntry':
        """Crée une entrée d'audit avec hash chaining"""
        import hashlib
        
        entry_id = str(uuid_lib.uuid4())
        ts = datetime.now().timestamp()
        
        # Hash des données
        data_str = str(data).encode()
        data_hash = hashlib.sha256(data_str).hexdigest()
        
        # Signature de l'entrée
        entry_data = f"{entry_id}{ts}{event_type}{actor}{action}{data_hash}{prev_hash}".encode()
        signature = hashlib.sha256(entry_data).hexdigest()
        
        return cls(
            id=entry_id,
            timestamp=ts,
            event_type=event_type,
            actor=actor,
            action=action,
            data_hash=data_hash,
            prev_hash=prev_hash,
            signature=signature,
            metadata={}
        )
    
    def compute_hash(self) -> str:
        """Calcule le hash de cette entrée (pour chain)"""
        import hashlib
        data = f"{self.id}{self.timestamp}{self.event_type}{self.signature}".encode()
        return hashlib.sha256(data).hexdigest()


@dataclass
class IndexEntry:
    """
    Entrée dans l'index HNSW-like pour fast retrieval.
    
    Stocke un vecteur avec métadonnées et voisins pour recherche rapide.
    """
    id: str
    vec: Vec
    metadata: Dict[str, str]
    neighbors: List[str]        # IDs des voisins HNSW
    layer: int                  # Couche HNSW (0 = base)
    timestamp: float
    distance_cache: Dict[str, float] = field(default_factory=dict)
    
    def __repr__(self) -> str:
        return f"IndexEntry({self.id[:8]}..., layer={self.layer}, {len(self.neighbors)} neighbors)"
    
    @classmethod
    def create(cls, vec: Vec, metadata: Dict[str, str] = None,
               layer: int = 0) -> 'IndexEntry':
        """Crée une entrée d'index"""
        return cls(
            id=str(uuid_lib.uuid4()),
            vec=vec,
            metadata=metadata or {},
            neighbors=[],
            layer=layer,
            timestamp=datetime.now().timestamp(),
            distance_cache={}
        )
    
    def add_neighbor(self, neighbor_id: str, distance: float):
        """Ajoute un voisin avec sa distance"""
        if neighbor_id not in self.neighbors:
            self.neighbors.append(neighbor_id)
            self.distance_cache[neighbor_id] = distance
    
    def get_distance(self, neighbor_id: str) -> Optional[float]:
        """Récupère la distance cachée vers un voisin"""
        return self.distance_cache.get(neighbor_id)


# ============================================
# Helpers
# ============================================

def generate_uuid() -> str:
    """Génère un UUID v4"""
    return str(uuid_lib.uuid4())


def now() -> float:
    """Retourne le timestamp actuel"""
    return datetime.now().timestamp()


# ============================================
# Phase 8.2 - Multimodal & Perception Types
# ============================================

@dataclass
class ImageTensor:
    """Représentation d'une image pour NORMiL
    
    Utilisé pour la perception visuelle dans O-RedMind.
    Compatible avec format HWC (Height × Width × Channels).
    """
    height: int
    width: int
    channels: int  # 1 (grayscale), 3 (RGB), 4 (RGBA)
    data: np.ndarray  # shape: (height, width, channels)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def create(height: int, width: int, channels: int = 3, 
               fill_value: float = 0.0, metadata: Optional[Dict] = None) -> 'ImageTensor':
        """Crée une ImageTensor initialisée"""
        data = np.full((height, width, channels), fill_value, dtype=np.float32)
        return ImageTensor(
            height=height,
            width=width,
            channels=channels,
            data=data,
            metadata=metadata or {}
        )
    
    @staticmethod
    def from_array(data: np.ndarray, metadata: Optional[Dict] = None) -> 'ImageTensor':
        """Crée une ImageTensor depuis un numpy array HWC"""
        if len(data.shape) == 2:
            # Grayscale: ajoute dimension channel
            data = data[:, :, np.newaxis]
        
        if len(data.shape) != 3:
            raise ValueError(f"Expected 2D or 3D array, got shape {data.shape}")
        
        height, width, channels = data.shape
        return ImageTensor(
            height=height,
            width=width,
            channels=channels,
            data=data.astype(np.float32),
            metadata=metadata or {}
        )
    
    def to_grayscale(self) -> 'ImageTensor':
        """Convertit en grayscale (moyenne des canaux)"""
        if self.channels == 1:
            return self
        
        gray_data = np.mean(self.data, axis=2, keepdims=True)
        return ImageTensor(
            height=self.height,
            width=self.width,
            channels=1,
            data=gray_data,
            metadata=self.metadata.copy()
        )
    
    def resize(self, new_height: int, new_width: int) -> 'ImageTensor':
        """Redimensionne l'image (bilinear interpolation simulée)"""
        # Interpolation simple: repeat ou subsample
        h_ratio = self.height / new_height
        w_ratio = self.width / new_width
        
        new_data = np.zeros((new_height, new_width, self.channels), dtype=np.float32)
        
        for i in range(new_height):
            for j in range(new_width):
                src_i = min(int(i * h_ratio), self.height - 1)
                src_j = min(int(j * w_ratio), self.width - 1)
                new_data[i, j] = self.data[src_i, src_j]
        
        return ImageTensor(
            height=new_height,
            width=new_width,
            channels=self.channels,
            data=new_data,
            metadata=self.metadata.copy()
        )


@dataclass
class AudioSegment:
    """Représentation d'un segment audio pour NORMiL
    
    Utilisé pour la perception auditive dans O-RedMind.
    """
    samples: np.ndarray  # 1D array de samples audio
    sample_rate: int     # Hz (ex: 16000, 44100)
    duration: float      # secondes
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def create(duration_sec: float, sample_rate: int = 16000, 
               fill_value: float = 0.0, metadata: Optional[Dict] = None) -> 'AudioSegment':
        """Crée un AudioSegment initialisé"""
        num_samples = int(duration_sec * sample_rate)
        samples = np.full(num_samples, fill_value, dtype=np.float32)
        return AudioSegment(
            samples=samples,
            sample_rate=sample_rate,
            duration=duration_sec,
            metadata=metadata or {}
        )
    
    @staticmethod
    def from_array(samples: np.ndarray, sample_rate: int = 16000, 
                   metadata: Optional[Dict] = None) -> 'AudioSegment':
        """Crée un AudioSegment depuis un numpy array"""
        if len(samples.shape) != 1:
            raise ValueError(f"Expected 1D array, got shape {samples.shape}")
        
        duration = len(samples) / sample_rate
        return AudioSegment(
            samples=samples.astype(np.float32),
            sample_rate=sample_rate,
            duration=duration,
            metadata=metadata or {}
        )
    
    def resample(self, new_sample_rate: int) -> 'AudioSegment':
        """Ré-échantillonne à un nouveau sample rate (interpolation simple)"""
        if new_sample_rate == self.sample_rate:
            return self
        
        ratio = new_sample_rate / self.sample_rate
        new_length = int(len(self.samples) * ratio)
        
        # Interpolation linéaire simple
        indices = np.linspace(0, len(self.samples) - 1, new_length)
        new_samples = np.interp(indices, np.arange(len(self.samples)), self.samples)
        
        return AudioSegment(
            samples=new_samples.astype(np.float32),
            sample_rate=new_sample_rate,
            duration=self.duration,
            metadata=self.metadata.copy()
        )
    
    def trim(self, start_sec: float, end_sec: float) -> 'AudioSegment':
        """Extrait un segment temporel"""
        start_idx = int(start_sec * self.sample_rate)
        end_idx = int(end_sec * self.sample_rate)
        
        start_idx = max(0, min(start_idx, len(self.samples)))
        end_idx = max(start_idx, min(end_idx, len(self.samples)))
        
        new_samples = self.samples[start_idx:end_idx]
        new_duration = (end_idx - start_idx) / self.sample_rate
        
        return AudioSegment(
            samples=new_samples,
            sample_rate=self.sample_rate,
            duration=new_duration,
            metadata=self.metadata.copy()
        )


@dataclass
class ModalityFusion:
    """Résultat de fusion multimodale
    
    Contient les embeddings de différentes modalités alignées temporellement.
    """
    modalities: Dict[str, Vec]  # {"image": Vec, "audio": Vec, ...}
    alignment_scores: Dict[str, float]  # Scores de confiance par modalité
    timestamp: float
    fused_vec: Optional[Vec] = None  # Vecteur fusionné (optionnel)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @staticmethod
    def create(modalities: Dict[str, Vec], 
               alignment_scores: Optional[Dict[str, float]] = None,
               fused_vec: Optional[Vec] = None,
               metadata: Optional[Dict] = None) -> 'ModalityFusion':
        """Crée une ModalityFusion"""
        if alignment_scores is None:
            alignment_scores = {k: 1.0 for k in modalities.keys()}
        
        return ModalityFusion(
            modalities=modalities,
            alignment_scores=alignment_scores,
            timestamp=datetime.now().timestamp(),
            fused_vec=fused_vec,
            metadata=metadata or {}
        )
    
    def get_modality(self, name: str) -> Optional[Vec]:
        """Récupère un vecteur de modalité spécifique"""
        return self.modalities.get(name)
    
    def add_modality(self, name: str, vec: Vec, alignment_score: float = 1.0):
        """Ajoute une modalité"""
        self.modalities[name] = vec
        self.alignment_scores[name] = alignment_score


# ============================================
# Tests
# ============================================

if __name__ == '__main__':
    print("=== Test NORMiL Types ===\n")
    
    # Test Vec
    print("1. Vec:")
    v1 = Vec.zeros(256)
    print(f"   Zeros: {v1}")
    
    v2 = Vec.random(256, mean=0.0, std=1.0)
    print(f"   Random: {v2}")
    
    v3 = Vec.from_list([1.0, 2.0, 3.0])
    print(f"   From list: {v3}\n")
    
    # Test EpisodicRecord
    print("2. EpisodicRecord:")
    vec = Vec.random(128)
    record = EpisodicRecord.create(
        summary="Test memory",
        vec=vec,
        trust=0.95
    )
    print(f"   {record}\n")
    
    # Test Concept
    print("3. Concept:")
    centroid = Vec.random(128)
    concept = Concept.create(
        centroid=centroid,
        labels=["test", "memory"],
        trust=0.85
    )
    print(f"   {concept}\n")
    
    # Test ProtoInstinct
    print("4. ProtoInstinct:")
    instinct_vec = Vec.random(128)
    rule = Rule(
        id="test_rule",
        condition="similarity > 0.8",
        action="boost_attention",
        priority=100
    )
    instinct = ProtoInstinct.create(
        id="privacy_guard",
        vec=instinct_vec,
        weight=1.5,
        rule=rule
    )
    print(f"   {instinct}")
    print(f"   Rule: {rule}\n")
    
    # Test InstinctPackage (Phase 8)
    print("5. InstinctPackage:")
    meta_params = MetaParams(
        attention_weights={"visual": 0.6, "audio": 0.4},
        base_plastic_rate=0.001,
        safety_threshold=0.95
    )
    core = InstinctCore(
        prototypes=[instinct],
        rules=[rule],
        meta_params=meta_params
    )
    overlay = InstinctOverlay(
        prototypes=[],
        rules=[],
        provenance="test_validator",
        validation_signature="abc123"
    )
    manifest = ValidationManifest(
        tests_passed=["test_1", "test_2"],
        metrics_before={"accuracy": 0.85},
        metrics_after={"accuracy": 0.90},
        validators=["validator_1"],
        timestamp=now()
    )
    package = InstinctPackage.create(
        package_id="instinct_v1",
        version="1.0.0",
        core=core,
        overlay=overlay,
        manifest=manifest
    )
    print(f"   {package}\n")
    
    # Test SafetyGuardrail (Phase 8)
    print("6. SafetyGuardrail:")
    guardrail = SafetyGuardrail.create(
        id="no_io_without_consent",
        condition="action.type == 'file_write'",
        action_blocked="write",
        require_consent=True,
        override_level=10,
        description="Empêche les écritures fichiers sans consentement"
    )
    print(f"   {guardrail}\n")
    
    # Test AuditLogEntry (Phase 8)
    print("7. AuditLogEntry:")
    entry1 = AuditLogEntry.create(
        event_type="memory_append",
        actor="system",
        action="append_episodic",
        data={"record_id": "12345"}
    )
    print(f"   Entry 1: {entry1}")
    
    # Hash chaining
    entry2 = AuditLogEntry.create(
        event_type="memory_retrieve",
        actor="user",
        action="query_index",
        data={"query": "test"},
        prev_hash=entry1.compute_hash()
    )
    print(f"   Entry 2: {entry2}")
    print(f"   Chain valid: {entry2.prev_hash == entry1.compute_hash()}\n")
    
    # Test IndexEntry (Phase 8)
    print("8. IndexEntry:")
    index_vec = Vec.random(128)
    entry = IndexEntry.create(
        vec=index_vec,
        metadata={"type": "image", "timestamp": "2025-11-01"},
        layer=0
    )
    entry.add_neighbor("neighbor_1", distance=0.15)
    entry.add_neighbor("neighbor_2", distance=0.23)
    print(f"   {entry}")
    print(f"   Distance to neighbor_1: {entry.get_distance('neighbor_1')}\n")
    
    # Test ImageTensor (Phase 8.2)
    print("9. ImageTensor:")
    img = ImageTensor.create(height=224, width=224, channels=3, fill_value=0.5)
    print(f"   Created: {img.height}x{img.width}x{img.channels}, mean={img.data.mean():.2f}")
    
    img_resized = img.resize(112, 112)
    print(f"   Resized: {img_resized.height}x{img_resized.width}")
    
    img_gray = img.to_grayscale()
    print(f"   Grayscale: {img_gray.channels} channel(s)\n")
    
    # Test AudioSegment (Phase 8.2)
    print("10. AudioSegment:")
    audio = AudioSegment.create(duration_sec=1.0, sample_rate=16000, fill_value=0.0)
    print(f"   Created: {len(audio.samples)} samples @ {audio.sample_rate}Hz, duration={audio.duration:.2f}s")
    
    audio_trimmed = audio.trim(0.2, 0.5)
    print(f"   Trimmed: {audio_trimmed.duration:.2f}s ({len(audio_trimmed.samples)} samples)")
    
    audio_resampled = audio.resample(8000)
    print(f"   Resampled: {audio_resampled.sample_rate}Hz ({len(audio_resampled.samples)} samples)\n")
    
    # Test ModalityFusion (Phase 8.2)
    print("11. ModalityFusion:")
    fusion = ModalityFusion.create(
        modalities={
            "image": Vec.random(512),
            "audio": Vec.random(512)
        },
        alignment_scores={"image": 0.95, "audio": 0.88}
    )
    print(f"   Modalities: {list(fusion.modalities.keys())}")
    print(f"   Alignment scores: {fusion.alignment_scores}")
    img_vec = fusion.get_modality('image')
    print(f"   Image vec dim: {img_vec.dim if img_vec else 'N/A'}\n")
    
    print("==> All types tests passed (including Phase 8 + Phase 8.2)!")
