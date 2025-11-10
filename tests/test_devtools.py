"""
Tests pour les primitives DevTools (Phase 8.4).

Tests pour introspection, trace, signatures, et visualisation.

Auteur : Diego Morales Magri
"""

import pytest
import sys
from pathlib import Path

# Ajouter le répertoire parent au path
sys.path.insert(0, str(Path(__file__).parent.parent))

from runtime.normil_types import Vec, EpisodicRecord, Concept, Rule
from runtime.primitives import (
    introspect_type, trace_execution, get_signature, list_primitives,
    viz_vec_space, viz_attention, viz_trace,
    random, norm, dot, zeros
)


# ============================================
# Tests Introspection
# ============================================

class TestIntrospectType:
    """Tests pour introspect_type"""
    
    def test_introspect_vec(self):
        """Introspection d'un Vec"""
        v = Vec.from_list([1.0, 2.0, 3.0, 4.0])
        info = introspect_type(v)
        
        assert info["type_name"] == "Vec"
        assert info["fields"]["dimension"] == 4
        assert "norm" in info["metadata"]
        assert "mean" in info["metadata"]
        assert "__add__" in info["methods"]
    
    def test_introspect_vec_metadata(self):
        """Vérifier les métadonnées d'un Vec"""
        v = Vec.from_list([3.0, 4.0])
        info = introspect_type(v)
        
        assert abs(info["metadata"]["norm"] - 5.0) < 0.01
        assert abs(info["metadata"]["mean"] - 3.5) < 0.01
        assert info["metadata"]["min"] == 3.0
        assert info["metadata"]["max"] == 4.0
    
    def test_introspect_episodic_record(self):
        """Introspection d'un EpisodicRecord"""
        vec = random(128)
        record = EpisodicRecord.create("Test memory", vec, trust=0.85)
        info = introspect_type(record)
        
        assert info["type_name"] == "EpisodicRecord"
        assert "id" in info["fields"]
        assert info["fields"]["trust"] == 0.85
        assert info["fields"]["vec_dim"] == 128
        assert info["metadata"]["num_vecs"] == 1
    
    def test_introspect_concept(self):
        """Introspection d'un Concept"""
        centroid = random(64)
        concept = Concept.create(centroid, labels=["test", "concept"], trust=0.9)
        info = introspect_type(concept)
        
        assert info["type_name"] == "Concept"
        assert info["fields"]["centroid_dim"] == 64
        assert info["fields"]["labels"] == ["test", "concept"]
        assert info["fields"]["trust"] == 0.9
        assert info["metadata"]["num_labels"] == 2
    
    def test_introspect_rule(self):
        """Introspection d'une Rule"""
        rule = Rule(
            id="test_rule",
            condition="x > 0.5",
            action="accept",
            priority=100
        )
        info = introspect_type(rule)
        
        assert info["type_name"] == "Rule"
        assert info["fields"]["priority"] == 100
        assert info["metadata"]["has_condition"] == True
    
    def test_introspect_zero_vec(self):
        """Introspection d'un vecteur nul"""
        v = zeros(10)
        info = introspect_type(v)
        
        assert info["metadata"]["norm"] == 0.0
        assert info["metadata"]["mean"] == 0.0
        assert info["metadata"]["min"] == 0.0
        assert info["metadata"]["max"] == 0.0


# ============================================
# Tests Trace Execution
# ============================================

class TestTraceExecution:
    """Tests pour trace_execution"""
    
    def test_trace_simple_call(self):
        """Tracer un appel simple"""
        trace = trace_execution("norm(random(32))")
        
        assert trace["success"] == True
        assert trace["calls"] >= 2  # random + norm
        assert trace["execution_time_ms"] > 0
        assert "trace_log" in trace
    
    def test_trace_result_type(self):
        """Vérifier le type de résultat tracé"""
        trace = trace_execution("norm(random(10))")
        
        assert trace["result_type"] == "float"
        assert isinstance(trace["result"], float)
    
    def test_trace_multiple_calls(self):
        """Tracer plusieurs appels de primitives"""
        trace = trace_execution("dot(random(16), random(16))")
        
        assert trace["calls"] >= 3  # 2x random + dot
        assert len(trace["trace_log"]) >= 3
    
    def test_trace_with_context(self):
        """Tracer avec contexte fourni"""
        v = random(8)
        trace = trace_execution("norm(v)", context={"v": v})
        
        assert trace["success"] == True
        assert trace["calls"] >= 1
    
    def test_trace_error_handling(self):
        """Tracer une erreur"""
        trace = trace_execution("undefined_function()")
        
        assert trace["success"] == False
        assert "error" in trace
        assert trace["result"] is None
    
    def test_trace_timing(self):
        """Vérifier que le timing est enregistré"""
        trace = trace_execution("random(128)")
        
        assert trace["execution_time_ms"] > 0
        for entry in trace["trace_log"]:
            assert "time_ms" in entry
            assert entry["time_ms"] >= 0
    
    def test_trace_log_structure(self):
        """Vérifier la structure du trace log"""
        trace = trace_execution("norm(random(16))")
        
        for entry in trace["trace_log"]:
            assert "function" in entry
            assert "args_types" in entry
            assert "time_ms" in entry
            assert "result_type" in entry


# ============================================
# Tests Get Signature
# ============================================

class TestGetSignature:
    """Tests pour get_signature"""
    
    def test_signature_vector_primitive(self):
        """Signature d'une primitive vectorielle"""
        sig = get_signature("dot")
        
        assert sig["found"] == True
        assert sig["name"] == "dot"
        assert sig["category"] == "vector"
        assert len(sig["doc"]) > 0
    
    def test_signature_multimodal_primitive(self):
        """Signature d'une primitive multimodale"""
        sig = get_signature("embed_image")
        
        assert sig["found"] == True
        assert sig["category"] == "multimodal"
        assert "args" in sig
    
    def test_signature_reasoner_primitive(self):
        """Signature d'une primitive reasoner"""
        sig = get_signature("neural_shortpass")
        
        assert sig["found"] == True
        assert sig["category"] == "reasoner"
    
    def test_signature_devtools_primitive(self):
        """Signature d'une primitive devtools"""
        sig = get_signature("introspect_type")
        
        assert sig["found"] == True
        assert sig["category"] == "devtools"
    
    def test_signature_not_found(self):
        """Signature d'une primitive inexistante"""
        sig = get_signature("nonexistent_primitive")
        
        assert sig["found"] == False
        assert "error" in sig
    
    def test_signature_has_doc(self):
        """Vérifier que la signature contient la doc"""
        sig = get_signature("norm")
        
        assert "doc" in sig
        assert len(sig["doc"]) > 0
        assert "full_doc" in sig


# ============================================
# Tests List Primitives
# ============================================

class TestListPrimitives:
    """Tests pour list_primitives"""
    
    def test_list_all_primitives(self):
        """Lister toutes les primitives"""
        all_prims = list_primitives()
        
        assert isinstance(all_prims, list)
        assert len(all_prims) >= 90  # Au moins 90 primitives
        assert "dot" in all_prims
        assert "norm" in all_prims
    
    def test_list_vector_primitives(self):
        """Lister les primitives vectorielles"""
        vec_prims = list_primitives("vector")
        
        assert "dot" in vec_prims
        assert "norm" in vec_prims
        assert "zeros" in vec_prims
        assert "ones" in vec_prims
        assert "random" in vec_prims
    
    def test_list_multimodal_primitives(self):
        """Lister les primitives multimodales"""
        mm_prims = list_primitives("multimodal")
        
        assert "embed_image" in mm_prims
        assert "embed_audio" in mm_prims
        assert "fusion_concat" in mm_prims
    
    def test_list_reasoner_primitives(self):
        """Lister les primitives reasoner"""
        reasoner_prims = list_primitives("reasoner")
        
        assert "symbolic_match" in reasoner_prims
        assert "neural_shortpass" in reasoner_prims
        assert "neural_longpass" in reasoner_prims
        assert "meta_controller_decide" in reasoner_prims
    
    def test_list_devtools_primitives(self):
        """Lister les primitives devtools"""
        dev_prims = list_primitives("devtools")
        
        assert "introspect_type" in dev_prims
        assert "trace_execution" in dev_prims
        assert "get_signature" in dev_prims
        assert "list_primitives" in dev_prims
        assert "viz_vec_space" in dev_prims
        assert "viz_attention" in dev_prims
        assert "viz_trace" in dev_prims
    
    def test_list_sorted(self):
        """Vérifier que la liste est triée"""
        all_prims = list_primitives()
        
        assert all_prims == sorted(all_prims)


# ============================================
# Tests Viz Vec Space
# ============================================

class TestVizVecSpace:
    """Tests pour viz_vec_space"""
    
    def test_viz_pca_basic(self):
        """Visualisation PCA basique"""
        vecs = [random(64) for _ in range(20)]
        viz = viz_vec_space(vecs, method="pca")
        
        assert viz["method"] == "pca"
        assert viz["n_vectors"] == 20
        assert viz["dimension"] == 64
        assert len(viz["coordinates_2d"]) == 20
        assert "explained_variance" in viz
    
    def test_viz_pca_with_labels(self):
        """Visualisation PCA avec labels"""
        vecs = [random(32) for _ in range(10)]
        labels = [f"vec_{i}" for i in range(10)]
        viz = viz_vec_space(vecs, labels=labels, method="pca")
        
        assert viz["labels"] == labels
        assert len(viz["coordinates_2d"]) == len(labels)
    
    def test_viz_tsne_basic(self):
        """Visualisation t-SNE basique"""
        vecs = [random(32) for _ in range(15)]
        viz = viz_vec_space(vecs, method="tsne")
        
        assert viz["method"] == "tsne"
        assert viz["n_vectors"] == 15
        assert "iterations" in viz
        assert len(viz["coordinates_2d"]) == 15
    
    def test_viz_coordinates_structure(self):
        """Vérifier la structure des coordonnées"""
        import numpy as np
        
        vecs = [random(16) for _ in range(5)]
        viz = viz_vec_space(vecs, method="pca")
        
        for coord in viz["coordinates_2d"]:
            assert len(coord) == 2  # x, y
            # Accepter int, float, ou types numpy
            assert isinstance(coord[0], (int, float, np.number))
            assert isinstance(coord[1], (int, float, np.number))
    
    def test_viz_empty_vectors(self):
        """Visualisation avec liste vide"""
        viz = viz_vec_space([], method="pca")
        
        assert "error" in viz
        assert viz["coordinates_2d"] == []
    
    def test_viz_unknown_method(self):
        """Méthode de visualisation inconnue"""
        vecs = [random(16) for _ in range(5)]
        viz = viz_vec_space(vecs, method="unknown")
        
        assert "error" in viz


# ============================================
# Tests Viz Attention
# ============================================

class TestVizAttention:
    """Tests pour viz_attention"""
    
    def test_viz_attention_basic(self):
        """Visualisation attention basique"""
        query = random(64)
        keys = [random(64) for _ in range(10)]
        viz = viz_attention(query, keys)
        
        assert "attention_weights" in viz
        assert len(viz["attention_weights"]) == 10
        assert abs(sum(viz["attention_weights"]) - 1.0) < 0.01  # Somme = 1
    
    def test_viz_attention_output(self):
        """Vérifier l'output de l'attention"""
        query = random(32)
        keys = [random(32) for _ in range(5)]
        viz = viz_attention(query, keys)
        
        assert isinstance(viz["output"], Vec)
        assert len(viz["output"].data) == 32
    
    def test_viz_attention_multihead(self):
        """Attention multi-têtes"""
        query = random(64)
        keys = [random(64) for _ in range(8)]
        viz = viz_attention(query, keys, num_heads=4)
        
        assert viz["num_heads"] == 4
        assert len(viz["head_contributions"]) == 4
        
        for head in viz["head_contributions"]:
            assert "head" in head
            assert "weights" in head
            assert "entropy" in head
    
    def test_viz_attention_with_values(self):
        """Attention avec values séparés"""
        query = random(32)
        keys = [random(32) for _ in range(6)]
        values = [random(32) for _ in range(6)]
        viz = viz_attention(query, keys, values=values)
        
        assert len(viz["attention_weights"]) == 6
        assert isinstance(viz["output"], Vec)
    
    def test_viz_attention_entropy(self):
        """Vérifier l'entropie de l'attention"""
        query = random(32)
        keys = [random(32) for _ in range(8)]
        viz = viz_attention(query, keys)
        
        assert "entropy" in viz
        assert viz["entropy"] >= 0  # Entropie positive
    
    def test_viz_attention_max_weight(self):
        """Vérifier le poids maximal"""
        query = random(16)
        keys = [random(16) for _ in range(4)]
        viz = viz_attention(query, keys)
        
        assert "max_weight" in viz
        assert "max_index" in viz
        assert viz["max_weight"] == max(viz["attention_weights"])
        assert 0 <= viz["max_index"] < 4
    
    def test_viz_attention_empty_keys(self):
        """Attention avec keys vides"""
        query = random(32)
        viz = viz_attention(query, [])
        
        assert "error" in viz


# ============================================
# Tests Viz Trace
# ============================================

class TestVizTrace:
    """Tests pour viz_trace"""
    
    def test_viz_trace_basic(self):
        """Formatage basique d'un trace"""
        trace = trace_execution("norm(random(16))")
        formatted = viz_trace(trace["trace_log"])
        
        assert isinstance(formatted, str)
        assert len(formatted) > 0
        assert "Trace" in formatted
    
    def test_viz_trace_structure(self):
        """Vérifier la structure du trace formaté"""
        trace = trace_execution("dot(random(8), random(8))")
        formatted = viz_trace(trace["trace_log"])
        
        lines = formatted.split('\n')
        assert len(lines) >= 3  # Header + au moins 2 calls
        assert "calls" in lines[0]
        assert "total" in lines[0]
    
    def test_viz_trace_empty(self):
        """Formatage d'un trace vide"""
        formatted = viz_trace([])
        
        assert formatted == "(empty trace)"
    
    def test_viz_trace_timing_info(self):
        """Vérifier les infos de timing"""
        trace = trace_execution("random(32)")
        formatted = viz_trace(trace["trace_log"])
        
        assert "ms" in formatted
        assert "%" in formatted


# ============================================
# Tests d'Intégration DevTools
# ============================================

class TestDevToolsIntegration:
    """Tests d'intégration des outils de développement"""
    
    def test_introspect_then_trace(self):
        """Introspection suivie de trace"""
        v = random(32)
        info = introspect_type(v)
        
        # Utiliser l'info pour tracer
        assert info["fields"]["dimension"] == 32
        
        trace = trace_execution("norm(v)", context={"v": v})
        assert trace["success"] == True
    
    def test_discover_and_inspect(self):
        """Découvrir puis inspecter des primitives"""
        # Découvrir
        dev_prims = list_primitives("devtools")
        assert len(dev_prims) >= 7
        
        # Inspecter chacune
        for prim_name in dev_prims[:3]:
            sig = get_signature(prim_name)
            assert sig["found"] == True
            assert sig["category"] == "devtools"
    
    def test_trace_and_visualize(self):
        """Tracer puis visualiser"""
        trace = trace_execution("norm(random(16))")
        
        # Visualiser le trace
        formatted = viz_trace(trace["trace_log"])
        assert len(formatted) > 0
        
        # Vérifier cohérence
        assert trace["calls"] == len(trace["trace_log"])
    
    def test_full_debugging_workflow(self):
        """Workflow complet de debugging"""
        # 1. Créer des vecteurs
        vecs = [random(32) for _ in range(10)]
        
        # 2. Introspection
        info = introspect_type(vecs[0])
        assert info["fields"]["dimension"] == 32
        
        # 3. Visualisation
        viz = viz_vec_space(vecs, method="pca")
        assert viz["n_vectors"] == 10
        
        # 4. Trace d'une opération
        trace = trace_execution("dot(vecs[0], vecs[1])", context={"vecs": vecs})
        assert trace["success"] == True
        
        # 5. Formatage du trace
        formatted = viz_trace(trace["trace_log"])
        assert "dot" in formatted
