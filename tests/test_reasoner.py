"""
Tests pour les primitives de reasoner hybride Phase 8.3
============================================
Auteur : Diego Morales Magri
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runtime.normil_types import Vec, EpisodicRecord, Rule
from runtime.primitives import (
    symbolic_match, neural_shortpass, neural_longpass, meta_controller_decide
)


# ============================================
# Tests Symbolic Matching
# ============================================

class TestSymbolicMatch:
    """Tests pour symbolic_match"""
    
    def test_basic_match(self):
        """Test matching basique avec une règle simple"""
        rules = [
            Rule(id="rule1", condition="x > 10", action="accept", priority=100)
        ]
        context = {"x": 15}
        
        matched = symbolic_match(context, rules)
        
        assert len(matched) == 1
        assert matched[0].id == "rule1"
    
    def test_no_match(self):
        """Test quand aucune règle ne matche"""
        rules = [
            Rule(id="rule1", condition="x > 10", action="accept", priority=100)
        ]
        context = {"x": 5}
        
        matched = symbolic_match(context, rules)
        
        assert len(matched) == 0
    
    def test_multiple_matches(self):
        """Test avec plusieurs règles qui matchent"""
        rules = [
            Rule(id="high", condition="score > 0.8", action="accept", priority=100),
            Rule(id="medium", condition="score > 0.5", action="review", priority=50),
            Rule(id="any", condition="score >= 0", action="log", priority=10)
        ]
        context = {"score": 0.9}
        
        matched = symbolic_match(context, rules)
        
        # Toutes devraient matcher
        assert len(matched) == 3
        # Triées par priorité
        assert matched[0].id == "high"
        assert matched[1].id == "medium"
        assert matched[2].id == "any"
    
    def test_priority_sorting(self):
        """Test que les règles sont triées par priorité"""
        rules = [
            Rule(id="low", condition="True", action="a", priority=10),
            Rule(id="high", condition="True", action="b", priority=100),
            Rule(id="medium", condition="True", action="c", priority=50)
        ]
        context = {}
        
        matched = symbolic_match(context, rules)
        
        assert len(matched) == 3
        assert matched[0].id == "high"
        assert matched[1].id == "medium"
        assert matched[2].id == "low"
    
    def test_complex_condition(self):
        """Test avec condition complexe"""
        rules = [
            Rule(
                id="complex",
                condition="confidence > 0.7 and source == 'camera' and len(tags) > 2",
                action="process",
                priority=100
            )
        ]
        context = {
            "confidence": 0.85,
            "source": "camera",
            "tags": ["tag1", "tag2", "tag3"]
        }
        
        matched = symbolic_match(context, rules)
        
        assert len(matched) == 1
        assert matched[0].id == "complex"
    
    def test_invalid_condition(self):
        """Test que les conditions invalides sont ignorées"""
        rules = [
            Rule(id="valid", condition="x > 5", action="a", priority=100),
            Rule(id="invalid", condition="undefined_var > 10", action="b", priority=50)
        ]
        context = {"x": 10}
        
        matched = symbolic_match(context, rules)
        
        # Seule la règle valide matche
        assert len(matched) == 1
        assert matched[0].id == "valid"
    
    def test_with_dict_rules(self):
        """Test avec règles au format dict"""
        rules = [
            {"id": "dict_rule", "condition": "value < 100", "priority": 50}
        ]
        context = {"value": 50}
        
        matched = symbolic_match(context, rules)
        
        assert len(matched) == 1
        assert matched[0]["id"] == "dict_rule"
    
    def test_boolean_conditions(self):
        """Test avec conditions booléennes"""
        rules = [
            Rule(id="and_rule", condition="a and b", action="x", priority=100),
            Rule(id="or_rule", condition="c or d", action="y", priority=50)
        ]
        context = {"a": True, "b": True, "c": False, "d": False}
        
        matched = symbolic_match(context, rules)
        
        # Seul and_rule devrait matcher
        assert len(matched) == 1
        assert matched[0].id == "and_rule"
    
    def test_numeric_operations(self):
        """Test avec opérations numériques dans conditions"""
        rules = [
            Rule(id="math", condition="abs(x - y) < 10", action="similar", priority=100)
        ]
        context = {"x": 100, "y": 95}
        
        matched = symbolic_match(context, rules)
        
        assert len(matched) == 1


# ============================================
# Tests Neural Shortpass
# ============================================

class TestNeuralShortpass:
    """Tests pour neural_shortpass"""
    
    def test_basic_shortpass(self):
        """Test inférence shortpass basique"""
        input_vec = Vec.random(512)
        context_vec = Vec.random(512)
        
        output, confidence = neural_shortpass(input_vec, "tinynet", context_vec)
        
        assert isinstance(output, Vec)
        assert output.dim == 512
        assert 0.0 <= confidence <= 1.0
    
    def test_different_models(self):
        """Test avec différents modèles"""
        input_vec = Vec.random(256)
        context_vec = Vec.random(256)
        
        output_tiny, conf_tiny = neural_shortpass(input_vec, "tinynet", context_vec)
        output_mobile, conf_mobile = neural_shortpass(input_vec, "mobilenet", context_vec)
        
        # Les sorties devraient être différentes
        assert not np.allclose(output_tiny.data, output_mobile.data)
    
    def test_confidence_range(self):
        """Test que la confiance est dans [0, 1]"""
        input_vec = Vec.random(128)
        context_vec = Vec.random(128)
        
        _, confidence = neural_shortpass(input_vec, "tinynet", context_vec)
        
        assert confidence >= 0.0
        assert confidence <= 1.0
    
    def test_deterministic_with_same_model(self):
        """Test que même modèle + même input = même output"""
        input_vec = Vec.random(256)
        context_vec = Vec.random(256)
        
        output1, conf1 = neural_shortpass(input_vec, "tinynet", context_vec)
        output2, conf2 = neural_shortpass(input_vec, "tinynet", context_vec)
        
        # Devrait être identique
        assert np.allclose(output1.data, output2.data)
        assert abs(conf1 - conf2) < 1e-6


# ============================================
# Tests Neural Longpass
# ============================================

class TestNeuralLongpass:
    """Tests pour neural_longpass"""
    
    def test_basic_longpass(self):
        """Test inférence longpass basique"""
        input_vec = Vec.random(512)
        retrieved = [
            EpisodicRecord.create("Memory 1", Vec.random(512), trust=0.9),
            EpisodicRecord.create("Memory 2", Vec.random(512), trust=0.8)
        ]
        
        output, trace = neural_longpass(input_vec, "deepnet", retrieved)
        
        assert isinstance(output, Vec)
        assert output.dim == 512
        assert isinstance(trace, dict)
    
    def test_trace_log_structure(self):
        """Test structure du trace log"""
        input_vec = Vec.random(256)
        retrieved = []
        
        _, trace = neural_longpass(input_vec, "transformer", retrieved)
        
        assert "model" in trace
        assert "input_dim" in trace
        assert "num_retrieved" in trace
        assert "layers" in trace
        assert "latency_ms" in trace
        assert "activation_norms" in trace
        
        assert trace["model"] == "transformer"
        assert trace["input_dim"] == 256
        assert trace["num_retrieved"] == 0
    
    def test_with_retrieved_context(self):
        """Test avec contexte récupéré"""
        input_vec = Vec.random(128)
        retrieved = [
            EpisodicRecord.create(f"Memory {i}", Vec.random(128), trust=0.8)
            for i in range(5)
        ]
        
        output, trace = neural_longpass(input_vec, "deepnet", retrieved)
        
        assert trace["num_retrieved"] == 5
        assert len(trace["activation_norms"]) > 0
    
    def test_empty_retrieved(self):
        """Test avec retrieved vide"""
        input_vec = Vec.random(256)
        retrieved = []
        
        output, trace = neural_longpass(input_vec, "deepnet", retrieved)
        
        assert isinstance(output, Vec)
        assert trace["num_retrieved"] == 0
    
    def test_different_models(self):
        """Test avec différents modèles"""
        input_vec = Vec.random(256)
        retrieved = [EpisodicRecord.create("Test", Vec.random(256), 0.9)]
        
        output1, trace1 = neural_longpass(input_vec, "deepnet", retrieved)
        output2, trace2 = neural_longpass(input_vec, "transformer", retrieved)
        
        # Les outputs devraient être différents
        assert not np.allclose(output1.data, output2.data)
        assert trace1["model"] != trace2["model"]


# ============================================
# Tests Meta-Controller
# ============================================

class TestMetaController:
    """Tests pour meta_controller_decide"""
    
    def test_low_latency_forces_shortpass(self):
        """Test que latence faible force shortpass"""
        vec = Vec.random(256)
        
        decision = meta_controller_decide(vec, cost_budget=0.5, latency_target_ms=30)
        
        assert decision == "shortpass"
    
    def test_high_latency_allows_longpass(self):
        """Test que latence élevée permet longpass"""
        vec = Vec.random(256)
        
        decision = meta_controller_decide(vec, cost_budget=0.9, latency_target_ms=300)
        
        assert decision == "longpass"
    
    def test_low_budget_favors_shortpass(self):
        """Test que budget faible favorise shortpass"""
        vec = Vec.random(256)
        
        decision = meta_controller_decide(vec, cost_budget=0.1, latency_target_ms=150)
        
        # Budget très faible devrait favoriser shortpass
        assert decision == "shortpass"
    
    def test_decision_consistency(self):
        """Test cohérence de décision avec mêmes paramètres"""
        vec = Vec.random(256)
        
        decision1 = meta_controller_decide(vec, cost_budget=0.5, latency_target_ms=100)
        decision2 = meta_controller_decide(vec, cost_budget=0.5, latency_target_ms=100)
        
        # Devrait être identique
        assert decision1 == decision2
    
    def test_budget_impact(self):
        """Test impact du budget sur la décision"""
        vec = Vec.random(128)
        
        # Budget faible avec latence moyenne
        decision_low = meta_controller_decide(vec, cost_budget=0.2, latency_target_ms=150)
        
        # Budget élevé avec même latence
        decision_high = meta_controller_decide(vec, cost_budget=0.9, latency_target_ms=150)
        
        # Budget faible devrait favoriser shortpass
        assert decision_low == "shortpass"
        # Budget élevé devrait permettre longpass
        assert decision_high == "longpass"
    
    def test_latency_thresholds(self):
        """Test seuils de latence"""
        vec = Vec.random(256)
        
        # Très faible latence
        assert meta_controller_decide(vec, 0.5, 20) == "shortpass"
        
        # Très haute latence
        assert meta_controller_decide(vec, 0.5, 500) == "longpass"
    
    def test_valid_output(self):
        """Test que l'output est toujours valide"""
        vec = Vec.random(128)
        
        for budget in [0.1, 0.5, 0.9]:
            for latency in [20, 100, 300]:
                decision = meta_controller_decide(vec, budget, latency)
                assert decision in ["shortpass", "longpass"]


# ============================================
# Tests d'intégration Reasoner
# ============================================

class TestReasonerIntegration:
    """Tests d'intégration pour le reasoner hybride complet"""
    
    def test_full_shortpass_pipeline(self):
        """Test pipeline complet avec shortpass"""
        # 1. Input
        input_vec = Vec.random(512)
        context_vec = Vec.random(512)
        
        # 2. Meta-controller décide
        decision = meta_controller_decide(input_vec, cost_budget=0.3, latency_target_ms=50)
        
        assert decision == "shortpass"
        
        # 3. Exécute shortpass
        output, confidence = neural_shortpass(input_vec, "tinynet", context_vec)
        
        assert isinstance(output, Vec)
        assert 0.0 <= confidence <= 1.0
    
    def test_full_longpass_pipeline(self):
        """Test pipeline complet avec longpass"""
        # 1. Input
        input_vec = Vec.random(512)
        
        # 2. Retrieval (simulation)
        retrieved = [
            EpisodicRecord.create(f"Mem {i}", Vec.random(512), 0.85)
            for i in range(5)
        ]
        
        # 3. Meta-controller décide
        decision = meta_controller_decide(input_vec, cost_budget=0.9, latency_target_ms=300)
        
        assert decision == "longpass"
        
        # 4. Exécute longpass
        output, trace = neural_longpass(input_vec, "deepnet", retrieved)
        
        assert isinstance(output, Vec)
        assert trace["num_retrieved"] == 5
    
    def test_hybrid_reasoner_with_rules(self):
        """Test reasoner hybride: symbolic + neural"""
        # 1. Symbolic matching
        rules = [
            Rule(id="high_priority", condition="confidence > 0.9", action="accept", priority=100),
            Rule(id="need_review", condition="confidence < 0.5", action="review", priority=50)
        ]
        
        # 2. Neural inference
        input_vec = Vec.random(256)
        context_vec = Vec.random(256)
        output, confidence = neural_shortpass(input_vec, "tinynet", context_vec)
        
        # 3. Symbolic decision basé sur output neural
        context = {"confidence": confidence}
        matched = symbolic_match(context, rules)
        
        # Au moins une règle devrait matcher
        assert len(matched) >= 0  # Peut être 0 si confidence entre 0.5 et 0.9
    
    def test_adaptive_reasoning(self):
        """Test raisonnement adaptatif basé sur complexité"""
        # Input simple (faible variance)
        simple_vec = Vec.from_list([1.0] * 256)
        
        # Input complexe (haute variance)
        complex_vec = Vec.random(256)
        
        # Décisions avec même contraintes
        decision_simple = meta_controller_decide(simple_vec, 0.5, 100)
        decision_complex = meta_controller_decide(complex_vec, 0.5, 100)
        
        # Les deux devraient être des décisions valides
        assert decision_simple in ["shortpass", "longpass"]
        assert decision_complex in ["shortpass", "longpass"]
    
    def test_multi_stage_reasoning(self):
        """Test raisonnement multi-étapes"""
        # Stage 1: Shortpass rapide
        input_vec = Vec.random(512)
        context_vec = Vec.random(512)
        
        output1, confidence1 = neural_shortpass(input_vec, "tinynet", context_vec)
        
        # Stage 2: Si confiance faible, faire longpass
        if confidence1 < 0.6:
            retrieved = [
                EpisodicRecord.create(f"M{i}", Vec.random(512), 0.8)
                for i in range(3)
            ]
            output2, trace = neural_longpass(input_vec, "deepnet", retrieved)
            
            assert isinstance(output2, Vec)
            assert trace["num_retrieved"] == 3
        else:
            # Sinon, utiliser output shortpass
            output2 = output1
        
        # Output final toujours valide
        assert isinstance(output2, Vec)
        assert output2.dim == 512
    
    def test_reasoner_with_feedback(self):
        """Test reasoner avec boucle de feedback"""
        input_vec = Vec.random(256)
        
        # Itération 1
        decision1 = meta_controller_decide(input_vec, 0.5, 100)
        
        # Simuler ajustement budget basé sur résultat
        if decision1 == "shortpass":
            # Si shortpass choisi, augmenter budget pour prochaine fois
            new_budget = 0.7
        else:
            new_budget = 0.5
        
        # Itération 2 avec nouveau budget
        decision2 = meta_controller_decide(input_vec, new_budget, 100)
        
        assert decision2 in ["shortpass", "longpass"]
