"""
Tests pour les primitives de plasticité (Phase 7.3)
Tests unitaires pour normalize_plasticity, decay_learning_rate, compute_stability
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier parent au path
NORMIL_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(NORMIL_ROOT))

import pytest
import numpy as np
from runtime.normil_types import Vec
from runtime.primitives import (
    normalize_plasticity,
    decay_learning_rate,
    compute_stability
)


class TestNormalizePlasticity:
    """Tests pour la primitive normalize_plasticity"""
    
    def test_normalize_basic(self):
        """Test normalisation basique d'un vecteur"""
        v = Vec(np.array([3.0, 4.0], dtype=np.float16), 2)
        normalized = normalize_plasticity(v)
        
        # Norme devrait être 1.0
        norm = np.linalg.norm(normalized.data)
        assert abs(norm - 1.0) < 0.01, f"Expected norm ~1.0, got {norm}"
        
        # Direction préservée (ratio 3:4)
        ratio = normalized.data[0] / normalized.data[1]
        expected_ratio = 3.0 / 4.0
        assert abs(ratio - expected_ratio) < 0.01
    
    def test_normalize_already_normalized(self):
        """Test normalisation d'un vecteur déjà normalisé"""
        v = Vec(np.array([0.6, 0.8], dtype=np.float16), 2)
        normalized = normalize_plasticity(v)
        
        norm = np.linalg.norm(normalized.data)
        assert abs(norm - 1.0) < 0.01
    
    def test_normalize_zero_vector(self):
        """Test normalisation d'un vecteur nul (edge case)"""
        v = Vec(np.array([0.0, 0.0, 0.0], dtype=np.float16), 3)
        normalized = normalize_plasticity(v)
        
        # Devrait retourner le vecteur inchangé
        assert np.allclose(normalized.data, v.data, atol=1e-6)
    
    def test_normalize_large_vector(self):
        """Test normalisation d'un grand vecteur"""
        data = np.random.randn(100).astype(np.float16)
        v = Vec(data, 100)
        normalized = normalize_plasticity(v)
        
        norm = np.linalg.norm(normalized.data)
        assert abs(norm - 1.0) < 0.05  # Tolérance pour float16
    
    def test_normalize_single_element(self):
        """Test normalisation vecteur 1D"""
        v = Vec(np.array([5.0], dtype=np.float16), 1)
        normalized = normalize_plasticity(v)
        
        assert abs(abs(normalized.data[0]) - 1.0) < 0.01
    
    def test_normalize_preserves_dimension(self):
        """Test que la dimension est préservée"""
        v = Vec(np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float16), 4)
        normalized = normalize_plasticity(v)
        
        assert normalized.dim == v.dim


class TestDecayLearningRate:
    """Tests pour la primitive decay_learning_rate"""
    
    def test_decay_basic(self):
        """Test decay basique avec facteur 0.9"""
        lr = 0.1
        decayed = decay_learning_rate(lr, 0.9)
        
        expected = 0.1 * 0.9
        assert abs(decayed - expected) < 1e-6
    
    def test_decay_progressive(self):
        """Test decay progressif sur plusieurs étapes"""
        lr = 1.0
        factor = 0.95
        
        for i in range(10):
            lr = decay_learning_rate(lr, factor)
        
        expected = 1.0 * (0.95 ** 10)
        assert abs(lr - expected) < 1e-6
    
    def test_decay_factor_one(self):
        """Test avec facteur = 1.0 (pas de decay)"""
        lr = 0.5
        decayed = decay_learning_rate(lr, 1.0)
        
        assert decayed == lr
    
    def test_decay_factor_invalid_zero(self):
        """Test facteur invalide (0.0)"""
        with pytest.raises(ValueError):
            decay_learning_rate(0.1, 0.0)
    
    def test_decay_factor_invalid_negative(self):
        """Test facteur invalide (négatif)"""
        with pytest.raises(ValueError):
            decay_learning_rate(0.1, -0.5)
    
    def test_decay_factor_invalid_greater_than_one(self):
        """Test facteur invalide (>1.0)"""
        with pytest.raises(ValueError):
            decay_learning_rate(0.1, 1.5)
    
    def test_decay_very_small_lr(self):
        """Test decay avec learning rate très petit"""
        lr = 1e-6
        decayed = decay_learning_rate(lr, 0.99)
        
        assert decayed < lr
        assert decayed > 0
    
    def test_decay_different_factors(self):
        """Test différents facteurs de decay"""
        lr = 0.5
        
        slow_decay = decay_learning_rate(lr, 0.99)
        medium_decay = decay_learning_rate(lr, 0.95)
        fast_decay = decay_learning_rate(lr, 0.90)
        
        assert slow_decay > medium_decay > fast_decay


class TestComputeStability:
    """Tests pour la primitive compute_stability"""
    
    def test_stability_no_change(self):
        """Test stabilité avec vecteurs identiques"""
        v = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        
        is_stable = compute_stability(v, v, 0.01)
        assert is_stable == True
    
    def test_stability_small_change(self):
        """Test stabilité avec petit changement"""
        v1 = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        v2 = Vec(np.array([1.001, 2.001, 3.001], dtype=np.float16), 3)
        
        is_stable = compute_stability(v1, v2, 0.01)
        assert is_stable == True
    
    def test_stability_large_change(self):
        """Test instabilité avec grand changement"""
        v1 = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        v2 = Vec(np.array([1.5, 3.0, 4.5], dtype=np.float16), 3)
        
        is_stable = compute_stability(v1, v2, 0.01)
        assert is_stable == False
    
    def test_stability_threshold_sensitivity(self):
        """Test sensibilité au seuil"""
        v1 = Vec(np.array([1.0, 2.0], dtype=np.float16), 2)
        v2 = Vec(np.array([1.02, 2.04], dtype=np.float16), 2)
        
        # Avec seuil strict, devrait être instable
        assert compute_stability(v1, v2, 0.01) == False
        
        # Avec seuil large, devrait être stable
        assert compute_stability(v1, v2, 0.05) == True
    
    def test_stability_zero_vector_old(self):
        """Test avec vecteur initial nul"""
        v1 = Vec(np.array([0.0, 0.0, 0.0], dtype=np.float16), 3)
        v2 = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        
        # Vecteur nul considéré comme stable
        is_stable = compute_stability(v1, v2, 0.01)
        assert is_stable == True
    
    def test_stability_dimension_mismatch(self):
        """Test avec dimensions différentes"""
        v1 = Vec(np.array([1.0, 2.0], dtype=np.float16), 2)
        v2 = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        
        with pytest.raises(ValueError):
            compute_stability(v1, v2, 0.01)
    
    def test_stability_relative_change(self):
        """Test calcul du changement relatif"""
        # Vecteur avec norme 5
        v1 = Vec(np.array([3.0, 4.0], dtype=np.float16), 2)
        # Vecteur avec changement de 0.25 (5% de la norme)
        v2 = Vec(np.array([3.15, 4.20], dtype=np.float16), 2)
        
        # Devrait être stable avec seuil 10%
        assert compute_stability(v1, v2, 0.10) == True
        
        # Devrait être instable avec seuil 1%
        assert compute_stability(v1, v2, 0.01) == False


class TestPlasticityCombined:
    """Tests combinés pour scénarios réalistes"""
    
    def test_training_loop_simulation(self):
        """Simulation d'une boucle d'entraînement"""
        # Poids initiaux aléatoires
        weights = Vec(np.random.randn(10).astype(np.float16), 10)
        lr = 0.1
        
        for i in range(20):
            # Sauvegarder poids avant
            old_weights = weights
            
            # Simuler update (petit changement aléatoire)
            noise = np.random.randn(10).astype(np.float16) * lr
            new_data = weights.data + noise
            weights = Vec(new_data, 10)
            
            # Normaliser
            weights = normalize_plasticity(weights)
            
            # Vérifier stabilité
            if compute_stability(old_weights, weights, 0.01):
                break
            
            # Decay learning rate
            lr = decay_learning_rate(lr, 0.95)
        
        # Vérifier que les poids sont normalisés
        norm = np.linalg.norm(weights.data)
        assert abs(norm - 1.0) < 0.05
        
        # Vérifier que le LR a décru
        assert lr < 0.1
    
    def test_convergence_detection(self):
        """Test détection de convergence vers une cible"""
        target = Vec(np.array([1.0, 0.0, 0.0], dtype=np.float16), 3)
        weights = Vec(np.array([0.5, 0.5, 0.5], dtype=np.float16), 3)
        
        converged = False
        max_steps = 100
        
        for step in range(max_steps):
            old_weights = weights
            
            # Déplacement vers la cible
            direction = target.data - weights.data
            weights = Vec(weights.data + direction * 0.1, 3)
            weights = normalize_plasticity(weights)
            
            # Vérifier convergence
            if compute_stability(old_weights, weights, 0.001):
                converged = True
                break
        
        assert converged, "Should have converged within 100 steps"
        
        # Vérifier que les poids sont normalisés
        norm = np.linalg.norm(weights.data)
        assert abs(norm - 1.0) < 0.05
    
    def test_stability_after_normalization(self):
        """Test que la normalisation seule ne change pas trop les poids"""
        # Vecteur déjà proche de la norme 1
        weights = Vec(np.array([0.6, 0.8], dtype=np.float16), 2)
        normalized = normalize_plasticity(weights)
        
        # Devrait être considéré comme stable
        is_stable = compute_stability(weights, normalized, 0.05)
        assert is_stable == True


class TestEdgeCases:
    """Tests de cas limites"""
    
    def test_normalize_very_small_values(self):
        """Test normalisation de valeurs très petites"""
        v = Vec(np.array([1e-5, 1e-5, 1e-5], dtype=np.float16), 3)
        normalized = normalize_plasticity(v)
        
        # Ne devrait pas exploser
        assert not np.any(np.isnan(normalized.data))
        assert not np.any(np.isinf(normalized.data))
    
    def test_decay_accumulation(self):
        """Test accumulation de decay sur beaucoup d'étapes"""
        lr = 1.0
        
        for _ in range(1000):
            lr = decay_learning_rate(lr, 0.999)
        
        # Devrait avoir fortement décru mais rester > 0
        assert lr > 0
        assert lr < 0.5
    
    def test_stability_numerical_precision(self):
        """Test précision numérique dans compute_stability"""
        # Vecteurs très proches mais différents due à float16
        v1 = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        v2 = Vec((np.array([1.0, 2.0, 3.0], dtype=np.float16) * 1.0001).astype(np.float16), 3)
        
        # Devrait gérer gracieusement la précision limitée
        result = compute_stability(v1, v2, 0.001)
        assert isinstance(result, bool)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
