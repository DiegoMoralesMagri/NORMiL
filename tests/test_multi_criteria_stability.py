"""
Tests pour les multi-critères de stabilité (Phase 7.8)
"""
import pytest
import numpy as np
from runtime.primitives import compute_stability_window, compute_weight_variance
from runtime.normil_types import Vec


class TestStabilityWindow:
    """Tests pour compute_stability_window"""
    
    def test_empty_history(self):
        """Historique vide doit retourner True"""
        result = compute_stability_window([], 0.01)
        assert result is True
    
    def test_single_weight(self):
        """Un seul poids doit retourner True"""
        w1 = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        result = compute_stability_window([w1], 0.01)
        assert result is True
    
    def test_stable_window(self):
        """Petits changements consécutifs = stable"""
        w1 = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        w2 = Vec(np.array([1.001, 2.001, 3.001], dtype=np.float16), 3)
        w3 = Vec(np.array([1.002, 2.002, 3.002], dtype=np.float16), 3)
        
        result = compute_stability_window([w1, w2, w3], 0.01)
        assert result is True
    
    def test_unstable_window(self):
        """Grand changement au milieu = instable"""
        w1 = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        w2 = Vec(np.array([1.001, 2.001, 3.001], dtype=np.float16), 3)
        w3 = Vec(np.array([2.0, 4.0, 6.0], dtype=np.float16), 3)
        w4 = Vec(np.array([2.001, 4.001, 6.001], dtype=np.float16), 3)
        
        result = compute_stability_window([w1, w2, w3, w4], 0.01)
        assert result is False
    
    def test_threshold_sensitivity(self):
        """Le seuil doit influencer la détection"""
        w1 = Vec(np.array([1.0, 1.0], dtype=np.float16), 2)
        w2 = Vec(np.array([1.05, 1.05], dtype=np.float16), 2)
        
        # Avec seuil strict (0.01), c'est instable
        result_strict = compute_stability_window([w1, w2], 0.01)
        assert result_strict is False
        
        # Avec seuil permissif (0.1), c'est stable
        result_permissive = compute_stability_window([w1, w2], 0.1)
        assert result_permissive is True
    
    def test_long_stable_sequence(self):
        """Longue séquence stable"""
        weights = []
        for i in range(10):
            w = Vec(np.array([1.0 + i*0.0001, 2.0 + i*0.0001], dtype=np.float16), 2)
            weights.append(w)
        
        result = compute_stability_window(weights, 0.01)
        assert result is True


class TestWeightVariance:
    """Tests pour compute_weight_variance"""
    
    def test_empty_history(self):
        """Historique vide doit retourner 0"""
        result = compute_weight_variance([])
        assert result == 0.0
    
    def test_single_weight(self):
        """Un seul poids doit retourner 0"""
        w1 = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        result = compute_weight_variance([w1])
        assert result == 0.0
    
    def test_low_variance(self):
        """Poids similaires = faible variance"""
        w1 = Vec(np.array([1.0, 1.0], dtype=np.float16), 2)
        w2 = Vec(np.array([1.01, 1.01], dtype=np.float16), 2)
        w3 = Vec(np.array([1.02, 1.02], dtype=np.float16), 2)
        
        variance = compute_weight_variance([w1, w2, w3])
        assert variance < 0.001  # Très faible
    
    def test_high_variance(self):
        """Poids très différents = haute variance"""
        w1 = Vec(np.array([1.0, 1.0], dtype=np.float16), 2)
        w2 = Vec(np.array([2.0, 0.5], dtype=np.float16), 2)
        w3 = Vec(np.array([0.5, 2.5], dtype=np.float16), 2)
        
        variance = compute_weight_variance([w1, w2, w3])
        assert variance > 0.1  # Significative
    
    def test_variance_comparison(self):
        """Comparer variances de séquences différentes"""
        # Séquence stable
        stable = [
            Vec(np.array([1.0, 1.0], dtype=np.float16), 2),
            Vec(np.array([1.01, 1.01], dtype=np.float16), 2),
            Vec(np.array([1.02, 1.02], dtype=np.float16), 2)
        ]
        
        # Séquence variable
        variable = [
            Vec(np.array([1.0, 1.0], dtype=np.float16), 2),
            Vec(np.array([2.0, 0.5], dtype=np.float16), 2),
            Vec(np.array([0.5, 2.0], dtype=np.float16), 2)
        ]
        
        var_stable = compute_weight_variance(stable)
        var_variable = compute_weight_variance(variable)
        
        assert var_stable < var_variable
    
    def test_constant_weights(self):
        """Poids constants = variance nulle"""
        w = Vec(np.array([1.0, 2.0, 3.0], dtype=np.float16), 3)
        weights = [w, w, w, w, w]
        
        variance = compute_weight_variance(weights)
        assert variance == pytest.approx(0.0, abs=1e-6)


class TestCombinedCriteria:
    """Tests pour combinaison de critères"""
    
    def test_stable_by_both_criteria(self):
        """Stable selon window ET variance"""
        w1 = Vec(np.array([1.0, 1.0], dtype=np.float16), 2)
        w2 = Vec(np.array([1.001, 1.001], dtype=np.float16), 2)
        w3 = Vec(np.array([1.002, 1.002], dtype=np.float16), 2)
        weights = [w1, w2, w3]
        
        window_stable = compute_stability_window(weights, 0.01)
        variance = compute_weight_variance(weights)
        
        assert window_stable is True
        assert variance < 0.001
    
    def test_unstable_by_window_only(self):
        """Instable par window mais variance OK"""
        w1 = Vec(np.array([1.0, 1.0], dtype=np.float16), 2)
        w2 = Vec(np.array([1.5, 1.5], dtype=np.float16), 2)  # Grand saut
        w3 = Vec(np.array([1.501, 1.501], dtype=np.float16), 2)
        weights = [w1, w2, w3]
        
        window_stable = compute_stability_window(weights, 0.01)
        
        # Le grand saut rend la fenêtre instable
        assert window_stable is False
    
    def test_unstable_by_variance_only(self):
        """Variance élevée avec fenêtre stable"""
        # Oscillations constantes mais petites entre chaque step
        w1 = Vec(np.array([1.0, 0.0], dtype=np.float16), 2)
        w2 = Vec(np.array([1.005, 0.005], dtype=np.float16), 2)
        w3 = Vec(np.array([0.995, -0.005], dtype=np.float16), 2)
        w4 = Vec(np.array([1.005, 0.005], dtype=np.float16), 2)
        weights = [w1, w2, w3, w4]
        
        variance = compute_weight_variance(weights)
        
        # Haute variance due aux oscillations
        assert variance > 0.0


class TestEdgeCases:
    """Tests de cas limites"""
    
    def test_dimension_consistency(self):
        """Les vecteurs doivent avoir la même dimension"""
        w1 = Vec(np.array([1.0, 2.0], dtype=np.float16), 2)
        w2 = Vec(np.array([1.0, 2.0], dtype=np.float16), 2)
        
        # Doit fonctionner normalement
        result = compute_stability_window([w1, w2], 0.01)
        assert result is True
    
    def test_very_small_changes(self):
        """Changements très petits (précision numérique)"""
        w1 = Vec(np.array([1.0, 1.0], dtype=np.float16), 2)
        w2 = Vec(np.array([1.0000001, 1.0000001], dtype=np.float16), 2)
        
        variance = compute_weight_variance([w1, w2])
        
        # Variance doit être proche de 0
        assert variance < 1e-3
    
    def test_large_dimension(self):
        """Test avec grande dimension"""
        dim = 128
        w1 = Vec(np.ones(dim, dtype=np.float16), dim)
        w2 = Vec(np.ones(dim, dtype=np.float16) * 1.001, dim)
        
        window_stable = compute_stability_window([w1, w2], 0.01)
        variance = compute_weight_variance([w1, w2])
        
        assert window_stable is True
        assert variance < 0.001


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
