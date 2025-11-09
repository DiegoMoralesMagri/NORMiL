"""
Tests pour les primitives neurales - Phase 6
============================================

Tests unitaires pour :
- lowrankupdate() : Mise à jour de rang faible
- quantize() : Quantisation 8/4 bits
- onlinecluster_update() : Clustering incrémental
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier parent au path
NORMIL_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(NORMIL_ROOT))

import pytest
import numpy as np
from runtime.primitives import (
    lowrankupdate,
    quantize,
    onlinecluster_update,
    vec,
    zeros,
    ones,
    random
)
from runtime.normil_types import Vec


class TestLowRankUpdate:
    """Tests pour lowrankupdate()"""
    
    def test_basic_update_2x2(self):
        """Test mise à jour basique sur matrice 2x2"""
        # Matrice identité
        W = np.array([[1.0, 0.0], [0.0, 1.0]])
        u = vec(2, [1.0, 0.0])
        v = vec(2, [0.0, 1.0])
        
        # W' = W + u⊗v
        W_new = lowrankupdate(W, u, v)
        
        # Vérifications
        expected = np.array([[1.0, 1.0], [0.0, 1.0]])
        np.testing.assert_array_almost_equal(W_new, expected)
        
    def test_update_3x3(self):
        """Test mise à jour sur matrice 3x3"""
        W = np.eye(3)
        u = vec(3, [0.5, 0.5, 0.0])
        v = vec(3, [1.0, 0.0, 0.0])
        
        W_new = lowrankupdate(W, u, v)
        
        # Vérifier que la première colonne a changé
        expected = np.array([
            [1.5, 0.0, 0.0],
            [0.5, 1.0, 0.0],
            [0.0, 0.0, 1.0]
        ])
        np.testing.assert_array_almost_equal(W_new, expected)
        
    def test_symmetric_update(self):
        """Test mise à jour symétrique u=v"""
        W = np.zeros((3, 3))
        u = vec(3, [1.0, 0.0, 0.0])
        
        # u⊗u donne une matrice de rang 1
        W_new = lowrankupdate(W, u, u)
        
        # Seul le coin (0,0) devrait être 1.0
        assert W_new[0, 0] == 1.0
        assert W_new[0, 1] == 0.0
        assert W_new[1, 0] == 0.0
        
    def test_multiple_updates(self):
        """Test accumulation de plusieurs mises à jour"""
        W = np.eye(2)
        
        u1 = vec(2, [1.0, 0.0])
        v1 = vec(2, [0.0, 1.0])
        W = lowrankupdate(W, u1, v1)
        
        u2 = vec(2, [0.0, 1.0])
        v2 = vec(2, [1.0, 0.0])
        W = lowrankupdate(W, u2, v2)
        
        # Après deux updates
        expected = np.array([[1.0, 1.0], [1.0, 1.0]])
        np.testing.assert_array_almost_equal(W, expected)
        
    def test_preserves_shape(self):
        """Test que la forme de la matrice est préservée"""
        for n in [2, 5, 10]:
            W = np.eye(n)
            u = vec(n, [1.0] * n)
            v = vec(n, [0.5] * n)
            
            W_new = lowrankupdate(W, u, v)
            
            assert W_new.shape == (n, n)


class TestQuantize:
    """Tests pour quantize()"""
    
    def test_quantize_8bit_simple(self):
        """Test quantisation 8 bits sur valeurs simples"""
        v = vec(5, [0.0, 0.25, 0.5, 0.75, 1.0])
        v_q = quantize(v, 8)
        
        # Vérifier dimension préservée
        assert v_q.dim == 5
        
        # Vérifier que les valeurs sont proches
        np.testing.assert_allclose(v_q.data, v.data, rtol=0.01)
        
    def test_quantize_4bit_simple(self):
        """Test quantisation 4 bits (plus agressive)"""
        v = vec(5, [0.0, 0.25, 0.5, 0.75, 1.0])
        v_q = quantize(v, 4)
        
        # Vérifier dimension préservée
        assert v_q.dim == 5
        
        # 4 bits = 16 niveaux, donc moins précis
        # Tolérance plus élevée
        np.testing.assert_allclose(v_q.data, v.data, rtol=0.1)
        
    def test_quantize_preserves_dimension(self):
        """Test que la dimension est toujours préservée"""
        for dim in [8, 16, 32, 128]:
            v = random(dim, 0.0, 1.0)
            
            v_q8 = quantize(v, 8)
            v_q4 = quantize(v, 4)
            
            assert v_q8.dim == dim
            assert v_q4.dim == dim
            
    def test_quantize_range_preservation(self):
        """Test que le range min/max est préservé"""
        v = vec(10, list(range(10)))  # [0, 1, 2, ..., 9]
        v_q = quantize(v, 8)
        
        # Min et max doivent être proches
        assert abs(v_q.data.min() - v.data.min()) < 0.1
        assert abs(v_q.data.max() - v.data.max()) < 0.1
        
    def test_quantize_constant_vector(self):
        """Test quantisation d'un vecteur constant"""
        v = vec(5, [3.14] * 5)
        v_q = quantize(v, 8)
        
        # Tous les éléments doivent rester égaux
        assert np.allclose(v_q.data, v.data)
        
    def test_quantize_precision_8bit_vs_4bit(self):
        """Test que 8-bit est plus précis que 4-bit"""
        v = random(20, 0.0, 1.0)
        
        v_q8 = quantize(v, 8)
        v_q4 = quantize(v, 4)
        
        # Erreur de quantification
        error_8bit = np.mean(np.abs(v.data - v_q8.data))
        error_4bit = np.mean(np.abs(v.data - v_q4.data))
        
        # 8-bit devrait être plus précis
        assert error_8bit < error_4bit


class TestOnlineClusterUpdate:
    """Tests pour onlinecluster_update()"""
    
    def test_single_update_basic(self):
        """Test mise à jour incrémentale basique"""
        centroid = zeros(3)
        x = vec(3, [1.0, 0.0, 0.0])
        
        # Avec lr=0.5, on devrait obtenir [0.5, 0, 0]
        c_new = onlinecluster_update(centroid, x, 0.5)
        
        expected = np.array([0.5, 0.0, 0.0])
        np.testing.assert_array_almost_equal(c_new.data, expected)
        
    def test_learning_rate_extremes(self):
        """Test avec learning rates extrêmes"""
        centroid = zeros(3)
        x = ones(3)
        
        # lr=0 : pas de changement
        c0 = onlinecluster_update(centroid, x, 0.0)
        np.testing.assert_array_almost_equal(c0.data, centroid.data)
        
        # lr=1 : remplacement complet
        c1 = onlinecluster_update(centroid, x, 1.0)
        np.testing.assert_array_almost_equal(c1.data, x.data)
        
    def test_learning_rate_impact(self):
        """Test impact du learning rate"""
        base = zeros(4)
        target = ones(4)
        
        c_lr01 = onlinecluster_update(base, target, 0.1)
        c_lr05 = onlinecluster_update(base, target, 0.5)
        c_lr09 = onlinecluster_update(base, target, 0.9)
        
        # Vérifier que plus lr est grand, plus on se rapproche de target
        # Note: float16 a une précision limitée, utiliser rtol
        assert np.allclose(c_lr01.data, [0.1, 0.1, 0.1, 0.1], rtol=0.01)
        assert np.allclose(c_lr05.data, [0.5, 0.5, 0.5, 0.5], rtol=0.01)
        assert np.allclose(c_lr09.data, [0.9, 0.9, 0.9, 0.9], rtol=0.01)
        
    def test_convergence_sequence(self):
        """Test convergence sur une séquence de points"""
        c = zeros(2)
        
        # Séquence de mises à jour
        x1 = vec(2, [1.0, 0.0])
        x2 = vec(2, [0.0, 1.0])
        x3 = vec(2, [1.0, 1.0])
        
        lr = 0.3
        c = onlinecluster_update(c, x1, lr)
        c = onlinecluster_update(c, x2, lr)
        c = onlinecluster_update(c, x3, lr)
        
        # Le centroïde devrait être quelque part au milieu
        assert c.data[0] > 0.0
        assert c.data[1] > 0.0
        
    def test_convergence_to_mean(self):
        """Test convergence vers la moyenne sur plusieurs steps"""
        c = zeros(3)
        target = vec(3, [2.0, 3.0, 1.0])
        lr = 0.2
        
        # 10 mises à jour avec le même point
        for _ in range(10):
            c = onlinecluster_update(c, target, lr)
        
        # Devrait être proche de target (formule: (1-lr)^n -> 0 quand n -> inf)
        # Après 10 steps: c ≈ target * (1 - (1-lr)^10)
        expected_factor = 1 - (0.8 ** 10)  # 1 - (1-0.2)^10
        expected = target.data * expected_factor
        
        np.testing.assert_allclose(c.data, expected, rtol=0.01)
        
    def test_dimension_mismatch_error(self):
        """Test erreur si dimensions incompatibles"""
        c = zeros(3)
        x = zeros(5)
        
        with pytest.raises(ValueError, match="Dimension mismatch"):
            onlinecluster_update(c, x, 0.5)
            
    def test_invalid_learning_rate(self):
        """Test erreur si learning rate invalide"""
        c = zeros(3)
        x = ones(3)
        
        # lr < 0
        with pytest.raises(ValueError, match="Learning rate must be in"):
            onlinecluster_update(c, x, -0.1)
            
        # lr > 1
        with pytest.raises(ValueError, match="Learning rate must be in"):
            onlinecluster_update(c, x, 1.5)
            
    def test_preserves_dimension(self):
        """Test que la dimension est préservée"""
        for dim in [3, 8, 16, 128]:
            c = zeros(dim)
            x = random(dim, 0.0, 1.0)
            
            c_new = onlinecluster_update(c, x, 0.3)
            
            assert c_new.dim == dim


class TestCombinedScenarios:
    """Tests de scénarios combinés"""
    
    def test_clustering_then_quantization(self):
        """Test clustering suivi de quantization"""
        # Créer un cluster center
        cluster_center = zeros(8)
        
        # Ajouter 3 points
        for _ in range(3):
            sample = random(8, 0.0, 1.0)
            cluster_center = onlinecluster_update(cluster_center, sample, 0.3)
        
        # Quantifier le résultat
        cluster_q = quantize(cluster_center, 8)
        
        # Vérifier dimension et que les valeurs sont similaires
        assert cluster_q.dim == 8
        # float16 + quantization sur petites valeurs = tolérance élevée
        np.testing.assert_allclose(cluster_center.data, cluster_q.data, rtol=0.2, atol=0.01)
        
    def test_lowrank_with_quantized_vectors(self):
        """Test low-rank update avec vecteurs quantifiés"""
        W = np.eye(4)
        
        # Créer vecteurs et les quantifier
        u = random(4, 0.0, 1.0)
        v = random(4, 0.0, 1.0)
        
        u_q = quantize(u, 8)
        v_q = quantize(v, 8)
        
        # Update avec vecteurs quantifiés
        W_new = lowrankupdate(W, u_q, v_q)
        
        # Vérifier que ça fonctionne
        assert W_new.shape == (4, 4)
        
    def test_adaptive_learning_rate(self):
        """Test adaptation du learning rate pendant clustering"""
        c = zeros(5)
        
        # Simuler un learning rate décroissant
        lr_schedule = [0.5, 0.3, 0.2, 0.1, 0.05]
        
        for lr in lr_schedule:
            x = random(5, 0.0, 1.0)
            c = onlinecluster_update(c, x, lr)
        
        # Le centroïde devrait avoir convergé quelque part
        assert not np.allclose(c.data, 0.0)


class TestNumericalStability:
    """Tests de stabilité numérique"""
    
    def test_quantize_with_large_values(self):
        """Test quantization avec grandes valeurs"""
        v = vec(5, [1000.0, 2000.0, 3000.0, 4000.0, 5000.0])
        v_q = quantize(v, 8)
        
        # Range doit être préservé
        assert abs(v_q.data.min() - 1000.0) < 10
        assert abs(v_q.data.max() - 5000.0) < 10
        
    def test_quantize_with_negative_values(self):
        """Test quantization avec valeurs négatives"""
        v = vec(5, [-2.0, -1.0, 0.0, 1.0, 2.0])
        v_q = quantize(v, 8)
        
        # Devrait gérer les négatifs correctement
        # Note: float16 + quantization = tolérance plus élevée pour 0.0
        np.testing.assert_allclose(v_q.data, v.data, rtol=0.05, atol=0.01)
        
    def test_clustering_numerical_precision(self):
        """Test précision numérique du clustering"""
        c = zeros(3)
        
        # Beaucoup d'updates avec petits lr
        for _ in range(100):
            x = vec(3, [0.01, 0.01, 0.01])
            c = onlinecluster_update(c, x, 0.01)
        
        # Ne devrait pas diverger
        assert np.all(np.isfinite(c.data))
        assert np.all(c.data >= 0.0)
        assert np.all(c.data <= 1.0)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
