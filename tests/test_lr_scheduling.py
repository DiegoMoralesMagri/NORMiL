"""
Tests pour les primitives de Learning Rate Scheduling (Phase 7.9)
"""
import pytest
from runtime.primitives import (
    lr_warmup_linear,
    lr_cosine_annealing,
    lr_step_decay,
    lr_plateau_factor
)


class TestWarmupLinear:
    """Tests pour lr_warmup_linear"""
    
    def test_warmup_start(self):
        """Le LR doit commencer à 0"""
        lr = lr_warmup_linear(0, 10, 0.01)
        assert lr == 0.0
    
    def test_warmup_middle(self):
        """Le LR doit être à mi-chemin au milieu du warmup"""
        lr = lr_warmup_linear(5, 10, 0.01)
        assert lr == pytest.approx(0.005, abs=1e-6)
    
    def test_warmup_end(self):
        """Le LR doit atteindre la cible à la fin du warmup"""
        lr = lr_warmup_linear(10, 10, 0.01)
        assert lr == pytest.approx(0.01, abs=1e-6)
    
    def test_warmup_after(self):
        """Le LR doit rester constant après le warmup"""
        lr = lr_warmup_linear(15, 10, 0.01)
        assert lr == pytest.approx(0.01, abs=1e-6)
    
    def test_warmup_linear_progression(self):
        """Le LR doit augmenter linéairement"""
        lr_0 = lr_warmup_linear(0, 10, 0.01)
        lr_5 = lr_warmup_linear(5, 10, 0.01)
        lr_10 = lr_warmup_linear(10, 10, 0.01)
        
        assert lr_0 < lr_5 < lr_10
    
    def test_warmup_zero_steps(self):
        """Avec 0 warmup steps, retourner directement la cible"""
        lr = lr_warmup_linear(5, 0, 0.01)
        assert lr == 0.01


class TestCosineAnnealing:
    """Tests pour lr_cosine_annealing"""
    
    def test_cosine_start(self):
        """Le LR doit commencer au maximum"""
        lr = lr_cosine_annealing(0, 100, 0.0001, 0.01)
        assert lr == pytest.approx(0.01, abs=1e-6)
    
    def test_cosine_end(self):
        """Le LR doit finir au minimum"""
        lr = lr_cosine_annealing(100, 100, 0.0001, 0.01)
        assert lr == pytest.approx(0.0001, abs=1e-6)
    
    def test_cosine_middle(self):
        """Le LR au milieu doit être entre min et max"""
        lr = lr_cosine_annealing(50, 100, 0.0001, 0.01)
        assert 0.0001 < lr < 0.01
    
    def test_cosine_decreasing(self):
        """Le LR doit décroître de manière monotone"""
        lr_0 = lr_cosine_annealing(0, 100, 0.0001, 0.01)
        lr_25 = lr_cosine_annealing(25, 100, 0.0001, 0.01)
        lr_50 = lr_cosine_annealing(50, 100, 0.0001, 0.01)
        lr_75 = lr_cosine_annealing(75, 100, 0.0001, 0.01)
        lr_100 = lr_cosine_annealing(100, 100, 0.0001, 0.01)
        
        assert lr_0 > lr_25 > lr_50 > lr_75 > lr_100
    
    def test_cosine_zero_steps(self):
        """Avec 0 total steps, retourner le max"""
        lr = lr_cosine_annealing(5, 0, 0.0001, 0.01)
        assert lr == 0.01


class TestStepDecay:
    """Tests pour lr_step_decay"""
    
    def test_step_no_decay(self):
        """Avant le premier decay, LR doit être initial"""
        lr = lr_step_decay(0, 0.1, 0.5, 10)
        assert lr == pytest.approx(0.1, abs=1e-6)
        
        lr = lr_step_decay(9, 0.1, 0.5, 10)
        assert lr == pytest.approx(0.1, abs=1e-6)
    
    def test_step_one_decay(self):
        """Après 1 decay, LR = initial * rate"""
        lr = lr_step_decay(10, 0.1, 0.5, 10)
        assert lr == pytest.approx(0.05, abs=1e-6)
        
        lr = lr_step_decay(19, 0.1, 0.5, 10)
        assert lr == pytest.approx(0.05, abs=1e-6)
    
    def test_step_multiple_decays(self):
        """Plusieurs decays successifs"""
        lr_20 = lr_step_decay(20, 0.1, 0.5, 10)
        lr_30 = lr_step_decay(30, 0.1, 0.5, 10)
        lr_40 = lr_step_decay(40, 0.1, 0.5, 10)
        
        assert lr_20 == pytest.approx(0.025, abs=1e-6)
        assert lr_30 == pytest.approx(0.0125, abs=1e-6)
        assert lr_40 == pytest.approx(0.00625, abs=1e-6)
    
    def test_step_plateaus(self):
        """Le LR doit rester constant entre les paliers"""
        lr_10 = lr_step_decay(10, 0.1, 0.5, 10)
        lr_15 = lr_step_decay(15, 0.1, 0.5, 10)
        lr_19 = lr_step_decay(19, 0.1, 0.5, 10)
        
        assert lr_10 == lr_15 == lr_19
    
    def test_step_zero_decay_steps(self):
        """Avec 0 decay steps, retourner initial LR"""
        lr = lr_step_decay(100, 0.1, 0.5, 0)
        assert lr == 0.1


class TestPlateauFactor:
    """Tests pour lr_plateau_factor"""
    
    def test_plateau_improving(self):
        """Avec amélioration, ne pas réduire (factor=1.0)"""
        losses = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5]
        factor = lr_plateau_factor(losses, 3, 0.5, 0.01)
        assert factor == 1.0
    
    def test_plateau_detected(self):
        """Avec plateau, réduire (factor<1.0)"""
        losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
        factor = lr_plateau_factor(losses, 3, 0.5, 0.01)
        assert factor == 0.5
    
    def test_plateau_worsening(self):
        """Avec dégradation, réduire aussi"""
        losses = [0.5, 0.51, 0.52, 0.53, 0.54, 0.55]
        factor = lr_plateau_factor(losses, 3, 0.5, 0.01)
        assert factor == 0.5
    
    def test_plateau_insufficient_history(self):
        """Pas assez d'historique, ne pas réduire"""
        losses = [1.0, 0.9]
        factor = lr_plateau_factor(losses, 3, 0.5, 0.01)
        assert factor == 1.0
    
    def test_plateau_threshold_sensitivity(self):
        """Petit changement en dessous du seuil = plateau"""
        # Amélioration de seulement 0.005 (0.5%) < threshold (1%)
        losses = [1.0, 0.995, 0.99, 0.985, 0.98]
        factor = lr_plateau_factor(losses, 3, 0.5, 0.01)
        assert factor == 0.5
    
    def test_plateau_custom_factor(self):
        """Test avec un facteur différent"""
        losses = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
        factor = lr_plateau_factor(losses, 3, 0.1, 0.01)
        assert factor == 0.1


class TestSchedulingCombinations:
    """Tests pour combinaisons de stratégies"""
    
    def test_warmup_then_cosine(self):
        """Warmup suivi de cosine annealing"""
        warmup_steps = 10
        total_steps = 100
        
        # Pendant warmup
        lr_5 = lr_warmup_linear(5, warmup_steps, 0.01)
        assert 0.0 < lr_5 < 0.01
        
        # Après warmup, cosine annealing
        adjusted_step = 20 - warmup_steps
        adjusted_total = total_steps - warmup_steps
        lr_20 = lr_cosine_annealing(adjusted_step, adjusted_total, 0.0001, 0.01)
        
        assert 0.0001 < lr_20 < 0.01
    
    def test_step_decay_different_rates(self):
        """Step decay avec différents taux"""
        # Decay rapide (0.1)
        lr_fast = lr_step_decay(10, 0.1, 0.1, 10)
        
        # Decay lent (0.9)
        lr_slow = lr_step_decay(10, 0.1, 0.9, 10)
        
        assert lr_fast < lr_slow
    
    def test_scheduling_consistency(self):
        """Les valeurs doivent être cohérentes entre appels"""
        lr1 = lr_warmup_linear(5, 10, 0.01)
        lr2 = lr_warmup_linear(5, 10, 0.01)
        
        assert lr1 == lr2


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
