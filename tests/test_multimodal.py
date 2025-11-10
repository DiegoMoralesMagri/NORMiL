"""
Tests pour les types et primitives multimodales Phase 8.2
===========================================
Auteur : Diego Morales Magri
"""

import pytest
import numpy as np
import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from runtime.normil_types import Vec, ImageTensor, AudioSegment, ModalityFusion
from runtime.primitives import (
    embed_image, embed_audio, temporal_align, cross_attention,
    fusion_concat, fusion_weighted, vision_patch_extract, audio_spectrogram
)


# ============================================
# Tests ImageTensor
# ============================================

class TestImageTensor:
    """Tests pour le type ImageTensor"""
    
    def test_create_image(self):
        """Test création d'image basique"""
        img = ImageTensor.create(224, 224, 3, fill_value=0.5)
        assert img.height == 224
        assert img.width == 224
        assert img.channels == 3
        assert img.data.shape == (224, 224, 3)
        assert np.allclose(img.data, 0.5)
    
    def test_from_array(self):
        """Test création depuis numpy array"""
        data = np.random.rand(100, 100, 3)
        img = ImageTensor.from_array(data)
        assert img.height == 100
        assert img.width == 100
        assert img.channels == 3
        assert img.data.shape == (100, 100, 3)
    
    def test_from_array_grayscale(self):
        """Test création depuis array 2D (grayscale)"""
        data = np.random.rand(50, 50)
        img = ImageTensor.from_array(data)
        assert img.height == 50
        assert img.width == 50
        assert img.channels == 1
        assert img.data.shape == (50, 50, 1)
    
    def test_to_grayscale(self):
        """Test conversion en grayscale"""
        img = ImageTensor.create(100, 100, 3, fill_value=0.5)
        gray = img.to_grayscale()
        assert gray.channels == 1
        assert gray.data.shape == (100, 100, 1)
        assert np.allclose(gray.data, 0.5)
    
    def test_to_grayscale_already_gray(self):
        """Test conversion grayscale quand déjà grayscale"""
        img = ImageTensor.create(50, 50, 1, fill_value=0.3)
        gray = img.to_grayscale()
        assert gray.channels == 1
        assert np.allclose(gray.data, 0.3)
    
    def test_resize(self):
        """Test redimensionnement"""
        img = ImageTensor.create(200, 200, 3, fill_value=0.8)
        resized = img.resize(100, 100)
        assert resized.height == 100
        assert resized.width == 100
        assert resized.channels == 3
        # Valeur approximative (interpolation simple)
        assert np.allclose(resized.data, 0.8, atol=0.1)
    
    def test_resize_upscale(self):
        """Test agrandissement"""
        img = ImageTensor.create(50, 50, 3)
        resized = img.resize(100, 100)
        assert resized.height == 100
        assert resized.width == 100
    
    def test_metadata(self):
        """Test métadonnées"""
        metadata = {"source": "camera", "timestamp": 12345}
        img = ImageTensor.create(64, 64, 3, metadata=metadata)
        assert img.metadata["source"] == "camera"
        assert img.metadata["timestamp"] == 12345


# ============================================
# Tests AudioSegment
# ============================================

class TestAudioSegment:
    """Tests pour le type AudioSegment"""
    
    def test_create_audio(self):
        """Test création audio basique"""
        audio = AudioSegment.create(duration_sec=1.0, sample_rate=16000)
        assert audio.duration == 1.0
        assert audio.sample_rate == 16000
        assert len(audio.samples) == 16000
        assert np.allclose(audio.samples, 0.0)
    
    def test_from_array(self):
        """Test création depuis numpy array"""
        samples = np.random.randn(8000)
        audio = AudioSegment.from_array(samples, sample_rate=8000)
        assert len(audio.samples) == 8000
        assert audio.sample_rate == 8000
        assert audio.duration == 1.0
    
    def test_resample_downsample(self):
        """Test ré-échantillonnage (downsample)"""
        audio = AudioSegment.create(1.0, sample_rate=16000, fill_value=1.0)
        resampled = audio.resample(8000)
        assert resampled.sample_rate == 8000
        assert len(resampled.samples) == 8000
        assert resampled.duration == 1.0
        assert np.allclose(resampled.samples, 1.0, atol=0.1)
    
    def test_resample_upsample(self):
        """Test ré-échantillonnage (upsample)"""
        audio = AudioSegment.create(0.5, sample_rate=8000)
        resampled = audio.resample(16000)
        assert resampled.sample_rate == 16000
        assert len(resampled.samples) == 8000  # duration * new_rate
        assert resampled.duration == 0.5
    
    def test_resample_no_change(self):
        """Test ré-échantillonnage à même rate"""
        audio = AudioSegment.create(1.0, sample_rate=16000)
        resampled = audio.resample(16000)
        assert resampled.sample_rate == 16000
        assert len(resampled.samples) == 16000
    
    def test_trim(self):
        """Test extraction de segment temporel"""
        audio = AudioSegment.create(2.0, sample_rate=16000, fill_value=0.5)
        trimmed = audio.trim(0.5, 1.5)
        assert trimmed.duration == 1.0
        assert len(trimmed.samples) == 16000
        assert np.allclose(trimmed.samples, 0.5)
    
    def test_trim_boundaries(self):
        """Test trim avec bornes extrêmes"""
        audio = AudioSegment.create(1.0, sample_rate=16000)
        trimmed = audio.trim(0.0, 0.5)
        assert trimmed.duration == 0.5
        assert len(trimmed.samples) == 8000
    
    def test_metadata(self):
        """Test métadonnées"""
        metadata = {"source": "microphone", "quality": "high"}
        audio = AudioSegment.create(1.0, sample_rate=44100, metadata=metadata)
        assert audio.metadata["source"] == "microphone"
        assert audio.metadata["quality"] == "high"


# ============================================
# Tests ModalityFusion
# ============================================

class TestModalityFusion:
    """Tests pour le type ModalityFusion"""
    
    def test_create_fusion(self):
        """Test création fusion basique"""
        vec_img = Vec.random(512)
        vec_audio = Vec.random(512)
        
        fusion = ModalityFusion.create(
            modalities={"image": vec_img, "audio": vec_audio}
        )
        
        assert "image" in fusion.modalities
        assert "audio" in fusion.modalities
        assert fusion.alignment_scores["image"] == 1.0
        assert fusion.alignment_scores["audio"] == 1.0
    
    def test_create_with_scores(self):
        """Test création avec scores d'alignement"""
        vec_img = Vec.random(256)
        vec_audio = Vec.random(256)
        
        fusion = ModalityFusion.create(
            modalities={"image": vec_img, "audio": vec_audio},
            alignment_scores={"image": 0.9, "audio": 0.7}
        )
        
        assert fusion.alignment_scores["image"] == 0.9
        assert fusion.alignment_scores["audio"] == 0.7
    
    def test_get_modality(self):
        """Test récupération de modalité"""
        vec_img = Vec.random(128)
        fusion = ModalityFusion.create(modalities={"image": vec_img})
        
        retrieved = fusion.get_modality("image")
        assert retrieved is not None
        assert retrieved.dim == 128
        
        missing = fusion.get_modality("video")
        assert missing is None
    
    def test_add_modality(self):
        """Test ajout de modalité"""
        vec_img = Vec.random(256)
        fusion = ModalityFusion.create(modalities={"image": vec_img})
        
        vec_audio = Vec.random(256)
        fusion.add_modality("audio", vec_audio, alignment_score=0.85)
        
        assert "audio" in fusion.modalities
        assert fusion.alignment_scores["audio"] == 0.85
    
    def test_fused_vec(self):
        """Test vecteur fusionné optionnel"""
        vec_img = Vec.random(512)
        vec_fused = Vec.random(512)
        
        fusion = ModalityFusion.create(
            modalities={"image": vec_img},
            fused_vec=vec_fused
        )
        
        assert fusion.fused_vec is not None
        assert fusion.fused_vec.dim == 512


# ============================================
# Tests Primitives Embedding
# ============================================

class TestEmbeddingPrimitives:
    """Tests pour embed_image et embed_audio"""
    
    def test_embed_image_mobilenet(self):
        """Test embedding image avec MobileNet"""
        img = ImageTensor.create(224, 224, 3, fill_value=0.5)
        vec = embed_image(img, model="mobilenet")
        assert vec.dim == 512
        assert isinstance(vec, Vec)
    
    def test_embed_image_resnet(self):
        """Test embedding image avec ResNet"""
        img = ImageTensor.create(224, 224, 3)
        vec = embed_image(img, model="resnet")
        assert vec.dim == 2048
    
    def test_embed_image_vit(self):
        """Test embedding image avec ViT"""
        img = ImageTensor.create(224, 224, 3)
        vec = embed_image(img, model="vit")
        assert vec.dim == 768
    
    def test_embed_image_different_sizes(self):
        """Test embedding avec différentes tailles d'image"""
        img1 = ImageTensor.create(64, 64, 3)
        img2 = ImageTensor.create(512, 512, 3)
        
        vec1 = embed_image(img1)
        vec2 = embed_image(img2)
        
        # Même dimension de sortie
        assert vec1.dim == vec2.dim == 512
    
    def test_embed_audio_wavenet(self):
        """Test embedding audio avec Wavenet"""
        audio = AudioSegment.create(1.0, sample_rate=16000)
        vec = embed_audio(audio, model="wavenet")
        assert vec.dim == 512
        assert isinstance(vec, Vec)
    
    def test_embed_audio_wav2vec(self):
        """Test embedding audio avec Wav2Vec"""
        audio = AudioSegment.create(1.0, sample_rate=16000)
        vec = embed_audio(audio, model="wav2vec")
        assert vec.dim == 768
    
    def test_embed_audio_hubert(self):
        """Test embedding audio avec HuBERT"""
        audio = AudioSegment.create(1.0, sample_rate=16000)
        vec = embed_audio(audio, model="hubert")
        assert vec.dim == 1024
    
    def test_embed_audio_different_durations(self):
        """Test embedding avec différentes durées"""
        audio1 = AudioSegment.create(0.5, sample_rate=16000)
        audio2 = AudioSegment.create(2.0, sample_rate=16000)
        
        vec1 = embed_audio(audio1)
        vec2 = embed_audio(audio2)
        
        # Même dimension de sortie
        assert vec1.dim == vec2.dim == 512


# ============================================
# Tests Primitives Fusion & Alignment
# ============================================

class TestFusionPrimitives:
    """Tests pour temporal_align, cross_attention, fusion_*"""
    
    def test_temporal_align(self):
        """Test alignement temporel basique"""
        vec_img = Vec.random(512)
        vec_audio = Vec.random(512)
        
        fusion = temporal_align(
            {"image": vec_img, "audio": vec_audio},
            window_ms=500
        )
        
        assert isinstance(fusion, ModalityFusion)
        assert "image" in fusion.modalities
        assert "audio" in fusion.modalities
        assert "image" in fusion.alignment_scores
        assert "audio" in fusion.alignment_scores
    
    def test_temporal_align_scores(self):
        """Test que les scores d'alignement sont dans [0, 1]"""
        vec1 = Vec.random(256)
        vec2 = Vec.random(256)
        
        fusion = temporal_align({"v1": vec1, "v2": vec2})
        
        for score in fusion.alignment_scores.values():
            assert 0.0 <= score <= 1.0
    
    def test_cross_attention(self):
        """Test cross-attention basique"""
        query = Vec.random(512)
        key = Vec.random(512)
        value = Vec.random(512)
        
        result = cross_attention(query, key, value, num_heads=8)
        
        assert isinstance(result, Vec)
        assert result.dim == 512
    
    def test_cross_attention_different_heads(self):
        """Test cross-attention avec différents nombres de têtes"""
        query = Vec.random(256)
        key = Vec.random(256)
        value = Vec.random(256)
        
        result4 = cross_attention(query, key, value, num_heads=4)
        result8 = cross_attention(query, key, value, num_heads=8)
        
        assert result4.dim == result8.dim == 256
    
    def test_fusion_concat(self):
        """Test fusion par concaténation"""
        vec1 = Vec.random(256)
        vec2 = Vec.random(512)
        vec3 = Vec.random(128)
        
        fused = fusion_concat([vec1, vec2, vec3])
        
        assert fused.dim == 256 + 512 + 128
    
    def test_fusion_concat_empty(self):
        """Test fusion concat avec liste vide"""
        fused = fusion_concat([])
        assert fused.dim == 0
    
    def test_fusion_weighted(self):
        """Test fusion pondérée"""
        vec1 = Vec.random(512)
        vec2 = Vec.random(512)
        weights = Vec.from_list([0.7, 0.3])
        
        fused = fusion_weighted([vec1, vec2], weights)
        
        assert isinstance(fused, Vec)
        assert fused.dim == 512
    
    def test_fusion_weighted_equal_weights(self):
        """Test fusion avec poids égaux"""
        vec1 = Vec.from_list([1.0, 2.0, 3.0])
        vec2 = Vec.from_list([4.0, 5.0, 6.0])
        weights = Vec.from_list([0.5, 0.5])
        
        fused = fusion_weighted([vec1, vec2], weights)
        
        # Devrait être la moyenne
        expected = Vec.from_list([2.5, 3.5, 4.5])
        assert np.allclose(fused.data, expected.data, atol=0.01)
    
    def test_fusion_weighted_dimension_mismatch(self):
        """Test erreur si dimensions incompatibles"""
        vec1 = Vec.random(256)
        vec2 = Vec.random(512)  # Dimension différente
        weights = Vec.from_list([0.5, 0.5])
        
        with pytest.raises(ValueError):
            fusion_weighted([vec1, vec2], weights)
    
    def test_fusion_weighted_wrong_num_weights(self):
        """Test erreur si nombre de poids incorrect"""
        vec1 = Vec.random(128)
        vec2 = Vec.random(128)
        weights = Vec.from_list([0.5])  # Seulement 1 poids pour 2 vecs
        
        with pytest.raises(ValueError):
            fusion_weighted([vec1, vec2], weights)


# ============================================
# Tests Primitives Vision & Audio
# ============================================

class TestVisionAudioPrimitives:
    """Tests pour vision_patch_extract et audio_spectrogram"""
    
    def test_vision_patch_extract(self):
        """Test extraction de patches basique"""
        img = ImageTensor.create(224, 224, 3)
        patches = vision_patch_extract(img, patch_size=16)
        
        # 224 / 16 = 14 patches par dimension
        assert len(patches) == 14 * 14
        
        # Chaque patch: 16 * 16 * 3 = 768
        assert patches[0].dim == 768
    
    def test_vision_patch_extract_different_sizes(self):
        """Test extraction avec différentes tailles de patches"""
        img = ImageTensor.create(256, 256, 3)
        
        patches32 = vision_patch_extract(img, patch_size=32)
        assert len(patches32) == 8 * 8  # 256/32 = 8
        
        patches64 = vision_patch_extract(img, patch_size=64)
        assert len(patches64) == 4 * 4  # 256/64 = 4
    
    def test_vision_patch_extract_grayscale(self):
        """Test extraction patches sur image grayscale"""
        img = ImageTensor.create(128, 128, 1)
        patches = vision_patch_extract(img, patch_size=16)
        
        assert len(patches) == 8 * 8
        assert patches[0].dim == 16 * 16 * 1  # 256
    
    def test_audio_spectrogram(self):
        """Test calcul de spectrogramme basique"""
        audio = AudioSegment.create(1.0, sample_rate=16000)
        spec = audio_spectrogram(audio, n_fft=512, hop_length=256)
        
        assert isinstance(spec, ImageTensor)
        assert spec.height == 256  # n_fft / 2
        assert spec.channels == 1  # Grayscale
    
    def test_audio_spectrogram_dimensions(self):
        """Test dimensions du spectrogramme"""
        audio = AudioSegment.create(2.0, sample_rate=16000)
        spec = audio_spectrogram(audio, n_fft=1024, hop_length=512)
        
        assert spec.height == 512  # n_fft / 2
        # Width dépend de la durée et hop_length
        expected_width = (len(audio.samples) - 1024) // 512 + 1
        assert spec.width == expected_width
    
    def test_audio_spectrogram_metadata(self):
        """Test métadonnées du spectrogramme"""
        audio = AudioSegment.create(1.0, sample_rate=22050)
        spec = audio_spectrogram(audio, n_fft=512, hop_length=256)
        
        assert spec.metadata["type"] == "spectrogram"
        assert spec.metadata["n_fft"] == 512
        assert spec.metadata["hop_length"] == 256
        assert spec.metadata["sample_rate"] == 22050


# ============================================
# Tests d'intégration
# ============================================

class TestMultimodalIntegration:
    """Tests d'intégration pour workflows multimodaux complets"""
    
    def test_full_image_pipeline(self):
        """Test pipeline complet: image -> patches -> embedding"""
        # 1. Créer image
        img = ImageTensor.create(224, 224, 3, fill_value=0.5)
        
        # 2. Extraire patches
        patches = vision_patch_extract(img, patch_size=16)
        assert len(patches) == 196
        
        # 3. Embedding
        emb = embed_image(img, model="mobilenet")
        assert emb.dim == 512
    
    def test_full_audio_pipeline(self):
        """Test pipeline complet: audio -> spectrogram -> embedding"""
        # 1. Créer audio
        audio = AudioSegment.create(1.0, sample_rate=16000)
        
        # 2. Spectrogramme
        spec = audio_spectrogram(audio, n_fft=512, hop_length=256)
        assert isinstance(spec, ImageTensor)
        
        # 3. Embedding
        emb = embed_audio(audio, model="wavenet")
        assert emb.dim == 512
    
    def test_multimodal_fusion_pipeline(self):
        """Test pipeline fusion multimodale complète"""
        # 1. Créer modalités
        img = ImageTensor.create(224, 224, 3)
        audio = AudioSegment.create(1.0, sample_rate=16000)
        
        # 2. Embeddings
        img_emb = embed_image(img)
        audio_emb = embed_audio(audio)
        
        # 3. Alignement temporel
        fusion = temporal_align({
            "image": img_emb,
            "audio": audio_emb
        }, window_ms=500)
        
        assert len(fusion.modalities) == 2
        
        # 4. Cross-attention
        attended = cross_attention(img_emb, audio_emb, audio_emb)
        assert attended.dim == 512
        
        # 5. Fusion finale
        fused = fusion_concat([img_emb, audio_emb])
        assert fused.dim == 1024
    
    def test_weighted_multimodal_fusion(self):
        """Test fusion pondérée de 3 modalités"""
        # Simuler 3 modalités
        img_emb = embed_image(ImageTensor.create(224, 224, 3))
        audio_emb = embed_audio(AudioSegment.create(1.0))
        text_emb = Vec.random(512)  # Simuler embedding texte
        
        # Fusion pondérée
        weights = Vec.from_list([0.5, 0.3, 0.2])  # Image dominant
        fused = fusion_weighted([img_emb, audio_emb, text_emb], weights)
        
        assert fused.dim == 512
        
        # Vérifier que fusion est différente de moyenne simple
        avg = fusion_weighted([img_emb, audio_emb, text_emb], 
                             Vec.from_list([1/3, 1/3, 1/3]))
        
        # Devrait être différent (sauf cas extrême)
        assert not np.allclose(fused.data, avg.data, atol=1e-6)
