"""
Tests pour les types O-RedMind Phase 8
=======================================

Auteur : Diego Morales Magri

Tests pour InstinctPackage, SafetyGuardrail, IndexEntry, AuditLogEntry
"""

import pytest
import sys
from pathlib import Path

# Ajouter le répertoire runtime au path
runtime_path = Path(__file__).parent.parent / "runtime"
sys.path.insert(0, str(runtime_path))

from normil_types import (
    Vec, InstinctPackage, SafetyGuardrail, IndexEntry, AuditLogEntry,
    MetaParams, ValidationManifest, InstinctCore, InstinctOverlay,
    ProtoInstinct, Rule, ConsentRequest, now
)


# ============================================
# Tests InstinctPackage
# ============================================

class TestInstinctPackage:
    """Tests pour InstinctPackage et types associés"""
    
    def test_meta_params_creation(self):
        """Test création MetaParams"""
        meta = MetaParams(
            attention_weights={"visual": 0.6, "audio": 0.4},
            base_plastic_rate=0.001,
            safety_threshold=0.95
        )
        
        assert meta.base_plastic_rate == 0.001
        assert meta.safety_threshold == 0.95
        assert meta.attention_weights["visual"] == 0.6
    
    def test_validation_manifest_creation(self):
        """Test création ValidationManifest"""
        manifest = ValidationManifest(
            tests_passed=["test_1", "test_2", "test_3"],
            metrics_before={"accuracy": 0.80},
            metrics_after={"accuracy": 0.90},
            validators=["validator_1", "validator_2"],
            timestamp=now()
        )
        
        assert len(manifest.tests_passed) == 3
        assert manifest.metrics_after["accuracy"] > manifest.metrics_before["accuracy"]
        assert len(manifest.validators) == 2
    
    def test_instinct_core_creation(self):
        """Test création InstinctCore"""
        vec = Vec.random(128)
        proto = ProtoInstinct.create("test_proto", vec, weight=1.5)
        rule = Rule("test_rule", "similarity > 0.8", "boost_attention", 100)
        meta = MetaParams({"visual": 0.6}, 0.001, 0.95)
        
        core = InstinctCore(
            prototypes=[proto],
            rules=[rule],
            meta_params=meta
        )
        
        assert len(core.prototypes) == 1
        assert len(core.rules) == 1
        assert core.meta_params.base_plastic_rate == 0.001
    
    def test_instinct_overlay_creation(self):
        """Test création InstinctOverlay"""
        overlay = InstinctOverlay(
            prototypes=[],
            rules=[],
            provenance="test_validator",
            validation_signature="signature_abc123"
        )
        
        assert overlay.provenance == "test_validator"
        assert overlay.validation_signature == "signature_abc123"
    
    def test_instinct_package_creation(self):
        """Test création InstinctPackage complet"""
        vec = Vec.random(128)
        proto = ProtoInstinct.create("safety", vec, weight=1.5)
        meta = MetaParams({"visual": 0.6}, 0.001, 0.95)
        
        core = InstinctCore([proto], [], meta)
        overlay = InstinctOverlay([], [], "validator", "sig123")
        manifest = ValidationManifest(["test_1"], {}, {}, ["v1"], now())
        
        package = InstinctPackage.create(
            package_id="instinct_v1",
            version="1.0.0",
            core=core,
            overlay=overlay,
            manifest=manifest
        )
        
        assert package.package_id == "instinct_v1"
        assert package.version == "1.0.0"
        assert len(package.signature) == 64  # SHA256 hash
        assert package.timestamp > 0


# ============================================
# Tests SafetyGuardrail
# ============================================

class TestSafetyGuardrail:
    """Tests pour SafetyGuardrail"""
    
    def test_guardrail_creation(self):
        """Test création SafetyGuardrail"""
        guardrail = SafetyGuardrail.create(
            id="no_delete",
            condition="action == 'delete'",
            action_blocked="delete",
            require_consent=True,
            override_level=10,
            description="Empêche suppressions sans consentement"
        )
        
        assert guardrail.id == "no_delete"
        assert guardrail.require_consent is True
        assert guardrail.override_level == 10
    
    def test_guardrail_without_consent(self):
        """Test guardrail sans consentement requis"""
        guardrail = SafetyGuardrail.create(
            id="rate_limit",
            condition="requests_per_sec > 100",
            action_blocked="request",
            require_consent=False,
            override_level=5
        )
        
        assert guardrail.require_consent is False
        assert guardrail.override_level == 5
    
    def test_consent_request_creation(self):
        """Test création ConsentRequest"""
        request = ConsentRequest(
            action="delete_file",
            reason="User requested deletion",
            data_accessed=["file1.txt", "file2.txt"],
            expiry_ttl=3600000  # 1 heure
        )
        
        assert request.action == "delete_file"
        assert len(request.data_accessed) == 2
        assert request.expiry_ttl == 3600000


# ============================================
# Tests AuditLogEntry
# ============================================

class TestAuditLogEntry:
    """Tests pour AuditLogEntry et hash chaining"""
    
    def test_audit_entry_creation(self):
        """Test création AuditLogEntry"""
        entry = AuditLogEntry.create(
            event_type="memory_append",
            actor="system",
            action="append_episodic",
            data={"record_id": "12345"}
        )
        
        assert entry.event_type == "memory_append"
        assert entry.actor == "system"
        assert len(entry.signature) == 64
        assert len(entry.data_hash) == 64
    
    def test_audit_hash_chaining(self):
        """Test hash chaining entre entrées"""
        entry1 = AuditLogEntry.create(
            event_type="event_1",
            actor="user",
            action="action_1",
            data={"key": "value1"}
        )
        
        hash1 = entry1.compute_hash()
        
        entry2 = AuditLogEntry.create(
            event_type="event_2",
            actor="system",
            action="action_2",
            data={"key": "value2"},
            prev_hash=hash1
        )
        
        # Vérifier que entry2 référence bien entry1
        assert entry2.prev_hash == hash1
        assert entry2.prev_hash != "0" * 64
    
    def test_audit_chain_integrity(self):
        """Test intégrité d'une chaîne d'audit"""
        entries = []
        prev_hash = "0" * 64
        
        for i in range(5):
            entry = AuditLogEntry.create(
                event_type=f"event_{i}",
                actor="system",
                action=f"action_{i}",
                data={"index": i},
                prev_hash=prev_hash
            )
            entries.append(entry)
            prev_hash = entry.compute_hash()
        
        # Vérifier la chaîne
        for i in range(1, len(entries)):
            expected_hash = entries[i-1].compute_hash()
            assert entries[i].prev_hash == expected_hash


# ============================================
# Tests IndexEntry
# ============================================

class TestIndexEntry:
    """Tests pour IndexEntry et HNSW"""
    
    def test_index_entry_creation(self):
        """Test création IndexEntry"""
        vec = Vec.random(128)
        entry = IndexEntry.create(
            vec=vec,
            metadata={"type": "image", "timestamp": "2025-11-01"},
            layer=0
        )
        
        assert entry.vec.dim == 128
        assert entry.layer == 0
        assert entry.metadata["type"] == "image"
        assert len(entry.neighbors) == 0
    
    def test_add_neighbor(self):
        """Test ajout de voisins"""
        vec = Vec.random(128)
        entry = IndexEntry.create(vec, {}, layer=0)
        
        entry.add_neighbor("neighbor_1", distance=0.15)
        entry.add_neighbor("neighbor_2", distance=0.23)
        
        assert len(entry.neighbors) == 2
        assert "neighbor_1" in entry.neighbors
        assert entry.get_distance("neighbor_1") == 0.15
    
    def test_distance_cache(self):
        """Test cache de distances"""
        vec = Vec.random(128)
        entry = IndexEntry.create(vec, {}, layer=0)
        
        entry.add_neighbor("n1", 0.1)
        entry.add_neighbor("n2", 0.2)
        entry.add_neighbor("n3", 0.3)
        
        assert entry.get_distance("n1") == 0.1
        assert entry.get_distance("n2") == 0.2
        assert entry.get_distance("n3") == 0.3
        assert entry.get_distance("nonexistent") is None
    
    def test_multiple_layers(self):
        """Test entrées sur différentes couches HNSW"""
        vec = Vec.random(128)
        
        entry_layer0 = IndexEntry.create(vec, {"layer": "0"}, layer=0)
        entry_layer1 = IndexEntry.create(vec, {"layer": "1"}, layer=1)
        entry_layer2 = IndexEntry.create(vec, {"layer": "2"}, layer=2)
        
        assert entry_layer0.layer == 0
        assert entry_layer1.layer == 1
        assert entry_layer2.layer == 2


# ============================================
# Tests d'intégration
# ============================================

class TestORedMindIntegration:
    """Tests d'intégration des types O-RedMind"""
    
    def test_full_instinct_workflow(self):
        """Test workflow complet de création d'un package d'instinct"""
        # 1. Créer des prototypes
        vec1 = Vec.random(128)
        vec2 = Vec.random(128)
        proto1 = ProtoInstinct.create("safety", vec1, weight=1.5)
        proto2 = ProtoInstinct.create("attention", vec2, weight=1.0)
        
        # 2. Créer des règles
        rule1 = Rule("safety_check", "similarity > 0.9", "block_action", priority=100)
        rule2 = Rule("attention_boost", "novelty > 0.8", "increase_lr", priority=50)
        
        # 3. Créer meta params
        meta = MetaParams(
            attention_weights={"visual": 0.6, "audio": 0.3, "text": 0.1},
            base_plastic_rate=0.001,
            safety_threshold=0.95
        )
        
        # 4. Assembler le core
        core = InstinctCore(
            prototypes=[proto1, proto2],
            rules=[rule1, rule2],
            meta_params=meta
        )
        
        # 5. Créer overlay (vide pour commencer)
        overlay = InstinctOverlay([], [], "test_validator", "sig_abc123")
        
        # 6. Créer manifest de validation
        manifest = ValidationManifest(
            tests_passed=["test_safety", "test_performance", "test_ethics"],
            metrics_before={"accuracy": 0.85, "safety_score": 0.90},
            metrics_after={"accuracy": 0.88, "safety_score": 0.95},
            validators=["validator_1", "validator_2", "validator_3"],
            timestamp=now()
        )
        
        # 7. Créer package final
        package = InstinctPackage.create(
            package_id="safety_instinct_v2",
            version="2.1.0",
            core=core,
            overlay=overlay,
            manifest=manifest
        )
        
        # Vérifications
        assert package.package_id == "safety_instinct_v2"
        assert len(package.core.prototypes) == 2
        assert len(package.core.rules) == 2
        assert len(package.validation_manifest.tests_passed) == 3
    
    def test_audit_and_safety_integration(self):
        """Test intégration audit + safety"""
        # 1. Créer guardrail
        guardrail = SafetyGuardrail.create(
            id="no_io_without_consent",
            condition="action.type in ['file_write', 'network_send']",
            action_blocked="*",
            require_consent=True,
            override_level=10
        )
        
        # 2. Créer audit log pour tentative d'action
        entry1 = AuditLogEntry.create(
            event_type="guardrail_check",
            actor="system",
            action="check_io_action",
            data={"guardrail_id": guardrail.id, "action": "file_write"}
        )
        
        # 3. Créer consent request
        consent = ConsentRequest(
            action="file_write",
            reason="User requested save",
            data_accessed=["user_data.json"],
            expiry_ttl=3600000
        )
        
        # 4. Log consent decision
        entry2 = AuditLogEntry.create(
            event_type="consent_granted",
            actor="user",
            action="grant_consent",
            data={"consent": consent.action},
            prev_hash=entry1.compute_hash()
        )
        
        # 5. Log action executed
        entry3 = AuditLogEntry.create(
            event_type="action_executed",
            actor="system",
            action="file_write",
            data={"file": "user_data.json"},
            prev_hash=entry2.compute_hash()
        )
        
        # Vérifier la chaîne
        assert entry2.prev_hash == entry1.compute_hash()
        assert entry3.prev_hash == entry2.compute_hash()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
