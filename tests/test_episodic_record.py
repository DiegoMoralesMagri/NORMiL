"""
Tests pour Phase 5.1 - Type EpisodicRecord

Auteur : Diego Morales Magri

EpisodicRecord est un type spécialisé pour la mémoire épisodique de O-RedMind.
Structure: {
    id: str,
    timestamp: float,
    sources: [str],
    vecs: {str: Vec},
    summary: str,
    labels: [{label: str, score: float}],
    trust: float,
    provenance: {device_id: str, signature: str},
    outcome: str
}
"""

import sys
import os
from pathlib import Path

# Ajouter le dossier parent au path
NORMIL_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(NORMIL_ROOT))

from parser.lexer import Lexer
from parser.parser import Parser
from runtime.executor import Executor
import unittest


class TestEpisodicRecord(unittest.TestCase):
    """Tests pour le type EpisodicRecord"""
    
    def setUp(self):
        self.executor = Executor()
    
    def run_code(self, code: str):
        """Helper pour exécuter du code NORMiL"""
        lexer = Lexer(code)
        tokens = lexer.tokenize()
        parser = Parser(tokens)
        ast = parser.parse()
        self.executor.execute(ast)
    
    def test_empty_episodic_record(self):
        """Test 1: Créer un EpisodicRecord vide"""
        code = """
        let record = EpisodicRecord {
            id: "ep001",
            timestamp: 1234567890.0,
            sources: [],
            vecs: {},
            summary: "",
            labels: [],
            trust: 0.0,
            provenance: {},
            outcome: ""
        }
        """
        self.run_code(code)
        
        record = self.executor.current_scope.get_var("record")
        self.assertIsNotNone(record)
        self.assertEqual(record.id, "ep001")
        self.assertEqual(record.timestamp, 1234567890.0)
    
    def test_episodic_record_with_data(self):
        """Test 2: EpisodicRecord avec données"""
        code = """
        let record = EpisodicRecord {
            id: "ep002",
            timestamp: 1698000000.0,
            sources: ["camera", "mic"],
            vecs: {},
            summary: "Test event",
            labels: [],
            trust: 0.95,
            provenance: {},
            outcome: "success"
        }
        """
        self.run_code(code)
        
        record = self.executor.current_scope.get_var("record")
        self.assertEqual(len(record.sources), 2)
        self.assertEqual(record.sources[0], "camera")
        self.assertEqual(record.trust, 0.95)
        self.assertEqual(record.outcome, "success")
    
    def test_episodic_record_with_labels(self):
        """Test 3: EpisodicRecord avec labels"""
        code = """
        let record = EpisodicRecord {
            id: "ep003",
            timestamp: 1698000001.0,
            sources: ["text"],
            vecs: {},
            summary: "Conversation",
            labels: [
                {label: "greeting", score: 0.9},
                {label: "question", score: 0.7}
            ],
            trust: 0.85,
            provenance: {},
            outcome: ""
        }
        """
        self.run_code(code)
        
        record = self.executor.current_scope.get_var("record")
        self.assertEqual(len(record.labels), 2)
        self.assertEqual(record.labels[0]["label"], "greeting")
        self.assertEqual(record.labels[0]["score"], 0.9)
    
    def test_episodic_record_field_access(self):
        """Test 4: Accès aux champs d'un EpisodicRecord"""
        code = """
        let record = EpisodicRecord {
            id: "ep004",
            timestamp: 1698000002.0,
            sources: ["sensor"],
            vecs: {},
            summary: "Temperature reading",
            labels: [],
            trust: 1.0,
            provenance: {device_id: "sensor01", signature: "abc123"},
            outcome: "recorded"
        }
        
        let event_id = record.id
        let when = record.timestamp
        let how_trusted = record.trust
        """
        self.run_code(code)
        
        event_id = self.executor.current_scope.get_var("event_id")
        when = self.executor.current_scope.get_var("when")
        how_trusted = self.executor.current_scope.get_var("how_trusted")
        
        self.assertEqual(event_id, "ep004")
        self.assertEqual(when, 1698000002.0)
        self.assertEqual(how_trusted, 1.0)
    
    def test_episodic_record_with_provenance(self):
        """Test 5: EpisodicRecord avec provenance complète"""
        code = """
        let record = EpisodicRecord {
            id: "ep005",
            timestamp: 1698000003.0,
            sources: ["api"],
            vecs: {},
            summary: "API call",
            labels: [{label: "api_request", score: 1.0}],
            trust: 0.8,
            provenance: {
                device_id: "server01",
                signature: "sig_xyz789"
            },
            outcome: "completed"
        }
        
        let device = record.provenance.device_id
        """
        self.run_code(code)
        
        device = self.executor.current_scope.get_var("device")
        self.assertEqual(device, "server01")
    
    def test_episodic_record_update_field(self):
        """Test 6: Modifier un champ d'EpisodicRecord"""
        code = """
        let record = EpisodicRecord {
            id: "ep006",
            timestamp: 1698000004.0,
            sources: [],
            vecs: {},
            summary: "Initial",
            labels: [],
            trust: 0.5,
            provenance: {},
            outcome: "pending"
        }
        
        record.outcome = "completed"
        record.trust = 0.9
        """
        self.run_code(code)
        
        record = self.executor.current_scope.get_var("record")
        self.assertEqual(record.outcome, "completed")
        self.assertEqual(record.trust, 0.9)
    
    def test_episodic_record_in_function(self):
        """Test 7: Passer EpisodicRecord à une fonction"""
        code = """
        fn get_event_id(record: EpisodicRecord) -> str {
            return record.id
        }
        
        let record = EpisodicRecord {
            id: "ep007",
            timestamp: 1698000005.0,
            sources: [],
            vecs: {},
            summary: "",
            labels: [],
            trust: 0.0,
            provenance: {},
            outcome: ""
        }
        
        let id_result = get_event_id(record)
        """
        self.run_code(code)
        
        id_result = self.executor.current_scope.get_var("id_result")
        self.assertEqual(id_result, "ep007")
    
    def test_episodic_record_list(self):
        """Test 8: Liste d'EpisodicRecords"""
        code = """
        let records = [
            EpisodicRecord {
                id: "ep008a",
                timestamp: 1698000006.0,
                sources: [],
                vecs: {},
                summary: "Event A",
                labels: [],
                trust: 0.8,
                provenance: {},
                outcome: ""
            },
            EpisodicRecord {
                id: "ep008b",
                timestamp: 1698000007.0,
                sources: [],
                vecs: {},
                summary: "Event B",
                labels: [],
                trust: 0.9,
                provenance: {},
                outcome: ""
            }
        ]
        
        let first_id = records[0].id
        let second_summary = records[1].summary
        """
        self.run_code(code)
        
        first_id = self.executor.current_scope.get_var("first_id")
        second_summary = self.executor.current_scope.get_var("second_summary")
        
        self.assertEqual(first_id, "ep008a")
        self.assertEqual(second_summary, "Event B")
    
    def test_episodic_record_with_vectors(self):
        """Test 9: EpisodicRecord avec vecteurs"""
        code = """
        let v1 = zeros(dim: 64)
        let v2 = ones(dim: 64)
        
        let record = EpisodicRecord {
            id: "ep009",
            timestamp: 1698000008.0,
            sources: ["camera", "audio"],
            vecs: {
                img: v1,
                audio: v2
            },
            summary: "Multimodal event",
            labels: [],
            trust: 0.95,
            provenance: {},
            outcome: ""
        }
        
        let img_vec = record.vecs.img
        let audio_vec = record.vecs.audio
        """
        self.run_code(code)
        
        img_vec = self.executor.current_scope.get_var("img_vec")
        audio_vec = self.executor.current_scope.get_var("audio_vec")
        
        self.assertIsNotNone(img_vec)
        self.assertIsNotNone(audio_vec)
        self.assertEqual(len(img_vec), 64)
        self.assertEqual(len(audio_vec), 64)
    
    def test_episodic_record_validation(self):
        """Test 10: Validation des champs requis"""
        code = """
        fn is_valid_record(record: EpisodicRecord) -> bool {
            if record.id == "" {
                return false
            }
            if record.timestamp <= 0.0 {
                return false
            }
            return true
        }
        
        let good_record = EpisodicRecord {
            id: "ep010",
            timestamp: 1698000009.0,
            sources: [],
            vecs: {},
            summary: "",
            labels: [],
            trust: 0.0,
            provenance: {},
            outcome: ""
        }
        
        let bad_record = EpisodicRecord {
            id: "",
            timestamp: -1.0,
            sources: [],
            vecs: {},
            summary: "",
            labels: [],
            trust: 0.0,
            provenance: {},
            outcome: ""
        }
        
        let is_good = is_valid_record(good_record)
        let is_bad = is_valid_record(bad_record)
        """
        self.run_code(code)
        
        is_good = self.executor.current_scope.get_var("is_good")
        is_bad = self.executor.current_scope.get_var("is_bad")
        
        self.assertTrue(is_good)
        self.assertFalse(is_bad)


if __name__ == '__main__':
    unittest.main()
