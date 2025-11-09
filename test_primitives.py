"""
Script de test pour les primitives NORMiL
"""

import sys
sys.path.insert(0, '.')

from runtime.normil_types import Vec, EpisodicRecord, Concept
from runtime.primitives import *

print("=== Test NORMiL Primitives ===\n")

# Test vectorielles
print("1. Primitives Vectorielles:")
v1 = zeros(128)
v2 = random(128, mean=0.0, std=1.0)
v3 = vec_add(v1, v2)
print(f"   zeros(128): {v1}")
print(f"   random(128): {v2}")
print(f"   vec_add(v1, v2): {v3}")
print(f"   dot(v2, v2): {dot(v2, v2):.3f}")
print(f"   norm(v2): {norm(v2):.3f}\n")

# Test episodic
print("2. Primitives Episodic:")
vec = random(128)
record = EpisodicRecord.create("Test memory", vec, trust=0.9)
id1 = episodic_append(record)
print(f"   Appended: {id1}")

results = episodic_query(vec, k=5)
print(f"   Query results: {len(results)} records")
print(f"   First result: {results[0] if results else 'None'}\n")

# Test semantic
print("3. Primitives Semantic:")
centroid = random(128)
concept = Concept.create(centroid, labels=["test"], trust=0.85)
semantic_upsert(concept)
print(f"   Upserted concept: {concept.concept_id[:8]}...")

concepts = semantic_query(centroid, k=5)
print(f"   Query results: {len(concepts)} concepts\n")

# Test consolidation
print("4. Consolidation:")
episodes = [
    EpisodicRecord.create(f"Memory {i}", random(128), trust=0.8 + i*0.05)
    for i in range(3)
]
for ep in episodes:
    episodic_append(ep)

consolidated = consolidate(episodes, method="cluster")
print(f"   Consolidated {len(episodes)} episodes into {len(consolidated)} concepts")
if consolidated:
    print(f"   Concept: {consolidated[0]}\n")

# Test audit
print("5. Audit:")
audit_log("test_action", {"key": "value"}, level="info")
snapshot_hash = audit_snapshot("test_snapshot")
print(f"   Snapshot hash: {snapshot_hash[:16]}...\n")

print()
print("=== Primitives tests passed! ===")
