"""
Script de profiling pour NORMiL
Identifie les bottlenecks de performance avec cProfile
"""
import cProfile
import pstats
import io
from pathlib import Path
import sys

# Ajouter le chemin du module
sys.path.insert(0, str(Path(__file__).parent))

from normil_cli import main as normil_main
from parser.parser import Parser
from parser.lexer import Lexer
from runtime.executor import Executor


def profile_parsing(code: str):
    """Profile le parsing d'un fichier"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    return ast


def profile_execution(code: str):
    """Profile l'exécution d'un fichier"""
    lexer = Lexer(code)
    tokens = lexer.tokenize()
    parser = Parser(tokens)
    ast = parser.parse()
    executor = Executor()
    for stmt in ast:
        executor.exec_statement(stmt)


def profile_full_pipeline(filepath: str):
    """Profile le pipeline complet (parsing + exécution)"""
    with open(filepath, 'r', encoding='utf-8') as f:
        code = f.read()
    
    profile_execution(code)


def run_profiling():
    """Exécute le profiling sur différents scénarios"""
    
    print("=" * 80)
    print("PROFILING NORMIL - Identification des bottlenecks")
    print("=" * 80)
    print()
    
    # Test files à profiler
    test_files = [
        "examples/test_plasticity_primitives.nor",
        "examples/test_advanced_plasticity.nor",
        "examples/test_custom_plasticity_modes.nor",
        "examples/test_lr_scheduling.nor",
        "examples/test_multi_criteria_stability.nor",
    ]
    
    for test_file in test_files:
        filepath = Path(test_file)
        if not filepath.exists():
            print(f"⚠️  Fichier non trouvé: {test_file}")
            continue
        
        print(f"\n{'='*80}")
        print(f"Profiling: {test_file}")
        print(f"{'='*80}\n")
        
        # Créer un profiler
        profiler = cProfile.Profile()
        
        # Profiler l'exécution
        profiler.enable()
        try:
            profile_full_pipeline(str(filepath))
        except Exception as e:
            print(f"❌ Erreur durant profiling: {e}")
            profiler.disable()
            continue
        profiler.disable()
        
        # Analyser les résultats
        s = io.StringIO()
        ps = pstats.Stats(profiler, stream=s)
        ps.strip_dirs()
        
        # Top 20 fonctions les plus coûteuses en temps cumulé
        print("\n🔥 Top 20 fonctions (temps cumulé):")
        print("-" * 80)
        ps.sort_stats('cumulative')
        ps.print_stats(20)
        
        # Top 20 fonctions les plus appelées
        print("\n📊 Top 20 fonctions (nombre d'appels):")
        print("-" * 80)
        ps.sort_stats('calls')
        ps.print_stats(20)
        
        # Top 20 fonctions les plus coûteuses en temps propre
        print("\n⏱️  Top 20 fonctions (temps propre):")
        print("-" * 80)
        ps.sort_stats('time')
        ps.print_stats(20)
        
        # Sauvegarder les stats complètes
        stats_file = f"profiling_{filepath.stem}.stats"
        ps.dump_stats(stats_file)
        print(f"\n✅ Stats complètes sauvegardées: {stats_file}")


def profile_specific_operations():
    """Profile des opérations spécifiques"""
    
    print("\n" + "=" * 80)
    print("PROFILING - Opérations Spécifiques")
    print("=" * 80)
    
    # Test 1: Opérations vectorielles
    print("\n🧮 Test 1: Opérations vectorielles (1000 itérations)")
    code_vec = """
fn test_vec_ops() {
    let v1 = random(128, 0.0, 1.0)
    let v2 = random(128, 0.0, 1.0)
    
    let iter = 0
    while iter < 1000 {
        let v3 = v1 + v2
        let v4 = v3 - v1
        let v5 = scale(v4, 0.5)
        let n = norm(v5)
        iter = iter + 1
    }
}

test_vec_ops()
"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    profile_execution(code_vec)
    profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs().sort_stats('cumulative').print_stats(15)
    print(s.getvalue())
    
    # Test 2: Plasticité
    print("\n🧠 Test 2: Plasticité avec @plastic (100 itérations)")
    code_plastic = """
@plastic(rate: 0.01, mode: "hebbian")
fn plastic_learn(input: Vec) -> Vec {
    let w = zeros(input.dim)
    w = onlinecluster_update(w, input, 0.01)
    return w
}

fn train() {
    let data = random(64, 0.0, 1.0)
    let iter = 0
    while iter < 100 {
        let result = plastic_learn(data)
        iter = iter + 1
    }
}

train()
"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    profile_execution(code_plastic)
    profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs().sort_stats('cumulative').print_stats(15)
    print(s.getvalue())
    
    # Test 3: Pattern matching
    print("\n🔀 Test 3: Pattern matching (1000 itérations)")
    code_match = """
fn classify(x: int) -> string {
    match x {
        0 -> "zero"
        1 -> "one"
        2 -> "two"
        3 -> "three"
        _ -> "other"
    }
}

fn test_matching() {
    let iter = 0
    while iter < 1000 {
        let r1 = classify(0)
        let r2 = classify(1)
        let r3 = classify(2)
        let r4 = classify(5)
        iter = iter + 1
    }
}

test_matching()
"""
    
    profiler = cProfile.Profile()
    profiler.enable()
    profile_execution(code_match)
    profiler.disable()
    
    s = io.StringIO()
    ps = pstats.Stats(profiler, stream=s)
    ps.strip_dirs().sort_stats('cumulative').print_stats(15)
    print(s.getvalue())


def generate_report():
    """Génère un rapport de profiling"""
    print("\n" + "=" * 80)
    print("📝 GÉNÉRATION DU RAPPORT DE PROFILING")
    print("=" * 80)
    
    report = """
# Rapport de Profiling NORMiL
## Analyse des Performances

### Méthodologie
- Outil: cProfile (Python standard library)
- Métriques: temps cumulé, temps propre, nombre d'appels
- Fichiers testés: 5 exemples de Phase 7
- Opérations spécifiques: vectorielles, plasticité, pattern matching

### Zones à optimiser identifiées
(À compléter après analyse des résultats)

### Recommandations
(À compléter après analyse)

### Prochaines étapes
1. Optimiser les fonctions les plus coûteuses
2. Réduire le nombre d'appels pour les fonctions fréquentes
3. Implémenter du caching où pertinent
4. Optimiser les allocations mémoire
"""
    
    with open("profiling_report.md", "w", encoding="utf-8") as f:
        f.write(report)
    
    print("✅ Rapport de base généré: profiling_report.md")
    print("   (À compléter avec les résultats d'analyse)")


if __name__ == "__main__":
    print("🚀 Démarrage du profiling NORMiL...")
    print()
    
    try:
        # Profiling des fichiers de test
        run_profiling()
        
        # Profiling d'opérations spécifiques
        profile_specific_operations()
        
        # Générer le rapport
        generate_report()
        
        print("\n" + "=" * 80)
        print("✅ PROFILING TERMINÉ")
        print("=" * 80)
        print("\nConsultez les fichiers .stats générés pour analyse détaillée")
        print("Consultez profiling_report.md pour le rapport")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Profiling interrompu par l'utilisateur")
    except Exception as e:
        print(f"\n\n❌ Erreur durant le profiling: {e}")
        import traceback
        traceback.print_exc()
