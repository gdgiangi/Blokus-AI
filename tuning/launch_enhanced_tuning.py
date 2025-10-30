"""
Enhanced Hyperparameter Tuning Launch Script
Starts tuning with improved tie-breaking, adaptive opponents, and champion battles.
Uses current champions as baseline.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tuning.advanced_hyperparameter_tuning import (
    AdvancedHyperparameterTuner,
    TuningConfig
)


def main():
    """Launch enhanced tuning session with diverse champion pool"""
    
    print("=" * 80)
    print("ENHANCED HYPERPARAMETER TUNING")
    print("=" * 80)
    print("\nNew Features:")
    print("✓ Multi-criteria tie-breaking (win rate → avg score → timestamp)")
    print("✓ Adaptive sample sizes (150 games for high performers)")
    print("✓ Champion-vs-champion battles")
    print("✓ Adaptive opponents using champion pool (ACTIVE FROM START)")
    print("✓ Diverse champion pool initialization")
    print("\nStarting Configuration:")
    print("  🏆 5 Diverse Champions in pool:")
    print("     1. Uniform Baseline (all weights = 1.0)")
    print("     2. Aggressive Archetype (high opponent disruption)")
    print("     3. Balanced Archetype (moderate all-around)")
    print("     4. Defensive Archetype (high mobility & survival)")
    print("     5. UI Balanced Strategy (matches real UI opponents)")
    print("  🎯 Adaptive opponents ENABLED from Generation 1")
    print("  🧬 Genetic algorithm will compete against evolving champions")
    print("=" * 80)
    print()
    
    # Get user input for session configuration
    session_name = input("Session name (default: Enhanced_Phase2): ").strip()
    if not session_name:
        session_name = "Enhanced_Phase2"
    
    hours = input("Max duration in hours (default: 4): ").strip()
    try:
        max_hours = float(hours) if hours else 4.0
    except ValueError:
        max_hours = 4.0
    
    games_input = input("Games per evaluation (default: 50): ").strip()
    try:
        games_per_eval = int(games_input) if games_input else 50
    except ValueError:
        games_per_eval = 50
    
    # Create enhanced configuration
    config = TuningConfig(
        session_name=session_name,
        max_duration_hours=max_hours,
        games_per_evaluation=games_per_eval,
        champion_pool_size=5,
        population_size=20,
        mutation_rate=0.15,
        crossover_rate=0.7,
        elite_ratio=0.25,
        save_interval_minutes=10,
        verbose_logging=True,
        n_jobs=None,  # Use all CPUs
        
        # Enhanced features
        high_performer_threshold=0.90,  # 90%+ triggers re-evaluation
        high_performer_games=150,  # Increased sample size
        enable_champion_battles=True,  # Head-to-head battles
        champion_battle_games=100,  # Battle sample size
        use_adaptive_opponents=True,  # Use champion pool as opponents
        
        # Optimization parameters
        convergence_patience=50,
        convergence_threshold=0.001,
        simulated_annealing_temp=1.5,
        simulated_annealing_cooling=0.95,
        pso_particles=15,
        pso_inertia=0.7,
        pso_cognitive=1.5,
        pso_social=1.5
    )
    
    print(f"\nConfiguration:")
    print(f"  Session: {config.session_name}")
    print(f"  Duration: {config.max_duration_hours:.1f} hours")
    print(f"  Base games/evaluation: {config.games_per_evaluation}")
    print(f"  High-performer games: {config.high_performer_games}")
    print(f"  Champion battles: {config.enable_champion_battles}")
    print(f"  Adaptive opponents: {config.use_adaptive_opponents}")
    print()
    
    confirm = input("Start tuning? (yes/no): ").strip().lower()
    if confirm not in ['yes', 'y']:
        print("Tuning cancelled.")
        return
    
    # Initialize tuner with enhanced configuration
    tuner = AdvancedHyperparameterTuner(config)
    
    print("\n" + "=" * 80)
    print("STARTING ENHANCED TUNING SESSION")
    print("=" * 80 + "\n")
    
    # Display initial champion pool status
    tuner.session_manager.log_champion_pool_status()
    
    # Run initial champion battle to establish baseline rankings
    if config.enable_champion_battles:
        print("\n🔥 Running initial champion battle to establish rankings...")
        battle_results = tuner.conduct_champion_battles()
        if battle_results:
            print(f"✓ Initial battle complete")
            tuner.session_manager.log_champion_pool_status()
    
    # Run genetic algorithm with enhanced features
    print("\n" + "=" * 80)
    print("PHASE 1: Genetic Algorithm with Adaptive Opponents")
    print("=" * 80 + "\n")
    
    try:
        best_weights = tuner.genetic_algorithm()
        
        # Run another champion battle if enabled
        if config.enable_champion_battles:
            print("\n" + "=" * 80)
            print("PHASE 2: Final Champion Battle")
            print("=" * 80 + "\n")
            final_battle = tuner.conduct_champion_battles()
        
        # Final summary
        print("\n" + "=" * 80)
        print("TUNING SESSION COMPLETE")
        print("=" * 80)
        
        tuner.session_manager.log_champion_pool_status()
        
        if best_weights:
            print(f"\nBest weights found:")
            for name, weight in sorted(best_weights.items()):
                print(f"  {name}: {weight:.4f}")
        
        print(f"\nSession saved to: {tuner.session_manager.session_dir}")
        
    except KeyboardInterrupt:
        print("\n\nTuning interrupted by user.")
        print("Saving progress...")
        tuner.session_manager.save_progress()
        print(f"Progress saved to: {tuner.session_manager.session_dir}")
    except Exception as e:
        print(f"\n\nError during tuning: {e}")
        import traceback
        traceback.print_exc()
        print("\nSaving progress...")
        tuner.session_manager.save_progress()
        print(f"Progress saved to: {tuner.session_manager.session_dir}")


if __name__ == "__main__":
    main()
