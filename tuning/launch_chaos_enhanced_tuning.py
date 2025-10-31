"""
Random AI + Champion Battle Enhanced Tuning Launch Script
Features:
- Random AI player injection for chaos and robustness testing
- Frequent champion battles for accurate rankings
- Breaking out of local optima with unpredictable opponents
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tuning.advanced_hyperparameter_tuning import (
    AdvancedHyperparameterTuner,
    TuningConfig
)


def main():
    """Launch chaos-injection enhanced tuning session"""
    
    print("=" * 80)
    print("🎲 RANDOM AI + CHAMPION BATTLE ENHANCED TUNING")
    print("=" * 80)
    print("\nBreaking Local Optima Features:")
    print("🎲 Random AI injection: 30% of games include chaotic opponents")
    print("⚔️  Frequent champion battles: Every 10 evaluations")
    print("🏆 Battle-weighted rankings: 30% weight on head-to-head performance")
    print("🔄 Adaptive opponent selection from evolving champion pool")
    print("📊 Enhanced evaluation metrics with unpredictability resistance")
    
    print("\nWhy This Breaks Local Optima:")
    print("  • Random opponents disrupt overfitted strategies")
    print("  • Forces strategies to be robust against chaos")
    print("  • Champion battles prevent ranking stagnation")
    print("  • Continuous competitive pressure from pool evolution")
    print("  • Tests real-world performance vs unpredictable players")
    
    print("\nConfiguration:")
    print("  � Starting from Latest Champion (GeniusTime 60% win rate)")
    print("  �🎯 30% random AI injection (replaces defensive strategy)")
    print("  ⚔️  Champion battles every 10 evaluations")
    print("  🏆 Battle results weighted at 30% in final rankings")
    print("  🧬 Genetic algorithm + adaptive opponents")
    print("  📈 50 games per evaluation (75% vs strategies, 25% vs random)")
    print("=" * 80)
    print()
    
    # Get user input
    session_name = input("Session name (default: ChaosBreaker): ").strip()
    if not session_name:
        session_name = "ChaosBreaker"
    
    hours = input("Max duration in hours (default: 3): ").strip()
    try:
        max_hours = float(hours) if hours else 3.0
    except ValueError:
        max_hours = 3.0
    
    games = input("Games per evaluation (default: 50): ").strip()
    try:
        games_per_eval = int(games) if games else 50
    except ValueError:
        games_per_eval = 50
    
    random_freq = input("Random AI frequency 0.0-1.0 (default: 0.3): ").strip()
    try:
        random_frequency = float(random_freq) if random_freq else 0.3
    except ValueError:
        random_frequency = 0.3
    
    battle_freq = input("Champion battle frequency (default: 10): ").strip()
    try:
        battle_frequency = int(battle_freq) if battle_freq else 10
    except ValueError:
        battle_frequency = 10
    
    # Create enhanced configuration
    config = TuningConfig(
        session_name=session_name,
        max_duration_hours=max_hours,
        games_per_evaluation=games_per_eval,
        
        # Enhanced genetic algorithm parameters
        population_size=24,
        mutation_rate=0.15,  # Higher for more exploration
        crossover_rate=0.7,
        elite_ratio=0.25,  # Keep more elites
        
        # Champion pool management
        champion_pool_size=5,
        champion_weight_variation=0.20,  # Increased variation
        
        # Random AI injection parameters
        random_opponent_frequency=random_frequency,
        random_replacement_strategy="defensive",  # Replace defensive with chaos
        
        # Champion battle parameters  
        champion_battle_frequency=battle_frequency,
        champion_battle_weight=0.3,  # 30% weight on battle performance
        
        # Evaluation enhancements
        high_performer_threshold=0.85,  # Lower threshold for re-evaluation
        high_performer_games=100,  # More games for validation
        enable_champion_battles=True,
        champion_battle_games=75,  # Sufficient for statistical significance
        
        # Advanced optimization
        use_adaptive_opponents=True,
        convergence_patience=40,  # More patience before giving up
        convergence_threshold=0.002,
        
        # Simulated annealing for local optima escape
        simulated_annealing_temp=2.5,
        simulated_annealing_cooling=0.92,
        
        # Logging and saving
        save_interval_minutes=8,
        verbose_logging=True
    )
    
    print(f"\n🚀 Starting '{session_name}' session...")
    print(f"⏱️  Duration: {max_hours} hours")
    print(f"🎮 Games per evaluation: {games_per_eval}")
    print(f"🎲 Random AI injection: {random_frequency:.0%} of games")
    print(f"⚔️  Champion battles every: {battle_frequency} evaluations")
    print(f"🏆 Battle weight in rankings: {config.champion_battle_weight:.0%}")
    print("=" * 80)
    
    # Initialize and run tuner
    try:
        tuner = AdvancedHyperparameterTuner(config)
        
        print(f"\n🧬 Starting Genetic Algorithm with chaos injection...")
        print("   This will test strategies against:")
        print("   • Aggressive opponents (territorial)")
        print("   • Balanced opponents (all-around)")  
        print("   • Defensive opponents (survival-focused)")
        print(f"   • Random chaos agents ({random_frequency:.0%} of games)")
        print("   • Evolving champion strategies")
        print()
        
        # Run genetic algorithm with enhanced features
        best_weights = tuner.genetic_algorithm()
        
        print("\n" + "=" * 80)
        print("🏆 CHAOS-RESISTANT OPTIMIZATION COMPLETE!")
        print("=" * 80)
        
        if best_weights:
            print("✓ Found chaos-resistant strategy weights:")
            for name, weight in sorted(best_weights.items()):
                print(f"  {name:30s}: {weight:.3f}")
            
            print(f"\n✓ Best win rate: {tuner.session_manager.best_win_rate:.1%}")
            print(f"✓ Total evaluations: {len(tuner.session_manager.evaluations)}")
            print(f"✓ Total games played: {tuner.session_manager.total_games_played}")
            print(f"✓ Champion battles: {tuner.session_manager.total_battles_conducted}")
            
            # Show champion pool
            print(f"\n🏆 Final Champion Pool:")
            for i, champ in enumerate(tuner.session_manager.champion_pool, 1):
                print(f"  {i}. {champ.win_rate:.1%} win rate, {champ.avg_score:.1f} avg score")
                
        else:
            print("⚠️  No optimal weights found")
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Tuning interrupted by user")
        if 'tuner' in locals():
            tuner.session_manager.save_progress()
            print("✓ Progress saved")
    except Exception as e:
        print(f"\n❌ Error during tuning: {e}")
        if 'tuner' in locals():
            tuner.session_manager.save_progress()
            print("✓ Progress saved despite error")
    finally:
        if 'tuner' in locals():
            tuner.session_manager.cleanup()


if __name__ == "__main__":
    main()