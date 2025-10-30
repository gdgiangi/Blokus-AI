"""
Test and Validation Script for Advanced Hyperparameter Tuning
Ensures all components work correctly before running long tuning sessions.
"""

import sys
import time
import json
from pathlib import Path

# Add the project root to the path
sys.path.append(str(Path(__file__).parent))

from tuning.advanced_hyperparameter_tuning import (
    TuningConfig, AdvancedHyperparameterTuner, run_comprehensive_tuning,
    _play_championship_game
)


def test_single_game():
    """Test that a single championship game works correctly"""
    print("Testing single championship game...")
    
    # Test weights
    contender_weights = {
        "piece_size": 1.2,
        "new_paths": 1.9,
        "blocked_opponents": 2.4,
        "corner_control": 1.4,
        "compactness": 1.0,
        "flexibility": 1.7,
        "mobility": 0.7,
        "opponent_restriction": 0.9,
        "endgame_optimization": 1.2,
        "territory_expansion": 1.1
    }
    
    champion_weights = {
        "piece_size": 1.20,
        "new_paths": 1.89,
        "blocked_opponents": 2.42,
        "corner_control": 1.36,
        "compactness": 1.00,
        "flexibility": 1.71,
        "mobility": 0.73,
        "opponent_restriction": 0.89,
        "endgame_optimization": 1.20,
        "territory_expansion": 1.15
    }
    
    variant_weights = [
        {k: v * 1.1 for k, v in champion_weights.items()},
        {k: v * 0.9 for k, v in champion_weights.items()}
    ]
    
    game_config = (contender_weights, champion_weights, variant_weights, 0)
    
    start_time = time.time()
    result = _play_championship_game(game_config)
    game_time = time.time() - start_time
    
    print(f"✅ Single game completed in {game_time:.2f}s")
    print(f"   Contender: {result.contender_score}")
    print(f"   Champion: {result.champion_score}")
    print(f"   Variant 1: {result.variant1_score}")
    print(f"   Variant 2: {result.variant2_score}")
    print(f"   Winner: {result.winner}")
    print(f"   Game length: {result.game_length} moves")
    
    return True


def test_weight_evaluation():
    """Test weight evaluation with multiple games"""
    print("\nTesting weight evaluation...")
    
    config = TuningConfig(
        session_name="test",
        max_duration_hours=0.1,  # 6 minutes
        games_per_evaluation=4,  # Small number for testing
        n_jobs=2  # Use 2 processes for testing
    )
    
    tuner = AdvancedHyperparameterTuner(config)
    
    test_weights = {
        "piece_size": 1.2,
        "new_paths": 1.9,
        "blocked_opponents": 2.4,
        "corner_control": 1.4,
        "compactness": 1.0,
        "flexibility": 1.7,
        "mobility": 0.7,
        "opponent_restriction": 0.9,
        "endgame_optimization": 1.2,
        "territory_expansion": 1.1
    }
    
    start_time = time.time()
    result = tuner.evaluate_weights(test_weights)
    eval_time = time.time() - start_time
    
    print(f"✅ Weight evaluation completed in {eval_time:.2f}s")
    print(f"   Win rate: {result.win_rate:.1%}")
    print(f"   Avg score: {result.avg_score:.1f}")
    print(f"   Avg score diff vs champion: {result.avg_score_diff:+.1f}")
    print(f"   Games played: {len(result.games)}")
    
    # Cleanup test session directory
    tuner.session_manager.cleanup()  # Close logger handlers first
    if tuner.session_manager.session_dir.exists():
        import shutil
        time.sleep(0.5)  # Brief delay for file handles to close on Windows
        try:
            shutil.rmtree(tuner.session_manager.session_dir)
            print("   Test session directory cleaned up")
        except PermissionError:
            print("   Warning: Could not clean up test directory (files in use)")
    
    return True


def test_session_management():
    """Test session management and persistence"""
    print("\nTesting session management...")
    
    config = TuningConfig(
        session_name="test_session",
        max_duration_hours=0.05,  # 3 minutes
        games_per_evaluation=3,
        save_interval_minutes=1,  # Save every minute
        n_jobs=1
    )
    
    tuner = AdvancedHyperparameterTuner(config)
    
    # Run a few evaluations
    test_weights_1 = tuner.session_manager.champion_weights.copy()
    test_weights_2 = tuner.generate_random_weights(test_weights_1, 0.2)
    
    result1 = tuner.evaluate_weights(test_weights_1)
    tuner.session_manager.add_evaluation(result1)
    
    result2 = tuner.evaluate_weights(test_weights_2)
    tuner.session_manager.add_evaluation(result2)
    
    # Test saving
    tuner.session_manager.save_progress()
    
    # Check files were created
    session_dir = tuner.session_manager.session_dir
    results_file = session_dir / "results.json"
    summary_file = session_dir / "summary_stats.txt"
    log_file = session_dir / "tuning.log"
    
    files_exist = all(f.exists() for f in [results_file, summary_file, log_file])
    
    if files_exist:
        print("✅ Session management working correctly")
        print(f"   Session directory: {session_dir}")
        print(f"   Files created: results.json, summary_stats.txt, tuning.log")
        
        # Test loading results
        with open(results_file) as f:
            data = json.load(f)
        
        print(f"   Evaluations saved: {len(data['evaluations'])}")
        print(f"   Best win rate: {data['best_win_rate']:.1%}")
    else:
        print("❌ Session management failed - files not created")
        return False
    
    # Cleanup
    import shutil
    time.sleep(0.5)  # Brief delay for file handles to close
    try:
        shutil.rmtree(session_dir)
        print("   Test session directory cleaned up")
    except PermissionError:
        print("   Warning: Could not clean up test directory (files in use)")
    
    return True


def test_short_tuning_run():
    """Test a very short comprehensive tuning run"""
    print("\nTesting short comprehensive tuning run...")
    
    # Run for just 2 minutes with minimal games
    start_time = time.time()
    
    try:
        config = TuningConfig(
            session_name="short_test",
            max_duration_hours=0.033,  # 2 minutes
            games_per_evaluation=2,  # Minimal games
            population_size=4,  # Small population
            n_jobs=2
        )
        
        tuner = AdvancedHyperparameterTuner(config)
        
        # Just test the adaptive random search phase
        original_duration = tuner.config.max_duration_hours
        tuner.config.max_duration_hours = 0.033  # 2 minutes
        
        # Run adaptive search for a short time
        iteration_count = 0
        start_tuning = time.time()
        
        while tuner.session_manager.should_continue() and iteration_count < 3:
            test_weights = tuner.generate_random_weights(
                tuner.session_manager.champion_weights, 0.2
            )
            result = tuner.evaluate_weights(test_weights)
            tuner.session_manager.add_evaluation(result)
            iteration_count += 1
        
        tuning_time = time.time() - start_tuning
        
        print(f"✅ Short tuning run completed in {tuning_time:.1f}s")
        print(f"   Iterations completed: {iteration_count}")
        print(f"   Total games played: {tuner.session_manager.total_games_played}")
        print(f"   Best win rate: {tuner.session_manager.best_win_rate:.1%}")
        
        # Cleanup
        tuner.session_manager.cleanup()
        if tuner.session_manager.session_dir.exists():
            import shutil
            time.sleep(0.5)
            try:
                shutil.rmtree(tuner.session_manager.session_dir)
                print("   Test session directory cleaned up")
            except PermissionError:
                print("   Warning: Could not clean up test directory (files in use)")
        
        return True
        
    except Exception as e:
        print(f"❌ Short tuning run failed: {e}")
        return False


def run_all_tests():
    """Run all validation tests"""
    print("🧪 Running Advanced Hyperparameter Tuning Validation Tests")
    print("=" * 60)
    
    tests = [
        ("Single Game", test_single_game),
        ("Weight Evaluation", test_weight_evaluation),
        ("Session Management", test_session_management),
        ("Short Tuning Run", test_short_tuning_run)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n🔍 {test_name}")
        print("-" * 40)
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"VALIDATION RESULTS: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The system is ready for comprehensive tuning.")
        return True
    else:
        print("⚠️  Some tests failed. Please check the issues before running long sessions.")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    
    if success:
        print("\n" + "=" * 60)
        print("🚀 SYSTEM VALIDATION COMPLETE")
        print("=" * 60)
        print("\nThe advanced hyperparameter tuning system is ready!")
        print("\nTo run a comprehensive tuning session:")
        print("  python advanced_hyperparameter_tuning.py")
        print("\nRecommended settings for hours-long sessions:")
        print("  • Duration: 4-8 hours")
        print("  • Games per evaluation: 15-25")
        print("  • This will play 10,000-50,000+ games total")
        print("\nThe system will:")
        print("  ✓ Always test contenders vs champion + 2 variants")
        print("  ✓ Use adaptive search + genetic algorithms")
        print("  ✓ Save progress every 10 minutes")
        print("  ✓ Provide detailed logging and statistics")
        print("  ✓ Handle interruptions gracefully")
    else:
        print("\n❌ Please fix the issues before proceeding with tuning.")
        sys.exit(1)