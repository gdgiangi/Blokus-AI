"""
Hyperparameter Tuning for Blokus AI
Simulates games to find optimal heuristic weights.
Parallelized for faster execution.
"""

import random
import json
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import itertools
from multiprocessing import Pool, cpu_count
from functools import partial
import time

from game_state import GameState
from board import PlayerColor
from ai_player_enhanced import OptimizedAIStrategy, AIPlayer, EnhancedHeuristic


# Global helper function for parallel execution (must be at module level)
def _play_single_game(config: Tuple[Dict[str, float], List[Dict[str, float]], bool]) -> Tuple[int, int]:
    """
    Play a single game with given weight configurations.
    Returns (blue_score, max_score) tuple.
    
    This function must be at module level for multiprocessing.Pool to pickle it.
    """
    weights, opponent_weights, verbose = config
    
    game = GameState(num_players=4)
    game.start_game()
    
    # Create AI players with different weights
    ai_players = {
        PlayerColor.BLUE: AIPlayer(PlayerColor.BLUE, OptimizedAIStrategy(weights)),
        PlayerColor.YELLOW: AIPlayer(PlayerColor.YELLOW, OptimizedAIStrategy(opponent_weights[0])),
        PlayerColor.RED: AIPlayer(PlayerColor.RED, OptimizedAIStrategy(opponent_weights[1])),
        PlayerColor.GREEN: AIPlayer(PlayerColor.GREEN, OptimizedAIStrategy(opponent_weights[2]))
    }
    
    move_count = 0
    max_moves = 200  # Safety limit
    
    while not game.is_game_over() and move_count < max_moves:
        current_color = game.get_current_color()
        ai_player = ai_players[current_color]
        
        # Get AI's move
        best_move = ai_player.get_move(game)
        
        if best_move is None:
            game.pass_turn()
        else:
            success = game.place_piece(
                best_move.piece_type,
                best_move.piece,
                best_move.row,
                best_move.col
            )
            
            if not success:
                if verbose:
                    print(f"WARNING: Failed to place piece for {current_color.value}")
                game.pass_turn()
        
        move_count += 1
    
    # Get final scores
    scores = game.get_scores()
    blue_score = scores[PlayerColor.BLUE]
    max_score = max(scores.values())
    
    if verbose:
        print(f"\nGame finished in {move_count} moves")
        for color, score in game.get_rankings():
            print(f"  {color.value}: {score}")
    
    return (blue_score, max_score)


class HyperparameterTuner:
    """Tunes AI heuristic weights through simulated games (parallelized)"""
    
    def __init__(self, n_jobs: Optional[int] = None):
        """
        Initialize tuner.
        
        Args:
            n_jobs: Number of parallel processes. None or -1 uses all CPUs, 1 disables parallelization.
        """
        self.results = []
        self.best_weights = None
        self.best_win_rate = 0.0
        
        # Determine number of processes
        if n_jobs is None or n_jobs == -1:
            self.n_jobs = cpu_count()
        elif n_jobs == 0:
            self.n_jobs = 1
        else:
            self.n_jobs = max(1, n_jobs)
        
        print(f"Using {self.n_jobs} parallel processes")
    
    def generate_random_weights(self, base_weights: Dict[str, float], 
                               variation: float = 0.3) -> Dict[str, float]:
        """
        Generate random weight variations around base weights.
        
        Args:
            base_weights: Starting weights
            variation: How much to vary (0.3 = ±30%)
        """
        new_weights = {}
        for name, weight in base_weights.items():
            # Vary weight by ±variation%
            factor = 1.0 + random.uniform(-variation, variation)
            new_weights[name] = max(0.1, weight * factor)  # Keep positive
        return new_weights
    
    def evaluate_weights(self, weights: Dict[str, float], 
                        num_games: int = 10,
                        verbose: bool = False) -> float:
        """
        Evaluate a set of weights by playing multiple games (parallelized).
        Returns average win rate when Blue uses these weights.
        """
        # Base weights for opponents
        base_weights = {
            "piece_size": 1.5,
            "new_paths": 2.5,
            "blocked_opponents": 2.0,
            "corner_control": 1.5,
            "compactness": 1.0,
            "flexibility": 2.0,
            "mobility": 1.0,
            "opponent_restriction": 2.0,
            "endgame_optimization": 1.0,
            "territory_expansion": 1.0
        }
        
        # Generate opponent weights for all games
        game_configs = []
        for game_num in range(num_games):
            opponent_weights = [
                self.generate_random_weights(base_weights, 0.2) 
                for _ in range(3)
            ]
            game_configs.append((weights, opponent_weights, verbose and game_num == 0))
        
        # Play games in parallel
        if self.n_jobs > 1:
            with Pool(processes=self.n_jobs) as pool:
                results = pool.map(_play_single_game, game_configs)
        else:
            # Sequential execution
            results = [_play_single_game(config) for config in game_configs]
        
        # Calculate win rate
        wins = 0
        total_score = 0
        
        for blue_score, max_score in results:
            total_score += blue_score
            if blue_score == max_score:
                wins += 1
        
        win_rate = wins / num_games
        avg_score = total_score / num_games
        
        if verbose:
            print(f"Win rate: {win_rate:.1%}, Avg score: {avg_score:.1f}")
        
        return win_rate
    
    def random_search(self, base_weights: Dict[str, float],
                     num_iterations: int = 20,
                     games_per_iteration: int = 10,
                     variation: float = 0.3) -> Dict[str, float]:
        """
        Random search for optimal weights (parallelized).
        
        Args:
            base_weights: Starting weights
            num_iterations: How many random weight sets to try
            games_per_iteration: Games to play for each weight set
            variation: How much to vary weights
        """
        print("=" * 60)
        print("Hyperparameter Tuning - Random Search (Parallelized)")
        print("=" * 60)
        print(f"Iterations: {num_iterations}")
        print(f"Games per iteration: {games_per_iteration}")
        print(f"Variation: ±{variation*100}%")
        print(f"Parallel processes: {self.n_jobs}")
        print(f"Total games: {num_iterations * games_per_iteration}\n")
        
        start_time = time.time()
        
        best_weights = base_weights.copy()
        best_win_rate = self.evaluate_weights(best_weights, games_per_iteration, verbose=True)
        
        print(f"\nBaseline win rate: {best_win_rate:.1%}")
        print("\nSearching for better weights...\n")
        
        for i in range(num_iterations):
            iter_start = time.time()
            
            # Generate random weight variation
            test_weights = self.generate_random_weights(base_weights, variation)
            
            # Evaluate
            win_rate = self.evaluate_weights(test_weights, games_per_iteration)
            
            iter_time = time.time() - iter_start
            elapsed = time.time() - start_time
            estimated_total = (elapsed / (i + 1)) * (num_iterations + 1)  # +1 for baseline
            remaining = estimated_total - elapsed
            
            print(f"Iteration {i+1}/{num_iterations}: Win rate = {win_rate:.1%} "
                  f"[{iter_time:.1f}s] (Est. {remaining/60:.1f}m remaining)", end="")
            
            # Track best
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_weights = test_weights
                print(" ⭐ NEW BEST!")
            else:
                print()
            
            self.results.append({
                'weights': test_weights,
                'win_rate': win_rate
            })
        
        total_time = time.time() - start_time
        
        self.best_weights = best_weights
        self.best_win_rate = best_win_rate
        
        print("\n" + "=" * 60)
        print(f"Completed in {total_time/60:.1f} minutes")
        print(f"Best win rate: {best_win_rate:.1%}")
        print("Best weights:")
        for name, weight in sorted(best_weights.items()):
            print(f"  {name}: {weight:.2f}")
        print("=" * 60)
        
        return best_weights
    
    def grid_search(self, heuristic_name: str,
                   base_weights: Dict[str, float],
                   weight_range: Tuple[float, float, float],
                   games_per_weight: int = 10) -> float:
        """
        Grid search for a single heuristic weight (parallelized).
        
        Args:
            heuristic_name: Name of heuristic to tune
            base_weights: Base weights for all heuristics
            weight_range: (min, max, step) for the weight
            games_per_weight: Games to play for each weight value
        """
        min_w, max_w, step = weight_range
        test_values = []
        current = min_w
        while current <= max_w:
            test_values.append(current)
            current += step
        
        total_configs = len(test_values)
        
        print(f"\n{'='*60}")
        print(f"Grid Search for '{heuristic_name}' (Parallelized)")
        print(f"{'='*60}")
        print(f"Testing {total_configs} values: {min_w} to {max_w} (step {step})")
        print(f"Games per value: {games_per_weight}")
        print(f"Parallel processes: {self.n_jobs}")
        print(f"Total games: {total_configs * games_per_weight}\n")
        
        start_time = time.time()
        
        best_weight = base_weights[heuristic_name]
        best_win_rate = 0.0
        
        for i, weight_value in enumerate(test_values, 1):
            config_start = time.time()
            
            test_weights = base_weights.copy()
            test_weights[heuristic_name] = weight_value
            
            win_rate = self.evaluate_weights(test_weights, games_per_weight)
            
            config_time = time.time() - config_start
            elapsed = time.time() - start_time
            estimated_total = (elapsed / i) * total_configs
            remaining = estimated_total - elapsed
            
            print(f"Value {i}/{total_configs} ({weight_value:.1f}): Win rate = {win_rate:.1%} "
                  f"[{config_time:.1f}s] (Est. {remaining/60:.1f}m remaining)", end="")
            
            if win_rate > best_win_rate:
                best_win_rate = win_rate
                best_weight = weight_value
                print(" ⭐ NEW BEST!")
            else:
                print()
        
        total_time = time.time() - start_time
        
        print(f"\n{'='*60}")
        print(f"Completed in {total_time/60:.1f} minutes")
        print(f"Best weight for {heuristic_name}: {best_weight:.1f} ({best_win_rate:.1%})")
        print(f"{'='*60}")
        
        return best_weight
    
    def save_results(self, filename: str = "tuning_results.json"):
        """Save tuning results to file"""
        data = {
            'best_weights': self.best_weights,
            'best_win_rate': self.best_win_rate,
            'all_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"\nResults saved to {filename}")
    
    def load_results(self, filename: str = "tuning_results.json"):
        """Load tuning results from file"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        self.best_weights = data['best_weights']
        self.best_win_rate = data['best_win_rate']
        self.results = data['all_results']
        
        print(f"Results loaded from {filename}")
        print(f"Best win rate: {self.best_win_rate:.1%}")


def quick_tune(n_jobs: Optional[int] = None):
    """
    Quick tuning session with reasonable parameters.
    
    Args:
        n_jobs: Number of parallel processes (None = use all CPUs)
    """
    base_weights = {
        "piece_size": 1.5,
        "new_paths": 2.5,
        "blocked_opponents": 2.0,
        "corner_control": 1.5,
        "compactness": 1.0,
        "flexibility": 2.0,
        "mobility": 1.0,
        "opponent_restriction": 2.0,
        "endgame_optimization": 1.0,
        "territory_expansion": 1.0
    }
    
    tuner = HyperparameterTuner(n_jobs=n_jobs)
    
    # Random search with 15 iterations, 8 games each
    best_weights = tuner.random_search(
        base_weights,
        num_iterations=15,
        games_per_iteration=8,
        variation=0.25
    )
    
    # Save results
    tuner.save_results()
    
    return best_weights


def intensive_tune(n_jobs: Optional[int] = None):
    """
    More intensive tuning (will take longer).
    
    Args:
        n_jobs: Number of parallel processes (None = use all CPUs)
    """
    base_weights = {
        "piece_size": 1.5,
        "new_paths": 2.5,
        "blocked_opponents": 2.0,
        "corner_control": 1.5,
        "compactness": 1.0,
        "flexibility": 2.0,
        "mobility": 1.0,
        "opponent_restriction": 2.0,
        "endgame_optimization": 1.0,
        "territory_expansion": 1.0
    }
    
    tuner = HyperparameterTuner(n_jobs=n_jobs)
    
    # Random search
    best_weights = tuner.random_search(
        base_weights,
        num_iterations=30,
        games_per_iteration=15,
        variation=0.3
    )
    
    # Fine-tune with grid search on top heuristics
    print("\n\nFine-tuning top heuristics...")
    for heuristic in ["new_paths", "opponent_restriction", "flexibility"]:
        current_val = best_weights[heuristic]
        best_weights[heuristic] = tuner.grid_search(
            heuristic,
            best_weights,
            (current_val * 0.5, current_val * 1.5, current_val * 0.1),
            games_per_weight=10
        )
    
    tuner.save_results("tuning_results_intensive.json")
    
    return best_weights


if __name__ == "__main__":
    print("\n🎮 Blokus AI Hyperparameter Tuning\n")
    print("Choose tuning mode:")
    print("1. Quick tune (~5-10 minutes)")
    print("2. Intensive tune (~30-60 minutes)")
    print("3. Custom")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    if choice == "1":
        best_weights = quick_tune()
    elif choice == "2":
        best_weights = intensive_tune()
    else:
        print("Running quick tune as default...")
        best_weights = quick_tune()
    
    print("\n✅ Tuning complete!")
    print("\nUse these optimized weights in your AI:")
    print(json.dumps(best_weights, indent=2))
