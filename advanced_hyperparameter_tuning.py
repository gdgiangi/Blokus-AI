"""
Advanced Comprehensive Hyperparameter Tuning for Blokus AI
Designed for hours-long tuning sessions with sophisticated opponent management,
comprehensive reporting, and advanced optimization strategies.
"""

import random
import json
import time
import os
import math
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from multiprocessing import Pool, cpu_count, Manager, Value
from functools import partial
import logging
from pathlib import Path

from game_state import GameState
from board import PlayerColor
from ai_player_enhanced import OptimizedAIStrategy, AIPlayer, EnhancedHeuristic


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning session"""
    session_name: str
    max_duration_hours: float
    games_per_evaluation: int
    champion_weight_variation: float = 0.15  # ±15% variation for champion variants
    population_size: int = 20  # For genetic algorithm
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_ratio: float = 0.2
    save_interval_minutes: int = 10
    verbose_logging: bool = True
    n_jobs: Optional[int] = None


@dataclass
class GameResult:
    """Result of a single game"""
    contender_score: int
    champion_score: int
    variant1_score: int
    variant2_score: int
    winner: Optional[str]  # PlayerColor as string for JSON serialization
    game_length: int
    timestamp: float


@dataclass
class EvaluationResult:
    """Result of evaluating a weight configuration"""
    weights: Dict[str, float]
    games: List[GameResult]
    win_rate: float
    avg_score: float
    avg_score_diff: float  # vs champion
    evaluation_time: float
    timestamp: float
    
    def to_dict(self) -> dict:
        return {
            **asdict(self),
            'games': [asdict(game) for game in self.games]
        }


class SessionManager:
    """Manages tuning session state, logging, and persistence"""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.session_dir = Path(f"tuning_sessions/{config.session_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        self.session_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.logger = self._setup_logging()
        
        # Session state
        self.start_time = time.time()
        self.last_save_time = time.time()
        self.evaluations = []
        self.best_weights = None
        self.best_win_rate = 0.0
        self.champion_weights = self._load_champion_weights()
        self.champion_variants = self._generate_champion_variants()
        
        # Statistics tracking
        self.total_games_played = 0
        self.total_evaluation_time = 0
        self.win_rate_history = deque(maxlen=100)  # Last 100 evaluations
        
        self.logger.info(f"Starting tuning session: {config.session_name}")
        self.logger.info(f"Session directory: {self.session_dir}")
        self.logger.info(f"Max duration: {config.max_duration_hours:.1f} hours")
        self.logger.info(f"Games per evaluation: {config.games_per_evaluation}")
        
    def _setup_logging(self) -> logging.Logger:
        """Setup comprehensive logging"""
        logger = logging.getLogger(f"tuning_{self.config.session_name}")
        logger.setLevel(logging.INFO if self.config.verbose_logging else logging.WARNING)
        
        # File handler with UTF-8 encoding
        log_file = self.session_dir / "tuning.log"
        file_handler = logging.FileHandler(log_file, encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        
        # Console handler with UTF-8 encoding
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        
        # Try to set UTF-8 encoding for console output on Windows
        try:
            import sys
            if hasattr(sys.stdout, 'reconfigure'):
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        except:
            pass  # Fallback gracefully if encoding setup fails
        
        # Formatter
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        logger.addHandler(file_handler)
        logger.addHandler(console_handler)
        
        return logger
    
    def _load_champion_weights(self) -> Dict[str, float]:
        """Load the current champion weights"""
        try:
            with open("tuning_results_intensive.json", "r") as f:
                data = json.load(f)
                return data["best_weights"]
        except FileNotFoundError:
            # Fallback to default weights
            self.logger.warning("No champion weights found, using defaults")
            return {
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
    
    def _generate_champion_variants(self) -> List[Dict[str, float]]:
        """Generate two variants of the champion weights"""
        variants = []
        for i in range(2):
            variant = {}
            for name, weight in self.champion_weights.items():
                variation = random.uniform(-self.config.champion_weight_variation, 
                                         self.config.champion_weight_variation)
                variant[name] = max(0.1, weight * (1 + variation))
            variants.append(variant)
        
        self.logger.info("Generated champion variants:")
        for i, variant in enumerate(variants):
            self.logger.info(f"  Variant {i+1}: {self._format_weights(variant)}")
        
        return variants
    
    def _format_weights(self, weights: Dict[str, float]) -> str:
        """Format weights for logging"""
        return ", ".join([f"{k}:{v:.2f}" for k, v in sorted(weights.items())])
    
    def should_continue(self) -> bool:
        """Check if tuning should continue"""
        elapsed_hours = (time.time() - self.start_time) / 3600
        return elapsed_hours < self.config.max_duration_hours
    
    def should_save(self) -> bool:
        """Check if it's time to save progress"""
        elapsed_minutes = (time.time() - self.last_save_time) / 60
        return elapsed_minutes >= self.config.save_interval_minutes
    
    def add_evaluation(self, result: EvaluationResult):
        """Add a new evaluation result"""
        self.evaluations.append(result)
        self.total_games_played += len(result.games)
        self.total_evaluation_time += result.evaluation_time
        self.win_rate_history.append(result.win_rate)
        
        if result.win_rate > self.best_win_rate:
            self.best_win_rate = result.win_rate
            self.best_weights = result.weights.copy()
            self.logger.info(f"** NEW BEST! ** Win rate: {result.win_rate:.1%} - {self._format_weights(result.weights)}")
        
        # Auto-save if needed
        if self.should_save():
            self.save_progress()
    
    def save_progress(self):
        """Save current progress to files"""
        self.last_save_time = time.time()
        
        # Main results file
        results_file = self.session_dir / "results.json"
        data = {
            "config": asdict(self.config),
            "start_time": self.start_time,
            "champion_weights": self.champion_weights,
            "champion_variants": self.champion_variants,
            "best_weights": self.best_weights,
            "best_win_rate": self.best_win_rate,
            "total_games_played": self.total_games_played,
            "total_evaluation_time": self.total_evaluation_time,
            "evaluations": [result.to_dict() for result in self.evaluations]
        }
        
        with open(results_file, "w") as f:
            json.dump(data, f, indent=2)
        
        # Summary statistics
        self._save_summary_stats()
        
        self.logger.info(f"Progress saved to {results_file}")
    
    def cleanup(self):
        """Clean up resources, especially logger handlers"""
        # Close all logger handlers to release file locks
        for handler in self.logger.handlers[:]:
            handler.close()
            self.logger.removeHandler(handler)
    
    def _save_summary_stats(self):
        """Save summary statistics"""
        if not self.evaluations:
            return
        
        stats_file = self.session_dir / "summary_stats.txt"
        elapsed_time = time.time() - self.start_time
        
        with open(stats_file, "w") as f:
            f.write(f"=== Tuning Session Summary ===\n")
            f.write(f"Session: {self.config.session_name}\n")
            f.write(f"Started: {datetime.fromtimestamp(self.start_time)}\n")
            f.write(f"Duration: {elapsed_time/3600:.2f} hours\n")
            f.write(f"Total evaluations: {len(self.evaluations)}\n")
            f.write(f"Total games played: {self.total_games_played}\n")
            f.write(f"Games per hour: {self.total_games_played/(elapsed_time/3600):.1f}\n\n")
            
            f.write(f"=== Performance ===\n")
            f.write(f"Best win rate: {self.best_win_rate:.1%}\n")
            
            if self.win_rate_history:
                recent_avg = sum(list(self.win_rate_history)[-10:]) / min(10, len(self.win_rate_history))
                f.write(f"Recent avg win rate (last 10): {recent_avg:.1%}\n")
                
            f.write(f"Champion baseline: {self.champion_weights}\n")
            f.write(f"Best weights: {self.best_weights}\n\n")
            
            # Top 10 results
            top_results = sorted(self.evaluations, key=lambda x: x.win_rate, reverse=True)[:10]
            f.write(f"=== Top 10 Results ===\n")
            for i, result in enumerate(top_results, 1):
                f.write(f"{i:2d}. {result.win_rate:.1%} - {self._format_weights(result.weights)}\n")


# Global helper function for multiprocessing
def _play_championship_game(game_config: Tuple[Dict[str, float], Dict[str, float], List[Dict[str, float]], int]) -> GameResult:
    """
    Play a single championship game: Contender vs Champion vs 2 Variants
    Returns detailed game result.
    """
    contender_weights, champion_weights, variant_weights, game_id = game_config
    
    game = GameState(num_players=4)
    game.start_game()
    
    # Create AI players
    # Blue = Contender (being tested)
    # Yellow = Champion (current best)
    # Red = Champion Variant 1
    # Green = Champion Variant 2
    ai_players = {
        PlayerColor.BLUE: AIPlayer(PlayerColor.BLUE, OptimizedAIStrategy(contender_weights)),
        PlayerColor.YELLOW: AIPlayer(PlayerColor.YELLOW, OptimizedAIStrategy(champion_weights)),
        PlayerColor.RED: AIPlayer(PlayerColor.RED, OptimizedAIStrategy(variant_weights[0])),
        PlayerColor.GREEN: AIPlayer(PlayerColor.GREEN, OptimizedAIStrategy(variant_weights[1]))
    }
    
    move_count = 0
    max_moves = 300  # Increased safety limit
    
    while not game.is_game_over() and move_count < max_moves:
        current_color = game.get_current_color()
        ai_player = ai_players[current_color]
        
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
                game.pass_turn()
        
        move_count += 1
    
    # Get final scores
    scores = game.get_scores()
    rankings = game.get_rankings()
    winner = rankings[0][0] if rankings else PlayerColor.BLUE  # Color with highest score
    
    return GameResult(
        contender_score=scores[PlayerColor.BLUE],
        champion_score=scores[PlayerColor.YELLOW],
        variant1_score=scores[PlayerColor.RED],
        variant2_score=scores[PlayerColor.GREEN],
        winner=winner.value,  # Convert PlayerColor to string
        game_length=move_count,
        timestamp=time.time()
    )


class AdvancedHyperparameterTuner:
    """Advanced hyperparameter tuner with multiple optimization strategies"""
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.session_manager = SessionManager(config)
        
        # Determine number of processes
        if config.n_jobs is None or config.n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)  # Leave one CPU free
        else:
            self.n_jobs = max(1, config.n_jobs)
        
        self.session_manager.logger.info(f"Using {self.n_jobs} parallel processes")
    
    def evaluate_weights(self, weights: Dict[str, float]) -> EvaluationResult:
        """Evaluate weights by playing championship games"""
        start_time = time.time()
        
        # Prepare game configurations
        game_configs = []
        for game_id in range(self.config.games_per_evaluation):
            config = (
                weights,
                self.session_manager.champion_weights,
                self.session_manager.champion_variants,
                game_id
            )
            game_configs.append(config)
        
        # Play games in parallel
        if self.n_jobs > 1:
            with Pool(processes=self.n_jobs) as pool:
                game_results = pool.map(_play_championship_game, game_configs)
        else:
            game_results = [_play_championship_game(config) for config in game_configs]
        
        # Calculate statistics
        wins = sum(1 for result in game_results if result.winner == "blue")
        win_rate = wins / len(game_results)
        avg_score = sum(result.contender_score for result in game_results) / len(game_results)
        avg_champion_score = sum(result.champion_score for result in game_results) / len(game_results)
        avg_score_diff = avg_score - avg_champion_score
        
        evaluation_time = time.time() - start_time
        
        return EvaluationResult(
            weights=weights.copy(),
            games=game_results,
            win_rate=win_rate,
            avg_score=avg_score,
            avg_score_diff=avg_score_diff,
            evaluation_time=evaluation_time,
            timestamp=time.time()
        )
    
    def generate_random_weights(self, base_weights: Dict[str, float], 
                               variation: float = 0.3) -> Dict[str, float]:
        """Generate random weight variations"""
        new_weights = {}
        for name, weight in base_weights.items():
            factor = 1.0 + random.uniform(-variation, variation)
            new_weights[name] = max(0.1, weight * factor)
        return new_weights
    
    def adaptive_random_search(self):
        """
        Adaptive random search that adjusts based on recent performance
        """
        logger = self.session_manager.logger
        logger.info("Starting Adaptive Random Search")
        
        # Start with champion as baseline
        base_weights = self.session_manager.champion_weights.copy()
        current_variation = 0.25  # Start with 25% variation
        
        # Evaluate baseline
        logger.info("Evaluating champion baseline...")
        baseline_result = self.evaluate_weights(base_weights)
        self.session_manager.add_evaluation(baseline_result)
        logger.info(f"Champion baseline win rate: {baseline_result.win_rate:.1%}")
        
        iteration = 0
        stagnation_counter = 0
        last_improvement_time = time.time()
        
        while self.session_manager.should_continue():
            iteration += 1
            iter_start_time = time.time()
            
            # Adaptive variation based on recent performance
            if len(self.session_manager.win_rate_history) >= 10:
                recent_improvement = (
                    max(list(self.session_manager.win_rate_history)[-5:]) - 
                    max(list(self.session_manager.win_rate_history)[-10:-5])
                )
                
                if recent_improvement > 0.02:  # 2% improvement
                    current_variation = min(0.4, current_variation * 1.1)  # Increase exploration
                    stagnation_counter = 0
                elif recent_improvement < 0.005:  # Less than 0.5% improvement
                    stagnation_counter += 1
                    if stagnation_counter > 5:
                        current_variation = max(0.1, current_variation * 0.9)  # Reduce exploration
                
            # Generate candidate weights
            if stagnation_counter > 10:
                # More aggressive exploration when stagnating
                candidate_weights = self.generate_random_weights(
                    self.session_manager.best_weights or base_weights, 
                    current_variation * 1.5
                )
                logger.info(f"Aggressive exploration (stagnation={stagnation_counter})")
            else:
                # Normal exploration around best known weights
                candidate_weights = self.generate_random_weights(
                    self.session_manager.best_weights or base_weights, 
                    current_variation
                )
            
            # Evaluate candidate
            result = self.evaluate_weights(candidate_weights)
            self.session_manager.add_evaluation(result)
            
            # Logging and progress tracking
            elapsed = time.time() - self.session_manager.start_time
            iter_time = time.time() - iter_start_time
            remaining_time = (self.config.max_duration_hours * 3600) - elapsed
            
            improvement = ""
            if result.win_rate > baseline_result.win_rate:
                improvement = f" (+{result.win_rate - baseline_result.win_rate:.1%} vs baseline)"
                if result.win_rate == self.session_manager.best_win_rate:
                    improvement += " ** NEW BEST! **"
                    last_improvement_time = time.time()
            
            logger.info(
                f"Iter {iteration}: {result.win_rate:.1%}{improvement} "
                f"[var={current_variation:.1%}, {iter_time:.1f}s] "
                f"({remaining_time/3600:.1f}h remaining)"
            )
            
            # Check for long stagnation and restart with different base
            time_since_improvement = time.time() - last_improvement_time
            if time_since_improvement > 1800:  # 30 minutes without improvement
                logger.info("Long stagnation detected, restarting from new random base")
                base_weights = self.generate_random_weights(
                    self.session_manager.champion_weights, 0.5
                )
                current_variation = 0.3
                stagnation_counter = 0
                last_improvement_time = time.time()
        
        logger.info("Adaptive Random Search completed")
        return self.session_manager.best_weights
    
    def genetic_algorithm(self):
        """
        Genetic algorithm for weight optimization
        """
        logger = self.session_manager.logger
        logger.info("Starting Genetic Algorithm")
        
        population_size = self.config.population_size
        mutation_rate = self.config.mutation_rate
        crossover_rate = self.config.crossover_rate
        elite_size = int(population_size * self.config.elite_ratio)
        
        # Initialize population
        logger.info(f"Initializing population of {population_size}")
        population = []
        
        # Add champion as seed
        population.append(self.session_manager.champion_weights)
        
        # Add variants of champion
        for _ in range(population_size // 4):
            variant = self.generate_random_weights(
                self.session_manager.champion_weights, 0.2
            )
            population.append(variant)
        
        # Fill rest with random weights
        base_ranges = {
            name: (weight * 0.5, weight * 2.0) 
            for name, weight in self.session_manager.champion_weights.items()
        }
        
        while len(population) < population_size:
            weights = {}
            for name, (min_val, max_val) in base_ranges.items():
                weights[name] = random.uniform(min_val, max_val)
            population.append(weights)
        
        generation = 0
        
        while self.session_manager.should_continue():
            generation += 1
            gen_start_time = time.time()
            
            logger.info(f"Generation {generation}: Evaluating population...")
            
            # Evaluate population (in parallel batches to manage memory)
            fitness_scores = []
            for i, individual in enumerate(population):
                result = self.evaluate_weights(individual)
                self.session_manager.add_evaluation(result)
                fitness_scores.append((result.win_rate, i, individual))
                
                if (i + 1) % 5 == 0:  # Progress update every 5 evaluations
                    logger.info(f"  Evaluated {i + 1}/{population_size}")
            
            # Sort by fitness (win rate)
            fitness_scores.sort(reverse=True)
            
            # Log generation results
            gen_time = time.time() - gen_start_time
            best_fitness = fitness_scores[0][0]
            avg_fitness = sum(score for score, _, _ in fitness_scores) / len(fitness_scores)
            
            logger.info(
                f"Gen {generation} complete: Best={best_fitness:.1%}, "
                f"Avg={avg_fitness:.1%} [{gen_time/60:.1f}m]"
            )
            
            # Selection and reproduction
            new_population = []
            
            # Elitism: Keep best individuals
            for i in range(elite_size):
                _, _, individual = fitness_scores[i]
                new_population.append(individual.copy())
            
            # Generate offspring
            while len(new_population) < population_size:
                # Tournament selection
                parent1 = self._tournament_selection(fitness_scores, 3)
                parent2 = self._tournament_selection(fitness_scores, 3)
                
                # Crossover
                if random.random() < crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < mutation_rate:
                    child1 = self._mutate(child1, 0.1)
                if random.random() < mutation_rate:
                    child2 = self._mutate(child2, 0.1)
                
                new_population.extend([child1, child2])
            
            # Trim to exact population size
            population = new_population[:population_size]
        
        logger.info("Genetic Algorithm completed")
        return self.session_manager.best_weights
    
    def _tournament_selection(self, fitness_scores: List[Tuple[float, int, Dict]], 
                             tournament_size: int) -> Dict[str, float]:
        """Tournament selection for genetic algorithm"""
        tournament = random.sample(fitness_scores, min(tournament_size, len(fitness_scores)))
        winner = max(tournament, key=lambda x: x[0])
        return winner[2].copy()
    
    def _crossover(self, parent1: Dict[str, float], 
                  parent2: Dict[str, float]) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Uniform crossover for genetic algorithm"""
        child1, child2 = {}, {}
        
        for name in parent1.keys():
            if random.random() < 0.5:
                child1[name] = parent1[name]
                child2[name] = parent2[name]
            else:
                child1[name] = parent2[name]
                child2[name] = parent1[name]
        
        return child1, child2
    
    def _mutate(self, individual: Dict[str, float], mutation_strength: float) -> Dict[str, float]:
        """Gaussian mutation for genetic algorithm"""
        mutated = {}
        for name, weight in individual.items():
            if random.random() < 0.3:  # 30% chance to mutate each weight
                noise = random.gauss(0, mutation_strength)
                mutated[name] = max(0.1, weight * (1 + noise))
            else:
                mutated[name] = weight
        return mutated
    
    def run_comprehensive_tuning(self):
        """Run the complete comprehensive tuning session"""
        logger = self.session_manager.logger
        logger.info("=" * 80)
        logger.info("STARTING COMPREHENSIVE HYPERPARAMETER TUNING SESSION")
        logger.info("=" * 80)
        
        try:
            # Phase 1: Adaptive Random Search (70% of time)
            phase1_duration = self.config.max_duration_hours * 0.7
            logger.info(f"PHASE 1: Adaptive Random Search ({phase1_duration:.1f}h)")
            
            # Temporarily reduce max duration for phase 1
            original_duration = self.config.max_duration_hours
            self.config.max_duration_hours = phase1_duration
            
            self.adaptive_random_search()
            
            # Phase 2: Genetic Algorithm (30% of time)
            remaining_time = (original_duration * 3600 - (time.time() - self.session_manager.start_time)) / 3600
            if remaining_time > 0.1:  # At least 6 minutes remaining
                logger.info(f"PHASE 2: Genetic Algorithm ({remaining_time:.1f}h)")
                self.config.max_duration_hours = original_duration  # Restore original duration
                
                # Reduce population size if time is limited
                if remaining_time < 1.0:  # Less than 1 hour
                    self.config.population_size = max(8, self.config.population_size // 2)
                    logger.info(f"Reduced population size to {self.config.population_size} due to time constraints")
                
                self.genetic_algorithm()
            
        except KeyboardInterrupt:
            logger.info("Tuning interrupted by user")
        except Exception as e:
            logger.error(f"Error during tuning: {e}")
            raise
        finally:
            # Final save and summary
            self.session_manager.save_progress()
            self._print_final_summary()
            # Cleanup resources
            self.session_manager.cleanup()
    
    def _print_final_summary(self):
        """Print comprehensive final summary"""
        logger = self.session_manager.logger
        elapsed_time = time.time() - self.session_manager.start_time
        
        logger.info("=" * 80)
        logger.info("TUNING SESSION COMPLETED")
        logger.info("=" * 80)
        logger.info(f"Duration: {elapsed_time/3600:.2f} hours")
        logger.info(f"Total evaluations: {len(self.session_manager.evaluations)}")
        logger.info(f"Total games: {self.session_manager.total_games_played}")
        logger.info(f"Games per hour: {self.session_manager.total_games_played/(elapsed_time/3600):.1f}")
        logger.info("")
        logger.info(f"Champion baseline: {self.session_manager.champion_weights}")
        logger.info(f"Best win rate achieved: {self.session_manager.best_win_rate:.1%}")
        logger.info(f"Best weights: {self.session_manager.best_weights}")
        logger.info("")
        logger.info(f"Results saved in: {self.session_manager.session_dir}")
        logger.info("=" * 80)


def run_comprehensive_tuning(
    session_name: str,
    max_hours: float = 4.0,
    games_per_eval: int = 20,
    n_jobs: Optional[int] = None
):
    """
    Run a comprehensive hyperparameter tuning session.
    
    Args:
        session_name: Name for this tuning session
        max_hours: Maximum duration in hours
        games_per_eval: Games to play per weight evaluation
        n_jobs: Number of parallel processes (None = auto)
    """
    config = TuningConfig(
        session_name=session_name,
        max_duration_hours=max_hours,
        games_per_evaluation=games_per_eval,
        n_jobs=n_jobs,
        population_size=16,  # Reasonable for long sessions
        save_interval_minutes=10
    )
    
    tuner = AdvancedHyperparameterTuner(config)
    tuner.run_comprehensive_tuning()
    
    return tuner.session_manager.best_weights


if __name__ == "__main__":
    print("\n🎮 Advanced Blokus AI Hyperparameter Tuning\n")
    
    # Get session parameters
    session_name = input("Session name (default: 'comprehensive'): ").strip()
    if not session_name:
        session_name = "comprehensive"
    
    hours_input = input("Max duration in hours (default: 4.0): ").strip()
    try:
        max_hours = float(hours_input) if hours_input else 4.0
    except ValueError:
        max_hours = 4.0
    
    games_input = input("Games per evaluation (default: 20): ").strip()
    try:
        games_per_eval = int(games_input) if games_input else 20
    except ValueError:
        games_per_eval = 20
    
    print(f"\nStarting comprehensive tuning session:")
    print(f"  Session: {session_name}")
    print(f"  Duration: {max_hours} hours")
    print(f"  Games per evaluation: {games_per_eval}")
    print(f"  Estimated total games: {int(max_hours * 60 * games_per_eval)}")
    
    confirm = input("\nProceed? (y/N): ").strip().lower()
    if confirm == 'y':
        best_weights = run_comprehensive_tuning(
            session_name=session_name,
            max_hours=max_hours,
            games_per_eval=games_per_eval
        )
        
        print("\n✅ Comprehensive tuning complete!")
        print("\nOptimized weights:")
        print(json.dumps(best_weights, indent=2))
    else:
        print("Tuning cancelled.")