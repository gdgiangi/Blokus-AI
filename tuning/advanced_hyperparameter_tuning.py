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

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from game_state import GameState
from board import PlayerColor
from ai_player_enhanced import OptimizedAIStrategy, AIPlayer, EnhancedHeuristic, AggressiveOptimizedStrategy, BalancedOptimizedStrategy, DefensiveOptimizedStrategy, RandomAIStrategy


# Fixed diverse strategy weights for robust evaluation against different playstyles
DIVERSE_STRATEGY_WEIGHTS = {
    "aggressive": {
        "piece_size": 0.8,
        "blocked_opponents": 4.5,
        "corner_control": 2.5,
        "compactness": 0.5,
        "mobility": 0.4,
        "opponent_restriction": 3.5,
        "endgame_optimization": 0.8,
        "corner_path_potential": 2.0,
        "opponent_territory_pressure": 2.5,
        "opponent_mobility_restriction": 3.0,
        "opponent_threat_assessment": 1.5,
        "strategic_positioning": 1.0
    },
    "balanced": {
        "piece_size": 1.5,
        "blocked_opponents": 1.8,
        "corner_control": 1.8,
        "compactness": 1.2,
        "mobility": 1.5,
        "opponent_restriction": 1.2,
        "endgame_optimization": 1.8,
        "corner_path_potential": 2.5,
        "opponent_territory_pressure": 1.0,
        "opponent_mobility_restriction": 1.0,
        "opponent_threat_assessment": 1.2,
        "strategic_positioning": 1.5
    },
    "defensive": {
        "piece_size": 2.0,
        "blocked_opponents": 0.8,
        "corner_control": 1.2,
        "compactness": 2.0,
        "mobility": 2.5,
        "opponent_restriction": 0.6,
        "endgame_optimization": 2.5,
        "corner_path_potential": 4.0,
        "opponent_territory_pressure": 0.5,
        "opponent_mobility_restriction": 0.8,
        "opponent_threat_assessment": 2.0,
        "strategic_positioning": 2.0
    }
}


@dataclass
class TuningConfig:
    """Configuration for hyperparameter tuning session"""
    session_name: str
    max_duration_hours: float
    games_per_evaluation: int
    champion_weight_variation: float = 0.15  # ±15% variation for champion variants
    champion_pool_size: int = 5  # Size of champion pool to maintain
    population_size: int = 20  # For genetic algorithm
    mutation_rate: float = 0.1
    crossover_rate: float = 0.7
    elite_ratio: float = 0.2
    save_interval_minutes: int = 10
    verbose_logging: bool = True
    n_jobs: Optional[int] = None
    
    # NEW: Random AI injection parameters
    random_opponent_frequency: float = 0.3  # 30% of games include random AI
    random_replacement_strategy: str = "defensive"  # Which fixed strategy to replace with random
    
    # NEW: Enhanced champion battle parameters
    champion_battle_frequency: int = 10  # Run battles every N evaluations
    champion_battle_weight: float = 0.3  # Weight of battle results in final ranking
    
    # Enhanced global optimization parameters
    convergence_patience: int = 50  # Generations without improvement before convergence
    convergence_threshold: float = 0.001  # Minimum improvement to be considered significant
    simulated_annealing_temp: float = 2.0  # Initial temperature for SA
    simulated_annealing_cooling: float = 0.95  # Cooling rate for SA
    pso_particles: int = 15  # Particle swarm size
    pso_inertia: float = 0.7  # PSO inertia weight
    pso_cognitive: float = 1.5  # PSO cognitive parameter
    pso_social: float = 1.5  # PSO social parameter
    bayesian_samples: int = 10  # Number of Bayesian optimization samples per iteration
    search_space_bounds: Optional[Dict[str, Tuple[float, float]]] = None  # Custom bounds for each heuristic
    
    # Enhanced evaluation parameters
    high_performer_threshold: float = 0.90  # Win rate threshold for high performers
    high_performer_games: int = 150  # Increased sample size for high performers
    enable_champion_battles: bool = True  # Enable champion-vs-champion evaluation
    champion_battle_games: int = 100  # Games for champion battles
    use_adaptive_opponents: bool = True  # Use champion pool as adaptive opponents


@dataclass
class GameResult:
    """Result of a single game against diverse strategies"""
    contender_score: int
    aggressive_score: int  # Yellow player (aggressive strategy)
    balanced_score: int    # Red player (balanced strategy)
    defensive_score: int   # Green player (defensive strategy)
    winner: Optional[str]  # PlayerColor as string for JSON serialization
    game_length: int
    timestamp: float


@dataclass
class ChampionEntry:
    """Entry in the champion pool"""
    weights: Dict[str, float]
    win_rate: float
    avg_score: float
    games_played: int
    timestamp: float
    
    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class EvaluationResult:
    """Result of evaluating a weight configuration"""
    weights: Dict[str, float]
    games: List[GameResult]
    win_rate: float
    avg_score: float
    avg_score_diff: float  # vs champion pool average
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
        self.champion_pool = self._load_champion_pool()
        self.champion_variants = self._generate_champion_variants()
        
        # Statistics tracking
        self.total_games_played = 0
        self.total_evaluation_time = 0
        self.win_rate_history = deque(maxlen=100)  # Last 100 evaluations
        
        # Champion battle tracking
        self.evaluations_since_last_battle = 0
        self.total_battles_conducted = 0
        
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
                sys.stdout.reconfigure(encoding='utf-8', errors='replace')  # type: ignore
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
    
    def _load_champion_pool(self) -> List[ChampionEntry]:
        """
        Initialize champion pool with diverse strategic archetypes.
        This ensures adaptive opponent mode activates from the start (needs >=3 champions).
        
        Pool includes:
        1. Uniform Baseline - Equal weights for balanced exploration
        2. Aggressive Archetype - High opponent disruption
        3. Balanced Archetype - Moderate across all heuristics
        4. Defensive Archetype - High mobility and self-preservation
        5. UI Baseline - Matches UI's Balanced strategy for real-world validation
        """
        current_champions = [
            # Champion 1: Latest Optimized Champion (60% win rate from GeniusTime session)
            {
                "name": "Latest Champion (GeniusTime 60%)",
                "weights": {
                    "piece_size": 0.70,
                    "blocked_opponents": 4.90,
                    "corner_control": 2.50,
                    "compactness": 0.49,
                    "mobility": 0.35,
                    "opponent_restriction": 3.81,
                    "endgame_optimization": 0.78,
                    "corner_path_potential": 1.90,
                    "opponent_territory_pressure": 2.42,
                    "opponent_mobility_restriction": 3.44,
                    "opponent_threat_assessment": 1.59,
                    "strategic_positioning": 0.86
                },
                "win_rate": 0.60,  # Actual performance from previous run
                "avg_score": 61.68
            },
            # Champion 2: Aggressive Archetype (from tuning fixed opponents)
            {
                "name": "Aggressive Archetype",
                "weights": DIVERSE_STRATEGY_WEIGHTS["aggressive"],
                "win_rate": 0.50,
                "avg_score": 60.0
            },
            # Champion 3: Balanced Archetype (from tuning fixed opponents)
            {
                "name": "Balanced Archetype",
                "weights": DIVERSE_STRATEGY_WEIGHTS["balanced"],
                "win_rate": 0.50,
                "avg_score": 60.0
            },
            # Champion 4: Defensive Archetype (from tuning fixed opponents)
            {
                "name": "Defensive Archetype",
                "weights": DIVERSE_STRATEGY_WEIGHTS["defensive"],
                "win_rate": 0.50,
                "avg_score": 60.0
            },
            # Champion 5: UI Balanced Strategy (matches real UI opponents)
            {
                "name": "UI Balanced Strategy",
                "weights": {
                    "piece_size": 1.5,
                    "blocked_opponents": 1.8,
                    "corner_control": 1.8,
                    "compactness": 1.2,
                    "mobility": 1.5,
                    "opponent_restriction": 1.2,
                    "endgame_optimization": 1.8,
                    "corner_path_potential": 2.5,
                    "opponent_territory_pressure": 1.2,  # UI version (stronger)
                    "opponent_mobility_restriction": 1.5,  # UI version (stronger)
                    "opponent_threat_assessment": 1.5,  # UI version (stronger)
                    "strategic_positioning": 2.0
                },
                "win_rate": 0.50,
                "avg_score": 60.0
            }
        ]
        
        # Initialize champion pool with all 5 diverse champions
        champion_pool = []
        
        for i, champ_data in enumerate(current_champions[:self.config.champion_pool_size]):
            champion_pool.append(ChampionEntry(
                weights=champ_data["weights"],
                win_rate=champ_data["win_rate"],
                avg_score=champ_data["avg_score"],
                games_played=0,
                timestamp=time.time() - (len(current_champions) - i)  # Slight time diff for ordering
            ))
        
        self.logger.info(f"Initialized diverse champion pool with {len(champion_pool)} strategic archetypes:")
        for i, (champ, champ_data) in enumerate(zip(champion_pool, current_champions), 1):
            win_rate_info = f" ({champ_data.get('win_rate', 0.5):.0%} win rate)" if 'win_rate' in champ_data else ""
            self.logger.info(
                f"  {i}. {champ_data['name']}{win_rate_info}"
            )
        
        return champion_pool
    
    def _generate_champion_variants(self) -> List[Dict[str, float]]:
        """Generate variants from the champion pool for competition"""
        variants = []
        
        # Use the top 2 champions from pool as variants
        for i in range(min(2, len(self.champion_pool))):
            champion = self.champion_pool[i]
            variant = {}
            for name, weight in champion.weights.items():
                variation = random.uniform(-self.config.champion_weight_variation, 
                                         self.config.champion_weight_variation)
                variant[name] = max(0.1, weight * (1 + variation))
            variants.append(variant)
        
        # If we need more variants, generate them from the best champion
        while len(variants) < 2:
            best_champion = self.champion_pool[0]
            variant = {}
            for name, weight in best_champion.weights.items():
                variation = random.uniform(-self.config.champion_weight_variation, 
                                         self.config.champion_weight_variation)
                variant[name] = max(0.1, weight * (1 + variation))
            variants.append(variant)
        
        self.logger.info("Generated champion variants from pool:")
        for i, variant in enumerate(variants):
            self.logger.info(f"  Variant {i+1}: {self._format_weights(variant)}")
        
        return variants
    
    def get_current_champions(self) -> List[Dict[str, float]]:
        """Get weights from champion pool for evaluation"""
        return [champion.weights for champion in self.champion_pool]
    
    def update_champion_pool(self, contender_weights: Dict[str, float], 
                           win_rate: float, avg_score: float, games_played: int):
        """
        Update champion pool with a strong contender, replacing weakest if better.
        Uses multi-criteria comparison with tie-breaking:
        1. Win rate (primary)
        2. Average score (tie-breaker)
        3. Timestamp (final tie-breaker - newer is better)
        """
        # Find the weakest champion using multi-criteria comparison
        def champion_strength(champion: ChampionEntry) -> Tuple[float, float, float]:
            """Return tuple for comparison: (win_rate, avg_score, -timestamp)"""
            return (champion.win_rate, champion.avg_score, -champion.timestamp)
        
        weakest_idx = min(range(len(self.champion_pool)), 
                         key=lambda i: champion_strength(self.champion_pool[i]))
        weakest_champion = self.champion_pool[weakest_idx]
        
        # Compare contender to weakest using same multi-criteria
        contender_strength = (win_rate, avg_score, -time.time())
        weakest_strength = champion_strength(weakest_champion)
        
        # If contender is better than weakest champion, replace it
        if contender_strength > weakest_strength:
            new_champion = ChampionEntry(
                weights=contender_weights.copy(),
                win_rate=win_rate,
                avg_score=avg_score,
                games_played=games_played,
                timestamp=time.time()
            )
            
            # Enhanced logging with tie-breaking details
            if win_rate == weakest_champion.win_rate:
                self.logger.info(
                    f"🏆 NEW CHAMPION (tie-break)! Replacing pool position {weakest_idx + 1} "
                    f"(same {win_rate:.1%} win rate, but avg_score {avg_score:.1f} > {weakest_champion.avg_score:.1f})"
                )
            else:
                self.logger.info(
                    f"🏆 NEW CHAMPION! Replacing pool position {weakest_idx + 1} "
                    f"(was {weakest_champion.win_rate:.1%}, now {win_rate:.1%})"
                )
            
            self.champion_pool[weakest_idx] = new_champion
            
            # Re-sort pool by multi-criteria (best first)
            self.champion_pool.sort(key=champion_strength, reverse=True)
            
            # Regenerate variants after pool update
            self.champion_variants = self._generate_champion_variants()
            
            return True
        
        return False
    
    def get_champion_pool_stats(self) -> Dict[str, Any]:
        """Get statistics about the champion pool"""
        if not self.champion_pool:
            return {}
        
        win_rates = [c.win_rate for c in self.champion_pool]
        avg_scores = [c.avg_score for c in self.champion_pool]
        
        return {
            "pool_size": len(self.champion_pool),
            "best_win_rate": max(win_rates),
            "worst_win_rate": min(win_rates),
            "avg_win_rate": sum(win_rates) / len(win_rates),
            "best_avg_score": max(avg_scores),
            "worst_avg_score": min(avg_scores),
            "pool_avg_score": sum(avg_scores) / len(avg_scores)
        }
    
    def log_champion_pool_status(self):
        """Log current champion pool status"""
        self.logger.info("=== CHAMPION POOL STATUS ===")
        pool_stats = self.get_champion_pool_stats()
        self.logger.info(f"Pool range: {pool_stats.get('worst_win_rate', 0):.1%} - {pool_stats.get('best_win_rate', 0):.1%}")
        self.logger.info(f"Pool average: {pool_stats.get('avg_win_rate', 0):.1%}")
        
        self.logger.info("Current Champions:")
        for i, champion in enumerate(self.champion_pool, 1):
            self.logger.info(f"  {i}. {champion.win_rate:.1%} - {self._format_weights(champion.weights)}")
        self.logger.info("=" * 40)
    
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
        """Add a new evaluation result and check if champion battles should be triggered"""
        self.evaluations.append(result)
        self.total_games_played += len(result.games)
        self.total_evaluation_time += result.evaluation_time
        self.win_rate_history.append(result.win_rate)
        self.evaluations_since_last_battle += 1
        
        if result.win_rate > self.best_win_rate:
            self.best_win_rate = result.win_rate
            self.best_weights = result.weights.copy()
            self.logger.info(f"** NEW BEST OVERALL! ** Win rate: {result.win_rate:.1%} - {self._format_weights(result.weights)}")
        
        # Auto-save if needed
        if self.should_save():
            self.save_progress()
    
    def should_conduct_champion_battles(self) -> bool:
        """Check if champion battles should be conducted"""
        return (
            len(self.champion_pool) >= 4 and  # Need at least 4 champions
            self.evaluations_since_last_battle >= self.config.champion_battle_frequency
        )
    
    def record_champion_battle(self):
        """Record that a champion battle was conducted"""
        self.evaluations_since_last_battle = 0
        self.total_battles_conducted += 1
        self.logger.info(f"Champion battle #{self.total_battles_conducted} completed")
    
    def save_progress(self):
        """Save current progress to files"""
        self.last_save_time = time.time()
        
        # Main results file
        results_file = self.session_dir / "results.json"
        data = {
            "config": asdict(self.config),
            "start_time": self.start_time,
            "champion_pool": [champion.to_dict() for champion in self.champion_pool],
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
                
            # Champion pool information
            f.write(f"=== Champion Pool ===\n")
            pool_stats = self.get_champion_pool_stats()
            f.write(f"Pool size: {pool_stats.get('pool_size', 0)}\n")
            f.write(f"Best pool win rate: {pool_stats.get('best_win_rate', 0):.1%}\n")
            f.write(f"Pool average win rate: {pool_stats.get('avg_win_rate', 0):.1%}\n")
            
            f.write(f"\nChampion Pool Details:\n")
            for i, champion in enumerate(self.champion_pool, 1):
                f.write(f"  {i}. {champion.win_rate:.1%} - {self._format_weights(champion.weights)}\n")
            
            f.write(f"\nBest overall weights: {self.best_weights}\n\n")
            
            # Top 10 results
            top_results = sorted(self.evaluations, key=lambda x: x.win_rate, reverse=True)[:10]
            f.write(f"=== Top 10 Results ===\n")
            for i, result in enumerate(top_results, 1):
                f.write(f"{i:2d}. {result.win_rate:.1%} - {self._format_weights(result.weights)}\n")


# Global helper function for multiprocessing
def _play_diverse_strategy_game(game_config: Tuple[Dict[str, float], int, float, str]) -> GameResult:
    """
    Play a single game: Contender vs 3 Diverse Strategies (Aggressive, Balanced, Defensive)
    Optionally replaces one strategy with a random AI for chaos injection.
    Returns detailed game result.
    """
    contender_weights, game_id, random_frequency, replacement_strategy = game_config
    
    game = GameState(num_players=4)
    game.start_game()
    
    # Determine if this game should include a random player
    use_random = random.random() < random_frequency
    
    # Create AI players with diverse strategies for robust evaluation
    # Blue = Contender (being tested)
    # Yellow = Aggressive Strategy (high blocking, territorial control)
    # Red = Balanced Strategy (moderate across all heuristics)  
    # Green = Defensive Strategy (high mobility, path potential, self-preservation) OR Random AI
    
    if use_random and replacement_strategy == "defensive":
        # Replace defensive strategy with random AI
        ai_players = {
            PlayerColor.BLUE: AIPlayer(PlayerColor.BLUE, OptimizedAIStrategy(contender_weights)),
            PlayerColor.YELLOW: AIPlayer(PlayerColor.YELLOW, OptimizedAIStrategy(DIVERSE_STRATEGY_WEIGHTS["aggressive"])),
            PlayerColor.RED: AIPlayer(PlayerColor.RED, OptimizedAIStrategy(DIVERSE_STRATEGY_WEIGHTS["balanced"])),
            PlayerColor.GREEN: AIPlayer(PlayerColor.GREEN, RandomAIStrategy())  # RANDOM AI INJECTION
        }
    elif use_random and replacement_strategy == "aggressive":
        # Replace aggressive strategy with random AI
        ai_players = {
            PlayerColor.BLUE: AIPlayer(PlayerColor.BLUE, OptimizedAIStrategy(contender_weights)),
            PlayerColor.YELLOW: AIPlayer(PlayerColor.YELLOW, RandomAIStrategy()),  # RANDOM AI INJECTION
            PlayerColor.RED: AIPlayer(PlayerColor.RED, OptimizedAIStrategy(DIVERSE_STRATEGY_WEIGHTS["balanced"])),
            PlayerColor.GREEN: AIPlayer(PlayerColor.GREEN, OptimizedAIStrategy(DIVERSE_STRATEGY_WEIGHTS["defensive"]))
        }
    elif use_random and replacement_strategy == "balanced":
        # Replace balanced strategy with random AI
        ai_players = {
            PlayerColor.BLUE: AIPlayer(PlayerColor.BLUE, OptimizedAIStrategy(contender_weights)),
            PlayerColor.YELLOW: AIPlayer(PlayerColor.YELLOW, OptimizedAIStrategy(DIVERSE_STRATEGY_WEIGHTS["aggressive"])),
            PlayerColor.RED: AIPlayer(PlayerColor.RED, RandomAIStrategy()),  # RANDOM AI INJECTION
            PlayerColor.GREEN: AIPlayer(PlayerColor.GREEN, OptimizedAIStrategy(DIVERSE_STRATEGY_WEIGHTS["defensive"]))
        }
    else:
        # Standard diverse strategies (no random AI)
        ai_players = {
            PlayerColor.BLUE: AIPlayer(PlayerColor.BLUE, OptimizedAIStrategy(contender_weights)),
            PlayerColor.YELLOW: AIPlayer(PlayerColor.YELLOW, OptimizedAIStrategy(DIVERSE_STRATEGY_WEIGHTS["aggressive"])),
            PlayerColor.RED: AIPlayer(PlayerColor.RED, OptimizedAIStrategy(DIVERSE_STRATEGY_WEIGHTS["balanced"])),
            PlayerColor.GREEN: AIPlayer(PlayerColor.GREEN, OptimizedAIStrategy(DIVERSE_STRATEGY_WEIGHTS["defensive"]))
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
        aggressive_score=scores[PlayerColor.YELLOW],
        balanced_score=scores[PlayerColor.RED],
        defensive_score=scores[PlayerColor.GREEN],
        winner=winner.value,  # Convert PlayerColor to string
        game_length=move_count,
        timestamp=time.time()
    )


def _play_adaptive_opponent_game(game_config: Tuple[Dict[str, float], List[Dict[str, float]], int]) -> GameResult:
    """
    Play a single game: Contender vs 3 Adaptive Opponents from Champion Pool
    This tests against the current best strategies, providing evolving challenge.
    Returns detailed game result.
    """
    contender_weights, champion_weights_list, game_id = game_config
    
    # Select 3 champions for this game (cycle through if needed)
    num_champions = len(champion_weights_list)
    if num_champions >= 3:
        # Use different champions for variety
        idx = game_id % max(1, num_champions - 2)
        opponent_weights = [
            champion_weights_list[idx % num_champions],
            champion_weights_list[(idx + 1) % num_champions],
            champion_weights_list[(idx + 2) % num_champions]
        ]
    else:
        # Pad with diverse strategies if not enough champions
        opponent_weights = champion_weights_list.copy()
        while len(opponent_weights) < 3:
            strategies = ["aggressive", "balanced", "defensive"]
            opponent_weights.append(DIVERSE_STRATEGY_WEIGHTS[strategies[len(opponent_weights) % 3]])
    
    game = GameState(num_players=4)
    game.start_game()
    
    # Create AI players with adaptive opponents
    ai_players = {
        PlayerColor.BLUE: AIPlayer(PlayerColor.BLUE, OptimizedAIStrategy(contender_weights)),
        PlayerColor.YELLOW: AIPlayer(PlayerColor.YELLOW, OptimizedAIStrategy(opponent_weights[0])),
        PlayerColor.RED: AIPlayer(PlayerColor.RED, OptimizedAIStrategy(opponent_weights[1])),
        PlayerColor.GREEN: AIPlayer(PlayerColor.GREEN, OptimizedAIStrategy(opponent_weights[2]))
    }
    
    move_count = 0
    max_moves = 300
    
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
    winner = rankings[0][0] if rankings else PlayerColor.BLUE
    
    return GameResult(
        contender_score=scores[PlayerColor.BLUE],
        aggressive_score=scores[PlayerColor.YELLOW],
        balanced_score=scores[PlayerColor.RED],
        defensive_score=scores[PlayerColor.GREEN],
        winner=winner.value,
        game_length=move_count,
        timestamp=time.time()
    )


def _play_champion_battle_game(game_config: Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float], int]) -> GameResult:
    """
    Play a champion-vs-champion battle: 4 champions compete head-to-head.
    This determines superiority among high-performing strategies.
    Returns detailed game result.
    """
    champ1_weights, champ2_weights, champ3_weights, champ4_weights, game_id = game_config
    
    game = GameState(num_players=4)
    game.start_game()
    
    # All players are champions
    ai_players = {
        PlayerColor.BLUE: AIPlayer(PlayerColor.BLUE, OptimizedAIStrategy(champ1_weights)),
        PlayerColor.YELLOW: AIPlayer(PlayerColor.YELLOW, OptimizedAIStrategy(champ2_weights)),
        PlayerColor.RED: AIPlayer(PlayerColor.RED, OptimizedAIStrategy(champ3_weights)),
        PlayerColor.GREEN: AIPlayer(PlayerColor.GREEN, OptimizedAIStrategy(champ4_weights))
    }
    
    move_count = 0
    max_moves = 300
    
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
    winner = rankings[0][0] if rankings else PlayerColor.BLUE
    
    return GameResult(
        contender_score=scores[PlayerColor.BLUE],
        aggressive_score=scores[PlayerColor.YELLOW],
        balanced_score=scores[PlayerColor.RED],
        defensive_score=scores[PlayerColor.GREEN],
        winner=winner.value,
        game_length=move_count,
        timestamp=time.time()
    )


class AdvancedHyperparameterTuner:
    """
    Advanced hyperparameter tuner with diverse strategy evaluation.
    
    Tests contenders against 3 fixed diverse strategies (Aggressive, Balanced, Defensive)
    for robust evaluation against fundamentally different playstyles, ensuring globally
    optimal strategies rather than ones that only work against similar opponents.
    """
    
    def __init__(self, config: TuningConfig):
        self.config = config
        self.session_manager = SessionManager(config)
        
        # Determine number of processes
        if config.n_jobs is None or config.n_jobs == -1:
            self.n_jobs = max(1, cpu_count() - 1)  # Leave one CPU free
        else:
            self.n_jobs = max(1, config.n_jobs)
        
        self.session_manager.logger.info(f"Using {self.n_jobs} parallel processes")
    
    def evaluate_weights(self, weights: Dict[str, float], is_reevaluation: bool = False) -> EvaluationResult:
        """
        Evaluate weights with adaptive sample size and opponent selection.
        
        Args:
            weights: Weights to evaluate
            is_reevaluation: If True, this is a high-performer re-evaluation
        """
        start_time = time.time()
        
        # Determine number of games based on context
        if is_reevaluation:
            num_games = self.config.high_performer_games
            self.session_manager.logger.info(
                f"🔍 High-performer re-evaluation: {num_games} games for accuracy"
            )
        else:
            num_games = self.config.games_per_evaluation
        
        # Determine opponent type
        use_adaptive = self.config.use_adaptive_opponents and len(self.session_manager.champion_pool) >= 3
        
        if use_adaptive:
            # Use champion pool as adaptive opponents
            champion_weights = [c.weights for c in self.session_manager.champion_pool[:5]]
            game_configs = []
            for game_id in range(num_games):
                config = (weights, champion_weights, game_id)
                game_configs.append(config)
            
            # Play games in parallel
            if self.n_jobs > 1:
                with Pool(processes=self.n_jobs) as pool:
                    game_results = pool.map(_play_adaptive_opponent_game, game_configs)
            else:
                game_results = [_play_adaptive_opponent_game(config) for config in game_configs]
            
            opponent_type = "adaptive_champions"
        else:
            # Use fixed diverse strategies (with optional random AI injection)
            game_configs = []
            for game_id in range(num_games):
                config = (weights, game_id, self.config.random_opponent_frequency, self.config.random_replacement_strategy)
                game_configs.append(config)
            
            # Play games in parallel
            if self.n_jobs > 1:
                with Pool(processes=self.n_jobs) as pool:
                    game_results = pool.map(_play_diverse_strategy_game, game_configs)
            else:
                game_results = [_play_diverse_strategy_game(config) for config in game_configs]
            
            # Determine opponent type based on random injection
            if self.config.random_opponent_frequency > 0:
                opponent_type = f"diverse_fixed+random({self.config.random_opponent_frequency:.0%})"
            else:
                opponent_type = "diverse_fixed"
        
        # Calculate statistics
        wins = sum(1 for result in game_results if result.winner == "blue")
        win_rate = wins / len(game_results)
        avg_score = sum(result.contender_score for result in game_results) / len(game_results)
        
        # Calculate average opponent score
        avg_opponent_scores = []
        for result in game_results:
            opponent_avg = (result.aggressive_score + result.balanced_score + result.defensive_score) / 3
            avg_opponent_scores.append(opponent_avg)
        avg_opponent_score = sum(avg_opponent_scores) / len(avg_opponent_scores)
        avg_score_diff = avg_score - avg_opponent_score
        
        evaluation_time = time.time() - start_time
        
        result = EvaluationResult(
            weights=weights.copy(),
            games=game_results,
            win_rate=win_rate,
            avg_score=avg_score,
            avg_score_diff=avg_score_diff,
            evaluation_time=evaluation_time,
            timestamp=time.time()
        )
        
        # Log evaluation details
        self.session_manager.logger.info(
            f"Evaluated: {win_rate:.1%} win rate, {avg_score:.1f} avg score "
            f"({num_games} games vs {opponent_type})"
        )
        
        # Check if this is a high performer that needs re-evaluation
        if not is_reevaluation and win_rate >= self.config.high_performer_threshold:
            self.session_manager.logger.info(
                f"⭐ HIGH PERFORMER DETECTED! Win rate {win_rate:.1%} >= {self.config.high_performer_threshold:.1%}"
            )
            self.session_manager.logger.info(
                f"Re-evaluating with {self.config.high_performer_games} games for statistical confidence..."
            )
            # Re-evaluate with more games
            result = self.evaluate_weights(weights, is_reevaluation=True)
        
        # Check if this contender should join the champion pool
        self.session_manager.update_champion_pool(
            weights, result.win_rate, result.avg_score, len(result.games)
        )
        
        # Check if champion battles should be conducted
        if self.session_manager.should_conduct_champion_battles():
            self.session_manager.logger.info(
                f"⚔️ Triggering champion battles (every {self.config.champion_battle_frequency} evaluations)"
            )
            battle_results = self.conduct_champion_battles()
            self.session_manager.record_champion_battle()
        
        return result
    
    def conduct_champion_battles(self) -> Dict[str, Any]:
        """
        Conduct head-to-head battles among champions to refine rankings.
        Returns battle results and updated champion statistics.
        """
        if not self.config.enable_champion_battles:
            return {}
        
        if len(self.session_manager.champion_pool) < 4:
            self.session_manager.logger.info(
                "⚔️ Champion battles require at least 4 champions. Skipping for now."
            )
            return {}
        
        logger = self.session_manager.logger
        logger.info("=" * 80)
        logger.info("⚔️ CHAMPION BATTLE ARENA - Head-to-Head Competition")
        logger.info("=" * 80)
        
        # Take top 4 champions for battle
        battle_champions = self.session_manager.champion_pool[:4]
        logger.info(f"Competitors ({len(battle_champions)} champions):")
        for i, champ in enumerate(battle_champions, 1):
            logger.info(
                f"  {i}. Win Rate: {champ.win_rate:.1%}, Avg Score: {champ.avg_score:.1f}"
            )
        
        # Prepare battle configurations - rotate champion positions for fairness
        num_games = self.config.champion_battle_games
        game_configs = []
        
        for game_id in range(num_games):
            # Rotate starting positions to eliminate position bias
            rotation = game_id % 4
            rotated_champions = [
                battle_champions[(0 + rotation) % 4].weights,
                battle_champions[(1 + rotation) % 4].weights,
                battle_champions[(2 + rotation) % 4].weights,
                battle_champions[(3 + rotation) % 4].weights
            ]
            config = (*rotated_champions, game_id)
            game_configs.append(config)
        
        logger.info(f"Running {num_games} battle games with position rotation for fairness...")
        start_time = time.time()
        
        # Play battle games in parallel
        if self.n_jobs > 1:
            with Pool(processes=self.n_jobs) as pool:
                battle_results = pool.map(_play_champion_battle_game, game_configs)
        else:
            battle_results = [_play_champion_battle_game(config) for config in game_configs]
        
        battle_time = time.time() - start_time
        
        # Analyze results by tracking wins for each champion across all positions
        champion_stats = {i: {"wins": 0, "total_score": 0, "games": 0} for i in range(4)}
        
        color_to_position = {
            "blue": 0, "yellow": 1, "red": 2, "green": 3
        }
        
        for game_id, result in enumerate(battle_results):
            rotation = game_id % 4
            # Determine which champion was in which position
            winner_color = result.winner
            if winner_color is None:
                winner_color = "blue"  # Default fallback
            winner_position = color_to_position[winner_color]
            # Reverse rotation to find original champion index
            original_champion_idx = (winner_position - rotation) % 4
            champion_stats[original_champion_idx]["wins"] += 1
            
            # Track scores for each champion based on their rotated position
            scores_by_color = {
                "blue": result.contender_score,
                "yellow": result.aggressive_score,
                "red": result.balanced_score,
                "green": result.defensive_score
            }
            
            for color, score in scores_by_color.items():
                position = color_to_position[color]
                original_idx = (position - rotation) % 4
                champion_stats[original_idx]["total_score"] += score
                champion_stats[original_idx]["games"] += 1
        
        # Calculate final statistics
        logger.info("\n" + "=" * 80)
        logger.info("📊 BATTLE RESULTS")
        logger.info("=" * 80)
        
        for i in range(4):
            stats = champion_stats[i]
            battle_win_rate = stats["wins"] / num_games if num_games > 0 else 0
            battle_avg_score = stats["total_score"] / stats["games"] if stats["games"] > 0 else 0
            original_champ = battle_champions[i]
            
            logger.info(
                f"Champion {i+1}: {stats['wins']}/{num_games} wins ({battle_win_rate:.1%}), "
                f"Avg Score: {battle_avg_score:.1f} "
                f"[Original: {original_champ.win_rate:.1%} vs fixed opponents]"
            )
            
            # Update champion entry with battle performance using weighted average
            # Blend original win rate with battle win rate
            original_weight = 1.0 - self.config.champion_battle_weight
            battle_weight = self.config.champion_battle_weight
            
            new_win_rate = (
                original_weight * battle_champions[i].win_rate + 
                battle_weight * battle_win_rate
            )
            
            battle_champions[i].win_rate = new_win_rate
            battle_champions[i].avg_score = (
                0.7 * battle_champions[i].avg_score + 0.3 * battle_avg_score
            )
        
        # Re-sort champion pool based on updated statistics
        self.session_manager.champion_pool.sort(
            key=lambda c: (c.win_rate, c.avg_score, -c.timestamp), 
            reverse=True
        )
        
        logger.info("\n" + "=" * 80)
        logger.info(f"⚔️ Battle completed in {battle_time/60:.1f} minutes")
        logger.info("Champion pool rankings updated based on head-to-head performance")
        logger.info("=" * 80 + "\n")
        
        return {
            "num_games": num_games,
            "battle_time": battle_time,
            "champion_stats": champion_stats,
            "updated_rankings": [
                {"win_rate": c.win_rate, "avg_score": c.avg_score} 
                for c in self.session_manager.champion_pool[:4]
            ]
        }
    
    def get_evaluation_info(self) -> Dict[str, Any]:
        """Get information about the evaluation methodology"""
        return {
            "evaluation_method": "diverse_strategy_opponents",
            "description": "Contenders are evaluated against 3 fixed diverse strategies for robust testing",
            "opponent_strategies": {
                "aggressive": "High opponent blocking and territorial control",
                "balanced": "Even emphasis across all strategic heuristics", 
                "defensive": "High mobility, path potential, and self-preservation"
            },
            "advantages": [
                "Tests robustness against fundamentally different playstyles",
                "Provides consistent evaluation benchmarks", 
                "Prevents overfitting to similar high-performing strategies",
                "Ensures comprehensive strategic coverage"
            ]
        }
    
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
        
        # Start with best champion from pool as baseline
        base_weights = self.session_manager.champion_pool[0].weights.copy()
        current_variation = 0.25  # Start with 25% variation
        
        # Evaluate baseline
        logger.info("Evaluating champion pool baseline...")
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
            
            # Log champion pool status every 20 iterations
            if iteration % 20 == 0:
                self.session_manager.log_champion_pool_status()
            
            # Check for long stagnation and restart with different base
            time_since_improvement = time.time() - last_improvement_time
            if time_since_improvement > 1800:  # 30 minutes without improvement
                logger.info("Long stagnation detected, restarting from random champion in pool")
                # Pick a random champion from pool as new base
                random_champion_idx = random.randint(0, len(self.session_manager.champion_pool) - 1)
                base_weights = self.generate_random_weights(
                    self.session_manager.champion_pool[random_champion_idx].weights, 0.5
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
        
        # Add all champions from pool as seeds
        for champion in self.session_manager.champion_pool:
            population.append(champion.weights.copy())
        
        # Add variants of champions
        champions_to_vary = min(population_size // 4, len(self.session_manager.champion_pool))
        for i in range(champions_to_vary):
            champion_weights = self.session_manager.champion_pool[i % len(self.session_manager.champion_pool)].weights
            variant = self.generate_random_weights(champion_weights, 0.2)
            population.append(variant)
        
        # Fill rest with random weights based on best champion
        best_champion_weights = self.session_manager.champion_pool[0].weights
        base_ranges = {
            name: (weight * 0.5, weight * 2.0) 
            for name, weight in best_champion_weights.items()
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
    
    def simulated_annealing(self, initial_solution: Optional[Dict[str, float]] = None) -> Dict[str, float]:
        """
        Simulated Annealing optimization for fine-tuning
        Excellent for escaping local optima in the late stages of optimization
        """
        logger = self.session_manager.logger
        logger.info("Starting Simulated Annealing optimization")
        
        # Use best champion as starting point or provided initial solution
        if initial_solution is None:
            current_solution = self.session_manager.champion_pool[0].weights.copy()
        else:
            current_solution = initial_solution.copy()
        
        current_result = self.evaluate_weights(current_solution)
        current_fitness = current_result.win_rate + (current_result.avg_score_diff / 100.0)  # Combined fitness
        
        temperature = self.config.simulated_annealing_temp
        best_solution = current_solution.copy()
        best_fitness = current_fitness
        
        iteration = 0
        stagnation_count = 0
        
        while (time.time() - self.session_manager.start_time < self.config.max_duration_hours * 3600 and
               temperature > 0.01):  # Minimum temperature threshold
            
            # Generate neighbor solution by perturbing current solution
            neighbor_solution = {}
            for name, weight in current_solution.items():
                # Temperature-scaled perturbation
                perturbation = random.gauss(0, temperature * 0.1)  # Scale with temperature
                neighbor_solution[name] = max(0.1, weight * (1 + perturbation))
            
            # Evaluate neighbor
            neighbor_result = self.evaluate_weights(neighbor_solution)
            neighbor_fitness = neighbor_result.win_rate + (neighbor_result.avg_score_diff / 100.0)
            
            # Accept or reject based on Metropolis criterion
            delta = neighbor_fitness - current_fitness
            
            if delta > 0 or random.random() < math.exp(delta / temperature):
                current_solution = neighbor_solution
                current_fitness = neighbor_fitness
                
                # Update global best
                if neighbor_fitness > best_fitness:
                    best_solution = neighbor_solution.copy()
                    best_fitness = neighbor_fitness
                    stagnation_count = 0
                    logger.info(f"SA Iteration {iteration}: New best fitness {best_fitness:.4f}, temp={temperature:.3f}")
                else:
                    stagnation_count += 1
            else:
                stagnation_count += 1
            
            # Cool down temperature
            temperature *= self.config.simulated_annealing_cooling
            iteration += 1
            
            # Early stopping if no improvement for too long
            if stagnation_count > self.config.convergence_patience:
                logger.info(f"SA: Early stopping after {stagnation_count} iterations without improvement")
                break
                
            # Periodic logging
            if iteration % 10 == 0:
                logger.info(f"SA Iteration {iteration}: Fitness {current_fitness:.4f}, temp={temperature:.3f}")
        
        logger.info(f"Simulated Annealing completed after {iteration} iterations")
        return best_solution
    
    def particle_swarm_optimization(self) -> Dict[str, float]:
        """
        Particle Swarm Optimization for global search
        Good for exploring the search space and finding global optima
        """
        logger = self.session_manager.logger
        logger.info("Starting Particle Swarm Optimization")
        
        # Get heuristic names and bounds
        best_weights = self.session_manager.best_weights or self.session_manager.champion_pool[0].weights
        heuristic_names = list(best_weights.keys())
        bounds = self._get_search_bounds()
        
        # Initialize particles
        particles = []
        velocities = []
        personal_best = []
        personal_best_fitness = []
        
        for i in range(self.config.pso_particles):
            # Random initialization within bounds
            particle = {}
            velocity = {}
            for name in heuristic_names:
                min_val, max_val = bounds[name]
                particle[name] = random.uniform(min_val, max_val)
                velocity[name] = random.uniform(-0.1, 0.1)  # Small initial velocities
            
            particles.append(particle)
            velocities.append(velocity)
            
            # Evaluate initial particle
            result = self.evaluate_weights(particle)
            fitness = result.win_rate + (result.avg_score_diff / 100.0)
            personal_best.append(particle.copy())
            personal_best_fitness.append(fitness)
        
        # Find global best
        global_best = personal_best[0].copy()
        global_best_fitness = max(personal_best_fitness)
        global_best_index = personal_best_fitness.index(global_best_fitness)
        global_best = personal_best[global_best_index].copy()
        
        iteration = 0
        stagnation_count = 0
        
        while (time.time() - self.session_manager.start_time < self.config.max_duration_hours * 3600 and
               stagnation_count < self.config.convergence_patience):
            
            improved_global = False
            
            for i in range(len(particles)):
                # Update velocity
                for name in heuristic_names:
                    r1, r2 = random.random(), random.random()
                    
                    velocities[i][name] = (
                        self.config.pso_inertia * velocities[i][name] +
                        self.config.pso_cognitive * r1 * (personal_best[i][name] - particles[i][name]) +
                        self.config.pso_social * r2 * (global_best[name] - particles[i][name])
                    )
                    
                    # Velocity clamping
                    max_velocity = (bounds[name][1] - bounds[name][0]) * 0.1
                    velocities[i][name] = max(-max_velocity, min(max_velocity, velocities[i][name]))
                
                # Update position
                for name in heuristic_names:
                    particles[i][name] += velocities[i][name]
                    # Boundary constraints
                    min_val, max_val = bounds[name]
                    particles[i][name] = max(min_val, min(max_val, particles[i][name]))
                
                # Evaluate new position
                result = self.evaluate_weights(particles[i])
                fitness = result.win_rate + (result.avg_score_diff / 100.0)
                
                # Update personal best
                if fitness > personal_best_fitness[i]:
                    personal_best[i] = particles[i].copy()
                    personal_best_fitness[i] = fitness
                    
                    # Update global best
                    if fitness > global_best_fitness:
                        global_best = particles[i].copy()
                        global_best_fitness = fitness
                        improved_global = True
                        logger.info(f"PSO Iteration {iteration}: New global best fitness {global_best_fitness:.4f}")
            
            if improved_global:
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            iteration += 1
            
            # Periodic logging
            if iteration % 5 == 0:
                avg_fitness = sum(personal_best_fitness) / len(personal_best_fitness)
                logger.info(f"PSO Iteration {iteration}: Global best {global_best_fitness:.4f}, Average {avg_fitness:.4f}")
        
        logger.info(f"Particle Swarm Optimization completed after {iteration} iterations")
        return global_best
    
    def differential_evolution(self) -> Dict[str, float]:
        """
        Differential Evolution algorithm for robust global optimization
        Excellent at finding global optima through mutation and crossover
        """
        logger = self.session_manager.logger
        logger.info("Starting Differential Evolution optimization")
        
        best_weights = self.session_manager.best_weights or self.session_manager.champion_pool[0].weights
        heuristic_names = list(best_weights.keys())
        bounds = self._get_search_bounds()
        population_size = self.config.population_size
        
        # Initialize population randomly within bounds
        population = []
        fitness_scores = []
        
        for i in range(population_size):
            individual = {}
            for name in heuristic_names:
                min_val, max_val = bounds[name]
                individual[name] = random.uniform(min_val, max_val)
            
            population.append(individual)
            result = self.evaluate_weights(individual)
            fitness = result.win_rate + (result.avg_score_diff / 100.0)
            fitness_scores.append(fitness)
        
        # Find best individual
        best_fitness = max(fitness_scores)
        best_individual = population[fitness_scores.index(best_fitness)].copy()
        
        generation = 0
        stagnation_count = 0
        
        # DE parameters
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        
        while (time.time() - self.session_manager.start_time < self.config.max_duration_hours * 3600 and
               stagnation_count < self.config.convergence_patience):
            
            new_population = []
            new_fitness_scores = []
            improved_best = False
            
            for i in range(population_size):
                # Select three random individuals (different from current)
                candidates = list(range(population_size))
                candidates.remove(i)
                a, b, c = random.sample(candidates, 3)
                
                # Create mutant vector
                mutant = {}
                for name in heuristic_names:
                    mutant[name] = population[a][name] + F * (population[b][name] - population[c][name])
                    # Ensure bounds
                    min_val, max_val = bounds[name]
                    mutant[name] = max(min_val, min(max_val, mutant[name]))
                
                # Crossover
                trial = {}
                j_rand = random.randint(0, len(heuristic_names) - 1)  # Ensure at least one parameter is from mutant
                
                for j, name in enumerate(heuristic_names):
                    if random.random() < CR or j == j_rand:
                        trial[name] = mutant[name]
                    else:
                        trial[name] = population[i][name]
                
                # Selection
                trial_result = self.evaluate_weights(trial)
                trial_fitness = trial_result.win_rate + (trial_result.avg_score_diff / 100.0)
                
                if trial_fitness > fitness_scores[i]:
                    new_population.append(trial)
                    new_fitness_scores.append(trial_fitness)
                    
                    # Check if new global best
                    if trial_fitness > best_fitness:
                        best_individual = trial.copy()
                        best_fitness = trial_fitness
                        improved_best = True
                        logger.info(f"DE Generation {generation}: New best fitness {best_fitness:.4f}")
                else:
                    new_population.append(population[i])
                    new_fitness_scores.append(fitness_scores[i])
            
            population = new_population
            fitness_scores = new_fitness_scores
            
            if improved_best:
                stagnation_count = 0
            else:
                stagnation_count += 1
            
            generation += 1
            
            # Periodic logging
            if generation % 5 == 0:
                avg_fitness = sum(fitness_scores) / len(fitness_scores)
                logger.info(f"DE Generation {generation}: Best {best_fitness:.4f}, Average {avg_fitness:.4f}")
        
        logger.info(f"Differential Evolution completed after {generation} generations")
        return best_individual
    
    def _get_search_bounds(self) -> Dict[str, Tuple[float, float]]:
        """Get search space bounds for each heuristic"""
        if self.config.search_space_bounds:
            return self.config.search_space_bounds
        
        # Default intelligent bounds based on heuristic characteristics
        return {
            "piece_size": (0.5, 3.0),
            "blocked_opponents": (0.5, 10.0),  # Increased upper bound for aggressive blocking
            "corner_control": (0.5, 3.0),
            "compactness": (0.3, 2.5),
            "mobility": (0.3, 3.0),
            "opponent_restriction": (0.3, 4.0),
            "endgame_optimization": (0.5, 3.0),
            "corner_path_potential": (0.5, 5.0),  # Extended search radius, adjusted range
            # NEW: Opponent-aware heuristics (exploratory ranges)
            "opponent_territory_pressure": (0.0, 3.0),
            "opponent_mobility_restriction": (0.0, 4.0),
            "opponent_threat_assessment": (0.0, 2.5),
            "strategic_positioning": (0.0, 3.0)
        }
    
    def adaptive_multi_algorithm_search(self) -> Dict[str, float]:
        """
        Advanced adaptive search that combines multiple algorithms intelligently
        Uses convergence detection and algorithm switching for optimal results
        """
        logger = self.session_manager.logger
        logger.info("Starting Adaptive Multi-Algorithm Search")
        
        algorithms = [
            ("Genetic Algorithm", self.genetic_algorithm),
            ("Simulated Annealing", lambda: self.simulated_annealing()),
            ("Particle Swarm", self.particle_swarm_optimization),
            ("Differential Evolution", self.differential_evolution)
        ]
        
        best_weights = self.session_manager.best_weights or self.session_manager.champion_pool[0].weights
        best_solution = best_weights.copy()
        best_fitness = 0.0
        algorithm_performance = defaultdict(list)  # Track performance of each algorithm
        
        time_per_algorithm = self.config.max_duration_hours * 3600 / len(algorithms)
        
        for algo_name, algo_func in algorithms:
            if time.time() - self.session_manager.start_time >= self.config.max_duration_hours * 3600:
                break
                
            logger.info(f"Running {algo_name} phase")
            algo_start_time = time.time()
            
            try:
                # Temporarily adjust config for this algorithm
                original_duration = self.config.max_duration_hours
                remaining_time = (self.session_manager.start_time + original_duration * 3600 - time.time()) / 3600
                algorithm_time = min(time_per_algorithm / 3600, remaining_time)
                self.config.max_duration_hours = (time.time() - self.session_manager.start_time) / 3600 + algorithm_time
                
                solution = algo_func()
                
                # Evaluate solution quality
                result = self.evaluate_weights(solution)
                fitness = result.win_rate + (result.avg_score_diff / 100.0)
                
                algorithm_performance[algo_name].append({
                    'fitness': fitness,
                    'time': time.time() - algo_start_time,
                    'solution': solution
                })
                
                if fitness > best_fitness:
                    best_solution = solution.copy()
                    best_fitness = fitness
                    logger.info(f"{algo_name} found new best solution with fitness {best_fitness:.4f}")
                
                # Restore original duration
                self.config.max_duration_hours = original_duration
                
            except Exception as e:
                logger.warning(f"Algorithm {algo_name} failed: {e}")
                continue
        
        # Performance analysis
        logger.info("\n" + "="*60)
        logger.info("ALGORITHM PERFORMANCE SUMMARY")
        logger.info("="*60)
        for algo_name, performances in algorithm_performance.items():
            if performances:
                avg_fitness = sum(p['fitness'] for p in performances) / len(performances)
                total_time = sum(p['time'] for p in performances)
                logger.info(f"{algo_name}: Avg Fitness {avg_fitness:.4f}, Total Time {total_time:.1f}s")
        
        return best_solution
    
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
        """Run the complete comprehensive tuning session with advanced global optimization"""
        logger = self.session_manager.logger
        logger.info("=" * 80)
        logger.info("STARTING ADVANCED GLOBAL HYPERPARAMETER OPTIMIZATION")
        logger.info("=" * 80)
        
        try:
            total_duration = self.config.max_duration_hours * 3600
            
            # Phase 1: Adaptive Random Search (30% of time) - Initial exploration
            phase1_duration = self.config.max_duration_hours * 0.3
            logger.info(f"PHASE 1: Adaptive Random Search - Initial Exploration ({phase1_duration:.1f}h)")
            
            original_duration = self.config.max_duration_hours
            self.config.max_duration_hours = phase1_duration
            self.adaptive_random_search()
            
            # Phase 2: Multi-Algorithm Global Search (50% of time) - Global optimization
            remaining_time = (total_duration - (time.time() - self.session_manager.start_time)) / 3600
            if remaining_time > 0.2:  # At least 12 minutes remaining
                phase2_duration = min(remaining_time * 0.7, self.config.max_duration_hours * 0.5)
                logger.info(f"PHASE 2: Advanced Multi-Algorithm Global Search ({phase2_duration:.1f}h)")
                
                self.config.max_duration_hours = (time.time() - self.session_manager.start_time) / 3600 + phase2_duration
                
                # Use adaptive multi-algorithm search for better global optimization
                if remaining_time > 2.0:  # Use full multi-algorithm if enough time
                    self.adaptive_multi_algorithm_search()
                elif remaining_time > 1.0:  # Use PSO if moderate time
                    self.particle_swarm_optimization()
                else:  # Use genetic algorithm if limited time
                    if remaining_time < 1.0:
                        self.config.population_size = max(8, self.config.population_size // 2)
                        logger.info(f"Reduced population size to {self.config.population_size} due to time constraints")
                    self.genetic_algorithm()
            
            # Phase 3: Fine-tuning with Simulated Annealing (20% of time) - Local optimization
            remaining_time = (total_duration - (time.time() - self.session_manager.start_time)) / 3600
            if remaining_time > 0.1:  # At least 6 minutes remaining
                phase3_duration = min(remaining_time, self.config.max_duration_hours * 0.2)
                logger.info(f"PHASE 3: Simulated Annealing Fine-tuning ({phase3_duration:.1f}h)")
                
                self.config.max_duration_hours = (time.time() - self.session_manager.start_time) / 3600 + phase3_duration
                
                # Use current best solution as starting point for SA
                current_best = self.session_manager.best_weights or self.session_manager.champion_pool[0].weights
                self.simulated_annealing(current_best)
            
            # Restore original duration
            self.config.max_duration_hours = original_duration
            
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
        # Champion pool summary
        pool_stats = self.session_manager.get_champion_pool_stats()
        logger.info("CHAMPION POOL FINAL STATE:")
        logger.info(f"  Pool size: {pool_stats.get('pool_size', 0)}")
        logger.info(f"  Best win rate: {pool_stats.get('best_win_rate', 0):.1%}")
        logger.info(f"  Pool average win rate: {pool_stats.get('avg_win_rate', 0):.1%}")
        logger.info("")
        
        logger.info("Final Champion Pool:")
        for i, champion in enumerate(self.session_manager.champion_pool, 1):
            logger.info(f"  {i}. {champion.win_rate:.1%} ({champion.games_played} games) - {self.session_manager._format_weights(champion.weights)}")
        
        logger.info("")
        logger.info(f"Best overall win rate achieved: {self.session_manager.best_win_rate:.1%}")
        logger.info(f"Best overall weights: {self.session_manager.best_weights}")
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