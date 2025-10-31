"""
AI Player for Blokus - Simplified Interface
This file provides a simple interface to the advanced AI strategies.
All actual AI logic is now in ai_player_enhanced.py
"""

from typing import Optional
from board import PlayerColor

# Import everything from enhanced AI module
from ai_player_enhanced import (
    MoveEvaluation, 
    AIPlayer as EnhancedAIPlayer,
    OptimizedAIStrategy,
    AggressiveOptimizedStrategy,
    BalancedOptimizedStrategy,
    DefensiveOptimizedStrategy,
    CasualAIStrategy,
    RandomAIStrategy,
    difficulty_level_to_strategy,
    strategy_to_difficulty_level,
    get_difficulty_info
)

# Alias for backward compatibility
AIPlayer = EnhancedAIPlayer


def create_ai_player(color: PlayerColor, strategy_name: str = "balanced") -> EnhancedAIPlayer:
    """
    Factory function to create an AI player with a specific optimized strategy.
    """
    # Map strategy names to strategy classes
    strategy_mapping = {
        "aggressive": AggressiveOptimizedStrategy,
        "balanced": BalancedOptimizedStrategy,
        "defensive": DefensiveOptimizedStrategy,
        "casual": CasualAIStrategy,
        "optimized": OptimizedAIStrategy,  # Champion baseline
        "random": RandomAIStrategy,  # Random play
    }
    
    # Handle MCTS separately
    if strategy_name.lower() == "mcts":
        try:
            from mcts_ai import MCTSAIStrategy
            strategy = MCTSAIStrategy(time_limit=15.0, max_iterations=1000)
            return EnhancedAIPlayer(color, strategy)
        except ImportError:
            # Fallback to optimized if MCTS not available
            strategy_name = "optimized"
    
    # Get the strategy class, default to balanced
    strategy_class = strategy_mapping.get(strategy_name.lower(), BalancedOptimizedStrategy)
    strategy = strategy_class()
    
    return EnhancedAIPlayer(color, strategy)


# Export commonly used classes for backward compatibility
__all__ = [
    'AIPlayer', 
    'MoveEvaluation', 
    'create_ai_player',
    'OptimizedAIStrategy',
    'AggressiveOptimizedStrategy',
    'BalancedOptimizedStrategy',
    'DefensiveOptimizedStrategy',
    'CasualAIStrategy',
    'RandomAIStrategy',
    'difficulty_level_to_strategy',
    'strategy_to_difficulty_level',
    'get_difficulty_info'
]
