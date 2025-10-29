"""
AI Player for Blokus
Modular AI system with configurable heuristics and strategies.
"""

from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy

from board import Board, PlayerColor
from pieces import Piece, PieceType
from game_state import GameState


@dataclass
class MoveEvaluation:
    """Represents an evaluated move with its score and reasoning"""
    piece_type: PieceType
    row: int
    col: int
    piece: Piece
    score: float
    heuristic_breakdown: Dict[str, float]
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        return {
            'piece_type': self.piece_type.value,
            'row': self.row,
            'col': self.col,
            'shape': self.piece.get_coordinates(),
            'score': self.score,
            'heuristic_breakdown': self.heuristic_breakdown
        }


class Heuristic:
    """Collection of heuristic evaluation functions"""
    
    @staticmethod
    def piece_size_score(piece: Piece, game_state: GameState, 
                         color: PlayerColor, row: int, col: int) -> float:
        """
        Score based on piece size. Larger pieces are generally better early in game.
        Returns: 1-5 based on piece size
        """
        return float(piece.size())
    
    @staticmethod
    def new_paths_score(piece: Piece, game_state: GameState, 
                       color: PlayerColor, row: int, col: int) -> float:
        """
        Score based on how many new corner-adjacent positions are created.
        More paths = more future move options.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        # Find all new diagonal positions that are empty
        new_corners = set()
        for r, c in coordinates:
            for diag_r, diag_c in board.get_diagonal_cells(r, c):
                # Check if diagonal cell is empty and not already occupied by us
                if board.is_empty(diag_r, diag_c):
                    # Check if this creates a valid corner connection
                    # (not edge-adjacent to our own color)
                    is_valid_corner = True
                    for adj_r, adj_c in board.get_adjacent_cells(diag_r, diag_c):
                        if board.get_cell(adj_r, adj_c) == color:
                            # Would be edge-adjacent to our color at this diagonal position
                            is_valid_corner = False
                            break
                    
                    if is_valid_corner:
                        new_corners.add((diag_r, diag_c))
        
        # More new corners = better move
        return float(len(new_corners))
    
    @staticmethod
    def blocked_opponents_score(piece: Piece, game_state: GameState, 
                                color: PlayerColor, row: int, col: int) -> float:
        """
        Score based on how many potential opponent paths are blocked.
        Blocking opponent corners is valuable.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        blocked_positions = 0
        
        # For each coordinate in our piece
        for r, c in coordinates:
            # Check diagonal cells - these could be opponent corner connections
            for diag_r, diag_c in board.get_diagonal_cells(r, c):
                if board.is_empty(diag_r, diag_c):
                    # Check if this diagonal position is adjacent to any opponent
                    for opponent_color in game_state.players.keys():
                        if opponent_color == color:
                            continue
                        
                        # Check if opponent has pieces adjacent to this diagonal
                        for adj_r, adj_c in board.get_adjacent_cells(diag_r, diag_c):
                            if board.get_cell(adj_r, adj_c) == opponent_color:
                                # This diagonal was potentially useful for opponent
                                # But now it's edge-adjacent to our new piece
                                blocked_positions += 1
                                break
        
        return float(blocked_positions)
    
    @staticmethod
    def corner_control_score(piece: Piece, game_state: GameState, 
                            color: PlayerColor, row: int, col: int) -> float:
        """
        Score based on controlling key board positions (corners, center).
        Board corners and center are strategically valuable.
        """
        coordinates = piece.translate(row, col)
        score = 0.0
        
        # Board corners are valuable
        board_corners = [(0, 0), (0, 19), (19, 0), (19, 19)]
        for corner in board_corners:
            if corner in coordinates:
                score += 2.0
        
        # Center area control
        center_min, center_max = 7, 12
        center_squares = sum(1 for r, c in coordinates 
                           if center_min <= r <= center_max and center_min <= c <= center_max)
        score += center_squares * 0.5
        
        return score
    
    @staticmethod
    def edge_avoidance_score(piece: Piece, game_state: GameState, 
                            color: PlayerColor, row: int, col: int) -> float:
        """
        Penalty for pieces too close to board edges (except starting corner).
        Edge positions limit future expansion.
        """
        coordinates = piece.translate(row, col)
        edge_penalty = 0.0
        
        starting_corner = Board.STARTING_CORNERS.get(color)
        
        for r, c in coordinates:
            # Skip penalty near starting corner early in game
            if starting_corner and game_state.turn_number <= 3:
                corner_r, corner_c = starting_corner
                if abs(r - corner_r) <= 2 and abs(c - corner_c) <= 2:
                    continue
            
            # Penalize edge positions
            if r == 0 or r == 19 or c == 0 or c == 19:
                edge_penalty += 1.0
        
        # Return negative penalty (lower is worse)
        return -edge_penalty
    
    @staticmethod
    def compactness_score(piece: Piece, game_state: GameState, 
                         color: PlayerColor, row: int, col: int) -> float:
        """
        Score based on how compact the placement is with existing pieces.
        Encourages building connected territories.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        # Count diagonal connections to our own pieces
        diagonal_connections = 0
        for r, c in coordinates:
            for diag_r, diag_c in board.get_diagonal_cells(r, c):
                if board.get_cell(diag_r, diag_c) == color:
                    diagonal_connections += 1
        
        return float(diagonal_connections) * 0.5


class AIStrategy(ABC):
    """Abstract base class for AI strategies"""
    
    def __init__(self, name: str = "Base AI"):
        self.name = name
        self.heuristics: Dict[str, Tuple[Callable, float]] = {}
    
    def add_heuristic(self, name: str, heuristic_func: Callable, weight: float):
        """Add a heuristic function with its weight"""
        self.heuristics[name] = (heuristic_func, weight)
    
    def evaluate_move(self, piece: Piece, piece_type: PieceType, 
                     row: int, col: int, game_state: GameState, 
                     color: PlayerColor) -> MoveEvaluation:
        """
        Evaluate a single move using all configured heuristics.
        
        Returns:
            MoveEvaluation with total score and breakdown
        """
        heuristic_breakdown = {}
        total_score = 0.0
        
        for heuristic_name, (heuristic_func, weight) in self.heuristics.items():
            heuristic_score = heuristic_func(piece, game_state, color, row, col)
            weighted_score = heuristic_score * weight
            heuristic_breakdown[heuristic_name] = weighted_score
            total_score += weighted_score
        
        return MoveEvaluation(
            piece_type=piece_type,
            row=row,
            col=col,
            piece=piece,
            score=total_score,
            heuristic_breakdown=heuristic_breakdown
        )
    
    @abstractmethod
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """
        Select the best move from a list of evaluations.
        Can be overridden for different selection strategies.
        """
        pass
    
    def get_all_possible_moves(self, game_state: GameState, 
                               color: PlayerColor) -> List[MoveEvaluation]:
        """
        Generate and evaluate all possible moves for the current player.
        
        Returns:
            List of MoveEvaluation objects, one for each valid move
        """
        player = game_state.get_player(color)
        if not player:
            return []
        
        all_evaluations = []
        
        # For each available piece
        for piece_obj in player.available_pieces:
            piece_type = piece_obj.piece_type
            
            # Get all valid moves for this piece
            valid_moves = game_state.get_valid_moves_for_piece(piece_type, color)
            
            # Evaluate each valid move
            for row, col, oriented_piece in valid_moves:
                evaluation = self.evaluate_move(
                    oriented_piece, piece_type, row, col, game_state, color
                )
                all_evaluations.append(evaluation)
        
        return all_evaluations
    
    def choose_move(self, game_state: GameState, 
                   color: PlayerColor) -> Optional[MoveEvaluation]:
        """
        Main method to choose the best move.
        
        Returns:
            MoveEvaluation of the chosen move, or None if no valid moves
        """
        all_moves = self.get_all_possible_moves(game_state, color)
        
        if not all_moves:
            return None
        
        return self.select_move(all_moves)


class GreedyAIStrategy(AIStrategy):
    """AI that always picks the highest scoring move"""
    
    def __init__(self):
        super().__init__("Greedy AI")
        
        # Configure heuristics with weights
        self.add_heuristic("piece_size", Heuristic.piece_size_score, 2.0)
        self.add_heuristic("new_paths", Heuristic.new_paths_score, 3.0)
        self.add_heuristic("blocked_opponents", Heuristic.blocked_opponents_score, 2.5)
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with highest score"""
        if not evaluations:
            return None
        return max(evaluations, key=lambda e: e.score)


class BalancedAIStrategy(AIStrategy):
    """Balanced AI that considers multiple factors"""
    
    def __init__(self):
        super().__init__("Balanced AI")
        
        # More balanced weight distribution
        self.add_heuristic("piece_size", Heuristic.piece_size_score, 1.5)
        self.add_heuristic("new_paths", Heuristic.new_paths_score, 2.5)
        self.add_heuristic("blocked_opponents", Heuristic.blocked_opponents_score, 2.0)
        self.add_heuristic("corner_control", Heuristic.corner_control_score, 1.5)
        self.add_heuristic("compactness", Heuristic.compactness_score, 1.0)
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with highest score"""
        if not evaluations:
            return None
        return max(evaluations, key=lambda e: e.score)


class AggressiveAIStrategy(AIStrategy):
    """Aggressive AI focused on blocking opponents"""
    
    def __init__(self):
        super().__init__("Aggressive AI")
        
        # Heavy emphasis on blocking opponents
        self.add_heuristic("piece_size", Heuristic.piece_size_score, 1.0)
        self.add_heuristic("new_paths", Heuristic.new_paths_score, 1.5)
        self.add_heuristic("blocked_opponents", Heuristic.blocked_opponents_score, 4.0)
        self.add_heuristic("corner_control", Heuristic.corner_control_score, 2.0)
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with highest score"""
        if not evaluations:
            return None
        return max(evaluations, key=lambda e: e.score)


class ExpansiveAIStrategy(AIStrategy):
    """Expansive AI focused on creating many future move options"""
    
    def __init__(self):
        super().__init__("Expansive AI")
        
        # Focus on creating new paths and avoiding edges
        self.add_heuristic("piece_size", Heuristic.piece_size_score, 1.5)
        self.add_heuristic("new_paths", Heuristic.new_paths_score, 4.0)
        self.add_heuristic("blocked_opponents", Heuristic.blocked_opponents_score, 1.0)
        self.add_heuristic("edge_avoidance", Heuristic.edge_avoidance_score, 2.0)
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with highest score"""
        if not evaluations:
            return None
        return max(evaluations, key=lambda e: e.score)


class AIPlayer:
    """
    Main AI Player class that uses a strategy to make moves.
    This class is used by the game to interact with AI players.
    """
    
    def __init__(self, color: PlayerColor, strategy: AIStrategy):
        self.color = color
        self.strategy = strategy
    
    def get_move(self, game_state: GameState) -> Optional[MoveEvaluation]:
        """
        Get the AI's chosen move for the current game state.
        
        Returns:
            MoveEvaluation of the chosen move, or None if no valid moves
        """
        return self.strategy.choose_move(game_state, self.color)
    
    def get_all_evaluated_moves(self, game_state: GameState) -> List[MoveEvaluation]:
        """
        Get all possible moves with their evaluations.
        Useful for showing the AI's "thinking process".
        
        Returns:
            List of all evaluated moves, sorted by score (highest first)
        """
        all_moves = self.strategy.get_all_possible_moves(game_state, self.color)
        return sorted(all_moves, key=lambda e: e.score, reverse=True)


# Factory function to create AI players with different strategies
def create_ai_player(color: PlayerColor, strategy_name: str = "balanced") -> AIPlayer:
    """
    Factory function to create an AI player with a specific strategy.
    
    Args:
        color: The player's color
        strategy_name: Strategy type - "greedy", "balanced", "aggressive", "expansive", or "optimized"
    
    Returns:
        AIPlayer instance with the specified strategy
    """
    # Try importing enhanced AI first
    try:
        from ai_player_enhanced import OptimizedAIStrategy
        if strategy_name.lower() == "optimized":
            strategy = OptimizedAIStrategy()
            return AIPlayer(color, strategy)
    except ImportError:
        pass
    
    strategies = {
        "greedy": GreedyAIStrategy,
        "balanced": BalancedAIStrategy,
        "aggressive": AggressiveAIStrategy,
        "expansive": ExpansiveAIStrategy
    }
    
    strategy_class = strategies.get(strategy_name.lower(), BalancedAIStrategy)
    strategy = strategy_class()
    
    return AIPlayer(color, strategy)
