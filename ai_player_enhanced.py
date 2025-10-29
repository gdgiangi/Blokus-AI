"""
Enhanced AI Player for Blokus with Advanced Heuristics and Hyperparameter Tuning
Includes additional strategic heuristics and a framework for optimization.
"""

from typing import List, Tuple, Dict, Callable, Optional
from dataclasses import dataclass
from abc import ABC, abstractmethod
import copy
import random
import json

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


class EnhancedHeuristic:
    """Collection of enhanced heuristic evaluation functions"""
    
    @staticmethod
    def piece_size_score(piece: Piece, game_state: GameState, 
                         color: PlayerColor, row: int, col: int) -> float:
        """
        Score based on piece size with game phase awareness.
        Early game: prefer larger pieces. Late game: be more flexible.
        """
        size = float(piece.size())
        turn_ratio = game_state.turn_number / 20.0  # Normalize by expected game length
        
        # Early game: strong preference for large pieces
        # Late game: any piece that fits is good
        if turn_ratio < 0.3:  # Early game
            return size * 1.5
        elif turn_ratio < 0.7:  # Mid game
            return size
        else:  # Late game
            return size * 0.5
    
    @staticmethod
    def new_paths_score(piece: Piece, game_state: GameState, 
                       color: PlayerColor, row: int, col: int) -> float:
        """
        Score based on how many new valid corner-adjacent positions are created.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        new_corners = set()
        for r, c in coordinates:
            for diag_r, diag_c in board.get_diagonal_cells(r, c):
                if board.is_empty(diag_r, diag_c):
                    # Check if this would be a valid corner (not edge-adjacent to our color)
                    is_valid_corner = True
                    for adj_r, adj_c in board.get_adjacent_cells(diag_r, diag_c):
                        if board.get_cell(adj_r, adj_c) == color:
                            is_valid_corner = False
                            break
                    
                    if is_valid_corner:
                        new_corners.add((diag_r, diag_c))
        
        return float(len(new_corners))
    
    @staticmethod
    def blocked_opponents_score(piece: Piece, game_state: GameState, 
                                color: PlayerColor, row: int, col: int) -> float:
        """
        Score based on how many opponent corner connections are blocked.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        blocked_count = 0
        
        for r, c in coordinates:
            # Check all adjacent cells
            for adj_r, adj_c in board.get_adjacent_cells(r, c):
                # For each opponent
                for opponent_color in game_state.players.keys():
                    if opponent_color == color:
                        continue
                    
                    # If opponent has a piece here, this could block their diagonal expansions
                    if board.get_cell(adj_r, adj_c) == opponent_color:
                        # Check diagonals from this opponent piece
                        for diag_r, diag_c in board.get_diagonal_cells(adj_r, adj_c):
                            if (diag_r, diag_c) in coordinates:
                                # We're placing a piece on their potential corner expansion
                                blocked_count += 1
        
        return float(blocked_count)
    
    @staticmethod
    def corner_control_score(piece: Piece, game_state: GameState, 
                            color: PlayerColor, row: int, col: int) -> float:
        """
        Strategic board position control.
        """
        coordinates = piece.translate(row, col)
        score = 0.0
        
        # Board corners
        board_corners = [(0, 0), (0, 19), (19, 0), (19, 19)]
        for corner in board_corners:
            if corner in coordinates:
                score += 3.0
        
        # Center control (coordinates 8-11)
        center_min, center_max = 8, 11
        for r, c in coordinates:
            if center_min <= r <= center_max and center_min <= c <= center_max:
                score += 1.0
        
        # Strategic "thirds" - control different board sections
        for r, c in coordinates:
            if 6 <= r <= 13 and 6 <= c <= 13:
                score += 0.5  # Center third
        
        return score
    
    @staticmethod
    def edge_avoidance_score(piece: Piece, game_state: GameState, 
                            color: PlayerColor, row: int, col: int) -> float:
        """
        Penalty for being pushed to edges (except starting corner area).
        """
        coordinates = piece.translate(row, col)
        penalty = 0.0
        
        starting_corner = Board.STARTING_CORNERS.get(color)
        
        for r, c in coordinates:
            # Skip penalty near starting corner early game
            if starting_corner and game_state.turn_number <= 4:
                corner_r, corner_c = starting_corner
                if abs(r - corner_r) <= 3 and abs(c - corner_c) <= 3:
                    continue
            
            # Penalize edge positions
            edge_distance = min(r, 19-r, c, 19-c)
            if edge_distance == 0:
                penalty += 1.5
            elif edge_distance == 1:
                penalty += 0.5
        
        return -penalty
    
    @staticmethod
    def compactness_score(piece: Piece, game_state: GameState, 
                         color: PlayerColor, row: int, col: int) -> float:
        """
        Building connected territories.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        diagonal_connections = 0
        for r, c in coordinates:
            for diag_r, diag_c in board.get_diagonal_cells(r, c):
                if board.get_cell(diag_r, diag_c) == color:
                    diagonal_connections += 1
        
        return float(diagonal_connections) * 0.75
    
    @staticmethod
    def flexibility_score(piece: Piece, game_state: GameState,
                         color: PlayerColor, row: int, col: int) -> float:
        """
        NEW: Prefer moves that maintain multiple directions for expansion.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        # Count how many quadrants we have access to from this piece
        directions = set()
        for r, c in coordinates:
            for diag_r, diag_c in board.get_diagonal_cells(r, c):
                if board.is_empty(diag_r, diag_c):
                    # Determine direction
                    dr = diag_r - r
                    dc = diag_c - c
                    directions.add((dr > 0, dc > 0))  # Quadrant indicator
        
        return float(len(directions))
    
    @staticmethod
    def mobility_score(piece: Piece, game_state: GameState,
                      color: PlayerColor, row: int, col: int) -> float:
        """
        NEW: Score based on estimated future mobility after this move.
        Higher mobility = more options in future turns.
        """
        player = game_state.get_player(color)
        if not player:
            return 0.0
        
        # Estimate: count remaining small pieces that could fit in tight spaces
        small_pieces = sum(1 for p in player.available_pieces if p.size() <= 3)
        
        # If we have many small pieces, prioritize moves that create tight corner spaces
        # If we have few small pieces, avoid creating isolated corners
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        tight_corners = 0
        for r, c in coordinates:
            for diag_r, diag_c in board.get_diagonal_cells(r, c):
                if board.is_empty(diag_r, diag_c):
                    # Count how many sides are blocked
                    blocked_sides = sum(1 for adj_r, adj_c in board.get_adjacent_cells(diag_r, diag_c)
                                      if not board.is_empty(adj_r, adj_c))
                    if blocked_sides >= 2:
                        tight_corners += 1
        
        if small_pieces >= 3:
            return float(tight_corners) * 0.5  # Reward tight corners if we have small pieces
        else:
            return -float(tight_corners) * 0.3  # Penalize if we don't have small pieces
    
    @staticmethod
    def opponent_restriction_score(piece: Piece, game_state: GameState,
                                   color: PlayerColor, row: int, col: int) -> float:
        """
        NEW: Advanced opponent blocking - reduce their total available moves.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        restriction_value = 0.0
        
        # For each opponent
        for opponent_color in game_state.players.keys():
            if opponent_color == color:
                continue
            
            # Check how many of their current corner positions we're occupying
            opponent_positions = board.get_occupied_positions(opponent_color)
            for opp_r, opp_c in opponent_positions:
                for diag_r, diag_c in board.get_diagonal_cells(opp_r, opp_c):
                    if (diag_r, diag_c) in coordinates:
                        restriction_value += 2.0
        
        return restriction_value
    
    @staticmethod
    def endgame_optimization_score(piece: Piece, game_state: GameState,
                                   color: PlayerColor, row: int, col: int) -> float:
        """
        NEW: Special scoring for endgame scenarios.
        """
        player = game_state.get_player(color)
        if not player:
            return 0.0
        
        pieces_remaining = len(player.available_pieces)
        
        # If we're down to last few pieces, prioritize placing them anywhere
        if pieces_remaining <= 3:
            return 5.0  # High bonus just for being able to place
        
        # If we're in endgame, smaller pieces become more valuable
        if pieces_remaining <= 7:
            size_score = (6 - piece.size()) * 0.5  # Invert: smaller is better
            return size_score
        
        return 0.0
    
    @staticmethod
    def territory_expansion_score(piece: Piece, game_state: GameState,
                                  color: PlayerColor, row: int, col: int) -> float:
        """
        NEW: Reward expanding into unclaimed territory.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        expansion_score = 0.0
        
        # Check if we're moving into "open" space
        for r, c in coordinates:
            # Count nearby occupied cells (within 3 squares)
            nearby_occupied = 0
            for dr in range(-3, 4):
                for dc in range(-3, 4):
                    nr, nc = r + dr, c + dc
                    if board.is_valid_position(nr, nc) and not board.is_empty(nr, nc):
                        nearby_occupied += 1
            
            # Reward placing in less crowded areas
            if nearby_occupied < 5:
                expansion_score += 1.0
        
        return expansion_score


class AIStrategy(ABC):
    """Abstract base class for AI strategies"""
    
    def __init__(self, name: str = "Base AI"):
        self.name = name
        self.heuristics: Dict[str, Tuple[Callable, float]] = {}
    
    def add_heuristic(self, name: str, heuristic_func: Callable, weight: float):
        """Add a heuristic function with its weight"""
        self.heuristics[name] = (heuristic_func, weight)
    
    def set_weights(self, weights: Dict[str, float]):
        """Update heuristic weights"""
        for name, weight in weights.items():
            if name in self.heuristics:
                func = self.heuristics[name][0]
                self.heuristics[name] = (func, weight)
    
    def get_weights(self) -> Dict[str, float]:
        """Get current heuristic weights"""
        return {name: weight for name, (_, weight) in self.heuristics.items()}
    
    def evaluate_move(self, piece: Piece, piece_type: PieceType, 
                     row: int, col: int, game_state: GameState, 
                     color: PlayerColor) -> MoveEvaluation:
        """Evaluate a single move using all configured heuristics"""
        heuristic_breakdown = {}
        total_score = 0.0
        
        for heuristic_name, (heuristic_func, weight) in self.heuristics.items():
            try:
                heuristic_score = heuristic_func(piece, game_state, color, row, col)
                weighted_score = heuristic_score * weight
                heuristic_breakdown[heuristic_name] = weighted_score
                total_score += weighted_score
            except Exception as e:
                print(f"Error in heuristic {heuristic_name}: {e}")
                heuristic_breakdown[heuristic_name] = 0.0
        
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
        """Select the best move from evaluations"""
        pass
    
    def get_all_possible_moves(self, game_state: GameState, 
                               color: PlayerColor) -> List[MoveEvaluation]:
        """Generate and evaluate ALL possible moves exhaustively"""
        player = game_state.get_player(color)
        if not player:
            return []
        
        all_evaluations = []
        
        # For each available piece (these are pieces the player still has)
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
        """Main method to choose the best move"""
        all_moves = self.get_all_possible_moves(game_state, color)
        
        if not all_moves:
            return None
        
        return self.select_move(all_moves)


# Enhanced Strategies
class OptimizedAIStrategy(AIStrategy):
    """Optimized AI with tuned weights from hyperparameter search"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("Optimized AI")
        
        # Default optimized weights (can be updated via hyperparameter tuning)
        default_weights = {
            "piece_size": 1.8,
            "new_paths": 3.2,
            "blocked_opponents": 2.1,
            "corner_control": 1.6,
            "compactness": 0.9,
            "flexibility": 2.4,
            "mobility": 1.3,
            "opponent_restriction": 2.8,
            "endgame_optimization": 1.5,
            "territory_expansion": 1.1
        }
        
        weights = weights or default_weights
        
        self.add_heuristic("piece_size", EnhancedHeuristic.piece_size_score, weights.get("piece_size", 1.8))
        self.add_heuristic("new_paths", EnhancedHeuristic.new_paths_score, weights.get("new_paths", 3.2))
        self.add_heuristic("blocked_opponents", EnhancedHeuristic.blocked_opponents_score, weights.get("blocked_opponents", 2.1))
        self.add_heuristic("corner_control", EnhancedHeuristic.corner_control_score, weights.get("corner_control", 1.6))
        self.add_heuristic("compactness", EnhancedHeuristic.compactness_score, weights.get("compactness", 0.9))
        self.add_heuristic("flexibility", EnhancedHeuristic.flexibility_score, weights.get("flexibility", 2.4))
        self.add_heuristic("mobility", EnhancedHeuristic.mobility_score, weights.get("mobility", 1.3))
        self.add_heuristic("opponent_restriction", EnhancedHeuristic.opponent_restriction_score, weights.get("opponent_restriction", 2.8))
        self.add_heuristic("endgame_optimization", EnhancedHeuristic.endgame_optimization_score, weights.get("endgame_optimization", 1.5))
        self.add_heuristic("territory_expansion", EnhancedHeuristic.territory_expansion_score, weights.get("territory_expansion", 1.1))
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with highest score"""
        if not evaluations:
            return None
        return max(evaluations, key=lambda e: e.score)


class AIPlayer:
    """Main AI Player class"""
    
    def __init__(self, color: PlayerColor, strategy: AIStrategy):
        self.color = color
        self.strategy = strategy
    
    def get_move(self, game_state: GameState) -> Optional[MoveEvaluation]:
        """Get the AI's chosen move"""
        return self.strategy.choose_move(game_state, self.color)
    
    def get_all_evaluated_moves(self, game_state: GameState) -> List[MoveEvaluation]:
        """Get all possible moves with evaluations"""
        all_moves = self.strategy.get_all_possible_moves(game_state, self.color)
        return sorted(all_moves, key=lambda e: e.score, reverse=True)


def create_ai_player(color: PlayerColor, strategy_name: str = "optimized") -> AIPlayer:
    """Factory function to create AI players"""
    from ai_player import GreedyAIStrategy, BalancedAIStrategy, AggressiveAIStrategy, ExpansiveAIStrategy
    
    if strategy_name.lower() == "optimized":
        strategy = OptimizedAIStrategy()
    elif strategy_name.lower() == "greedy":
        strategy = GreedyAIStrategy()
    elif strategy_name.lower() == "balanced":
        strategy = BalancedAIStrategy()
    elif strategy_name.lower() == "aggressive":
        strategy = AggressiveAIStrategy()
    elif strategy_name.lower() == "expansive":
        strategy = ExpansiveAIStrategy()
    else:
        strategy = OptimizedAIStrategy()
    
    return AIPlayer(color, strategy)
