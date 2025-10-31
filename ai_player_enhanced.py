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
    corner_search_data: Optional[Dict] = None  # Visualization data for corner expansion
    
    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization"""
        result = {
            'piece_type': self.piece_type.value,
            'row': self.row,
            'col': self.col,
            'shape': self.piece.get_coordinates(),
            'score': self.score,
            'heuristic_breakdown': self.heuristic_breakdown
        }
        if self.corner_search_data:
            result['corner_search_data'] = self.corner_search_data
        return result


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

    @staticmethod
    def opponent_territory_pressure_score(piece: Piece, game_state: GameState,
                                         color: PlayerColor, row: int, col: int) -> float:
        """
        OPPONENT-AWARE: Score based on pressuring opponent territories and corners.
        Analyzes each opponent's expansion zones and rewards moves that constrict them.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        pressure_score = 0.0
        
        # For each opponent
        for opponent_color in game_state.players.keys():
            if opponent_color == color:
                continue
            
            opponent_player = game_state.get_player(opponent_color)
            if not opponent_player or opponent_player.has_passed:
                continue
            
            # Find all opponent corners (their expansion points)
            opponent_corners = set()
            for r in range(Board.BOARD_SIZE):
                for c in range(Board.BOARD_SIZE):
                    if board.get_cell(r, c) == opponent_color:
                        # Check diagonals for potential corners
                        for diag_r, diag_c in board.get_diagonal_cells(r, c):
                            if board.is_empty(diag_r, diag_c):
                                # Verify it's a valid corner
                                is_valid = True
                                for adj_r, adj_c in board.get_adjacent_cells(diag_r, diag_c):
                                    if board.get_cell(adj_r, adj_c) == opponent_color:
                                        is_valid = False
                                        break
                                if is_valid:
                                    opponent_corners.add((diag_r, diag_c))
            
            # Score based on proximity to opponent corners
            for opp_r, opp_c in opponent_corners:
                for my_r, my_c in coordinates:
                    distance = abs(my_r - opp_r) + abs(my_c - opp_c)  # Manhattan distance
                    
                    # Close proximity to opponent corners = high pressure
                    if distance <= 2:
                        pressure_score += 3.0  # Very close - immediate threat
                    elif distance <= 4:
                        pressure_score += 1.5  # Close - pressure zone
                    elif distance <= 6:
                        pressure_score += 0.5  # Medium distance - zone control
        
        return pressure_score
    
    @staticmethod
    def opponent_mobility_restriction_score(piece: Piece, game_state: GameState,
                                           color: PlayerColor, row: int, col: int) -> float:
        """
        OPPONENT-AWARE: Measure how much this move reduces opponent expansion options.
        Counts opponent corners that become unusable or less valuable.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        restriction_score = 0.0
        
        # Simulate placing the piece
        test_board = copy.deepcopy(board)
        test_board.place_piece(coordinates, color)
        
        # For each opponent
        for opponent_color in game_state.players.keys():
            if opponent_color == color:
                continue
            
            opponent_player = game_state.get_player(opponent_color)
            if not opponent_player or opponent_player.has_passed:
                continue
            
            # Count how many of opponent's current corners become blocked/restricted
            for r in range(Board.BOARD_SIZE):
                for c in range(Board.BOARD_SIZE):
                    if board.get_cell(r, c) == opponent_color:
                        for diag_r, diag_c in board.get_diagonal_cells(r, c):
                            if board.is_empty(diag_r, diag_c):
                                # Check if this corner is still valid after our move
                                corner_was_valid = True
                                for adj_r, adj_c in board.get_adjacent_cells(diag_r, diag_c):
                                    if board.get_cell(adj_r, adj_c) == opponent_color:
                                        corner_was_valid = False
                                        break
                                
                                if corner_was_valid:
                                    # Check if our piece blocks or restricts this corner
                                    if test_board.get_cell(diag_r, diag_c) == color:
                                        # We directly occupy their corner
                                        restriction_score += 2.0
                                    else:
                                        # Check if we reduce the expansion value around this corner
                                        open_space_before = 0
                                        open_space_after = 0
                                        
                                        for dr in [-1, 0, 1]:
                                            for dc in [-1, 0, 1]:
                                                nr, nc = diag_r + dr, diag_c + dc
                                                if board.is_valid_position(nr, nc):
                                                    if board.is_empty(nr, nc):
                                                        open_space_before += 1
                                                    if test_board.is_empty(nr, nc):
                                                        open_space_after += 1
                                        
                                        space_reduction = open_space_before - open_space_after
                                        if space_reduction > 0:
                                            restriction_score += space_reduction * 0.3
        
        return restriction_score
    
    @staticmethod
    def opponent_threat_assessment_score(piece: Piece, game_state: GameState,
                                        color: PlayerColor, row: int, col: int) -> float:
        """
        OPPONENT-AWARE: Identify and respond to opponent threats.
        Analyzes opponent positions, remaining pieces, and expansion potential.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        threat_score = 0.0
        
        player = game_state.get_player(color)
        if not player:
            return 0.0
        
        # Analyze each opponent
        for opponent_color in game_state.players.keys():
            if opponent_color == color:
                continue
            
            opponent_player = game_state.get_player(opponent_color)
            if not opponent_player or opponent_player.has_passed:
                continue
            
            # Threat level based on opponent strength
            opponent_pieces_remaining = len(opponent_player.available_pieces)
            our_pieces_remaining = len(player.available_pieces)
            
            # If opponent has more pieces, they're more threatening
            piece_advantage = opponent_pieces_remaining - our_pieces_remaining
            
            # Count opponent's active corners (expansion capability)
            opponent_active_corners = 0
            opponent_corner_positions = []
            
            for r in range(Board.BOARD_SIZE):
                for c in range(Board.BOARD_SIZE):
                    if board.get_cell(r, c) == opponent_color:
                        for diag_r, diag_c in board.get_diagonal_cells(r, c):
                            if board.is_empty(diag_r, diag_c):
                                is_valid = True
                                for adj_r, adj_c in board.get_adjacent_cells(diag_r, diag_c):
                                    if board.get_cell(adj_r, adj_c) == opponent_color:
                                        is_valid = False
                                        break
                                if is_valid:
                                    opponent_active_corners += 1
                                    opponent_corner_positions.append((diag_r, diag_c))
            
            # Threat multiplier based on opponent's expansion capability
            threat_multiplier = 1.0
            if opponent_active_corners > 8:
                threat_multiplier = 1.5  # Opponent is highly mobile
            elif opponent_active_corners > 5:
                threat_multiplier = 1.2  # Opponent has good options
            elif opponent_active_corners < 3:
                threat_multiplier = 0.7  # Opponent is struggling
            
            # Reward moves that defend against threatening opponents
            if piece_advantage > 3:  # Opponent is ahead
                # Check if our move creates a defensive barrier
                for my_r, my_c in coordinates:
                    for opp_corner_r, opp_corner_c in opponent_corner_positions:
                        distance = abs(my_r - opp_corner_r) + abs(my_c - opp_corner_c)
                        if distance <= 3:
                            threat_score += 1.0 * threat_multiplier
            
            # Bonus for blocking the leading opponent
            if piece_advantage > 0:
                threat_score *= (1.0 + piece_advantage * 0.1)
        
        return threat_score
    
    @staticmethod
    def strategic_positioning_score(piece: Piece, game_state: GameState,
                                   color: PlayerColor, row: int, col: int) -> float:
        """
        OPPONENT-AWARE: Advanced board positioning that considers opponent locations.
        Creates strategic zones of control and denies key areas to opponents.
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        position_score = 0.0
        
        # Calculate board control zones
        center = 10  # Board center (20x20, so center is at 10)
        
        # Analyze territory control in quadrants
        quadrant_occupancy = {
            'nw': {'us': 0, 'them': 0},  # Northwest
            'ne': {'us': 0, 'them': 0},  # Northeast
            'sw': {'us': 0, 'them': 0},  # Southwest
            'se': {'us': 0, 'them': 0}   # Southeast
        }
        
        # Count current occupancy
        for r in range(Board.BOARD_SIZE):
            for c in range(Board.BOARD_SIZE):
                cell_color = board.get_cell(r, c)
                if cell_color is None:
                    continue
                
                # Determine quadrant
                quad_key = ('n' if r < center else 's') + ('w' if c < center else 'e')
                
                if cell_color == color:
                    quadrant_occupancy[quad_key]['us'] += 1
                else:
                    quadrant_occupancy[quad_key]['them'] += 1
        
        # Reward moves in underrepresented quadrants
        for my_r, my_c in coordinates:
            my_quad = ('n' if my_r < center else 's') + ('w' if my_c < center else 'e')
            
            us_count = quadrant_occupancy[my_quad]['us']
            them_count = quadrant_occupancy[my_quad]['them']
            
            # Reward expanding into quadrants where opponents dominate
            if them_count > us_count:
                position_score += 1.5
            # Reward balanced expansion
            elif us_count < 10:
                position_score += 0.5
        
        # Reward control of key strategic lines (diagonals and center lines)
        for my_r, my_c in coordinates:
            # Main diagonals
            if my_r == my_c or my_r + my_c == 19:
                position_score += 0.8
            
            # Center lines (creates cross-board pressure)
            if 8 <= my_r <= 11 or 8 <= my_c <= 11:
                position_score += 0.6
        
        return position_score

    @staticmethod
    def corner_path_potential_score(piece: Piece, game_state: GameState,
                                    color: PlayerColor, row: int, col: int,
                                    collect_search_data: bool = False) -> Tuple[float, Optional[Dict]]:
        """
        NEW: Analyze the open space and expansion potential around each new corner.
        Higher score for corners with more open space and expansion opportunities.
        
        If collect_search_data is True, returns (score, search_visualization_data).
        Otherwise returns (score, None).
        """
        coordinates = piece.translate(row, col)
        board = game_state.board
        
        total_potential = 0.0
        all_corners_data = [] if collect_search_data else None
        
        # For each cell of the placed piece
        for r, c in coordinates:
            # Check each diagonal (potential new corner)
            for diag_r, diag_c in board.get_diagonal_cells(r, c):
                if not board.is_empty(diag_r, diag_c):
                    continue
                
                # Verify this is a valid corner (not edge-adjacent to our color)
                is_valid_corner = True
                for adj_r, adj_c in board.get_adjacent_cells(diag_r, diag_c):
                    if board.get_cell(adj_r, adj_c) == color:
                        is_valid_corner = False
                        break
                
                if not is_valid_corner:
                    continue
                
                # Analyze the potential around this corner
                corner_potential = 0.0
                cells_examined_list = [] if collect_search_data else None
                
                # 1. Count open space in expanding radius around corner (INCREASED RANGE)
                # Extended from radius 3 to radius 6 for better global awareness
                for radius in [1, 2, 3, 4, 5, 6]:
                    open_cells_at_radius = 0
                    total_cells_at_radius = 0
                    
                    for dr in range(-radius, radius + 1):
                        for dc in range(-radius, radius + 1):
                            if abs(dr) == radius or abs(dc) == radius:  # Cells at exact radius
                                nr, nc = diag_r + dr, diag_c + dc
                                if board.is_valid_position(nr, nc):
                                    total_cells_at_radius += 1
                                    is_empty = board.is_empty(nr, nc)
                                    if is_empty:
                                        open_cells_at_radius += 1
                                    
                                    # Collect search visualization data (only up to radius 3 for UI)
                                    if collect_search_data and cells_examined_list is not None and radius <= 3:
                                        cells_examined_list.append({
                                            'row': nr,
                                            'col': nc,
                                            'radius': radius,
                                            'is_empty': is_empty,
                                            'cell_type': 'expansion_search'
                                        })
                    
                    if total_cells_at_radius > 0:
                        openness_ratio = open_cells_at_radius / total_cells_at_radius
                        # Updated weighting: still favor closer cells but value distant open space
                        # Radius 1-3: high weight (original behavior)
                        # Radius 4-6: moderate weight for global awareness
                        if radius <= 3:
                            corner_potential += openness_ratio * (4 - radius)
                        else:
                            # Radius 4, 5, 6 get weights: 0.6, 0.4, 0.3
                            corner_potential += openness_ratio * (1.0 / radius)
                
                # 2. Bonus for corners near board edges (more directional freedom)
                edge_distance = min(diag_r, 19 - diag_r, diag_c, 19 - diag_c)
                if edge_distance <= 2:
                    corner_potential += 1.0
                
                # 3. Penalty if corner is surrounded by opponent pieces
                opponent_adjacent = 0
                for adj_r, adj_c in board.get_adjacent_cells(diag_r, diag_c):
                    cell_color = board.get_cell(adj_r, adj_c)
                    if cell_color != color and cell_color is not None:
                        opponent_adjacent += 1
                
                if opponent_adjacent >= 2:
                    corner_potential *= 0.5  # Reduce potential if hemmed in
                
                # 4. Bonus for corners that could connect to multiple directions (EXTENDED RANGE)
                # Extended from 2-3 cells to 2-5 cells for better global vision
                connection_directions = 0
                for direction in [(0, 1), (1, 0), (0, -1), (-1, 0)]:
                    dr, dc = direction
                    # Look further in each direction for open space
                    open_in_direction = True
                    for dist in [2, 3, 4, 5]:
                        nr, nc = diag_r + dr * dist, diag_c + dc * dist
                        if not board.is_valid_position(nr, nc) or not board.is_empty(nr, nc):
                            open_in_direction = False
                            break

                        # Collect directional search data (only dist 2-6 for UI visualization)
                        if collect_search_data and cells_examined_list is not None and board.is_valid_position(nr, nc) and dist <= 6:
                            cells_examined_list.append({
                                'row': nr,
                                'col': nc,
                                'radius': 0,  # Special marker for directional search
                                'is_empty': board.is_empty(nr, nc),
                                'cell_type': 'directional_search',
                                'direction': direction
                            })
                    
                    if open_in_direction:
                        connection_directions += 1
                
                # Increased bonus for multi-directional expansion (from 0.5 to 1.0 per direction)
                corner_potential += connection_directions * 1.0
                
                total_potential += corner_potential
                
                # Store corner search data
                if collect_search_data and all_corners_data is not None:
                    all_corners_data.append({
                        'position': [diag_r, diag_c],
                        'potential': corner_potential,
                        'cells_examined': cells_examined_list
                    })
        
        # Return results
        if collect_search_data:
            return (total_potential, {'corners': all_corners_data})
        else:
            return (total_potential, None)


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
                     color: PlayerColor, collect_visualization: bool = False) -> MoveEvaluation:
        """
        Evaluate a single move using all configured heuristics.
        
        If collect_visualization is True, collects corner expansion search data
        for UI visualization.
        """
        heuristic_breakdown = {}
        total_score = 0.0
        corner_search_data = None
        
        for heuristic_name, (heuristic_func, weight) in self.heuristics.items():
            try:
                # Special handling for corner_path_potential to collect visualization data
                if heuristic_name == "corner_path_potential" and collect_visualization:
                    heuristic_score, search_data = heuristic_func(
                        piece, game_state, color, row, col, collect_search_data=True
                    )
                    corner_search_data = search_data
                else:
                    # For corner_path_potential without visualization
                    if heuristic_name == "corner_path_potential":
                        result = heuristic_func(piece, game_state, color, row, col, collect_search_data=False)
                        heuristic_score = result[0] if isinstance(result, tuple) else result
                    else:
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
            heuristic_breakdown=heuristic_breakdown,
            corner_search_data=corner_search_data
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
    
    def choose_move_with_visualization(self, game_state: GameState, 
                                      color: PlayerColor) -> Optional[MoveEvaluation]:
        """
        Choose the best move and include visualization data for the selected move.
        This re-evaluates only the best move with visualization enabled.
        """
        # First, find the best move using normal evaluation
        best_move = self.choose_move(game_state, color)
        
        if not best_move:
            return None
        
        # Re-evaluate the best move with visualization data
        best_move_with_viz = self.evaluate_move(
            best_move.piece,
            best_move.piece_type,
            best_move.row,
            best_move.col,
            game_state,
            color,
            collect_visualization=True
        )
        
        return best_move_with_viz


# Enhanced Strategies
class OptimizedAIStrategy(AIStrategy):
    """Optimized AI with tuned weights from hyperparameter search (Champion Pool Baseline)"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("Champion Optimized AI")
        
        # 🏆 CHAMPION WEIGHTS - 48% Win Rate vs Adaptive Champions!
        # Updated: 2025-10-30 14:34:10 (GeniusTime diverse pool session)
        # Trained against 5 diverse champion archetypes with adaptive opponents
        # ULTRA-AGGRESSIVE strategy: Maximum opponent disruption & territorial control
        default_weights = {
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
        }
        
        weights = weights or default_weights
        
        # All 12 heuristics with uniform starting weights
        self.add_heuristic("piece_size", EnhancedHeuristic.piece_size_score, weights.get("piece_size", 1.0))
        self.add_heuristic("blocked_opponents", EnhancedHeuristic.blocked_opponents_score, weights.get("blocked_opponents", 1.0))
        self.add_heuristic("corner_control", EnhancedHeuristic.corner_control_score, weights.get("corner_control", 1.0))
        self.add_heuristic("compactness", EnhancedHeuristic.compactness_score, weights.get("compactness", 1.0))
        self.add_heuristic("mobility", EnhancedHeuristic.mobility_score, weights.get("mobility", 1.0))
        self.add_heuristic("opponent_restriction", EnhancedHeuristic.opponent_restriction_score, weights.get("opponent_restriction", 1.0))
        self.add_heuristic("endgame_optimization", EnhancedHeuristic.endgame_optimization_score, weights.get("endgame_optimization", 1.0))
        self.add_heuristic("corner_path_potential", EnhancedHeuristic.corner_path_potential_score, weights.get("corner_path_potential", 1.0))
        # Opponent-aware heuristics with equal starting weights
        self.add_heuristic("opponent_territory_pressure", EnhancedHeuristic.opponent_territory_pressure_score, weights.get("opponent_territory_pressure", 1.0))
        self.add_heuristic("opponent_mobility_restriction", EnhancedHeuristic.opponent_mobility_restriction_score, weights.get("opponent_mobility_restriction", 1.0))
        self.add_heuristic("opponent_threat_assessment", EnhancedHeuristic.opponent_threat_assessment_score, weights.get("opponent_threat_assessment", 1.0))
        self.add_heuristic("strategic_positioning", EnhancedHeuristic.strategic_positioning_score, weights.get("strategic_positioning", 1.0))
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with highest score"""
        if not evaluations:
            return None
        return max(evaluations, key=lambda e: e.score)


class AggressiveOptimizedStrategy(AIStrategy):
    """Aggressive AI focused on opponent disruption and territorial dominance"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("Aggressive Dominator AI")
        
        # Aggressive weights - MAXIMIZE opponent disruption and territory control
        default_weights = {
            "piece_size": 0.8,  # Less emphasis on piece size
            "blocked_opponents": 4.5,  # Very high opponent blocking
            "corner_control": 2.5,  # High board control
            # "compactness": REMOVED - aggressive play spreads out
            # "mobility": REMOVED - focus on offense, not self-preservation
            "opponent_restriction": 3.5,  # Very high opponent restriction
            "endgame_optimization": 0.8,  # Lower endgame optimization
            "corner_path_potential": 1.5,  # Moderate path potential (less conservative)
            # NEW: Aggressive opponent-aware heuristics
            "opponent_territory_pressure": 3.0,  # MAXIMUM pressure on opponents
            "opponent_mobility_restriction": 4.0,  # MAXIMUM mobility restriction
            "opponent_threat_assessment": 1.5,  # Target threatening opponents
            "strategic_positioning": 2.0  # Control key zones aggressively
        }
        
        weights = weights or default_weights
        
        # Aggressive heuristic set - focus on opponent disruption
        self.add_heuristic("piece_size", EnhancedHeuristic.piece_size_score, weights.get("piece_size", 0.8))
        self.add_heuristic("blocked_opponents", EnhancedHeuristic.blocked_opponents_score, weights.get("blocked_opponents", 4.5))
        self.add_heuristic("corner_control", EnhancedHeuristic.corner_control_score, weights.get("corner_control", 2.5))
        self.add_heuristic("opponent_restriction", EnhancedHeuristic.opponent_restriction_score, weights.get("opponent_restriction", 3.5))
        self.add_heuristic("endgame_optimization", EnhancedHeuristic.endgame_optimization_score, weights.get("endgame_optimization", 0.8))
        self.add_heuristic("corner_path_potential", EnhancedHeuristic.corner_path_potential_score, weights.get("corner_path_potential", 1.5))
        # Opponent-aware heuristics (aggressive values)
        self.add_heuristic("opponent_territory_pressure", EnhancedHeuristic.opponent_territory_pressure_score, weights.get("opponent_territory_pressure", 3.0))
        self.add_heuristic("opponent_mobility_restriction", EnhancedHeuristic.opponent_mobility_restriction_score, weights.get("opponent_mobility_restriction", 4.0))
        self.add_heuristic("opponent_threat_assessment", EnhancedHeuristic.opponent_threat_assessment_score, weights.get("opponent_threat_assessment", 1.5))
        self.add_heuristic("strategic_positioning", EnhancedHeuristic.strategic_positioning_score, weights.get("strategic_positioning", 2.0))
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with highest score, with strong preference for opponent disruption"""
        if not evaluations:
            return None
        
        # Sort by total score, then by opponent disruption metrics for tie-breaking
        return max(evaluations, key=lambda e: (
            e.score, 
            e.heuristic_breakdown.get("blocked_opponents", 0) + 
            e.heuristic_breakdown.get("opponent_restriction", 0) +
            e.heuristic_breakdown.get("opponent_mobility_restriction", 0) +
            e.heuristic_breakdown.get("opponent_territory_pressure", 0)
        ))


class BalancedOptimizedStrategy(AIStrategy):
    """Balanced AI with equal emphasis on offense and defense"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("Balanced Strategist AI")
        
        # Balanced weights - moderate values across all heuristics
        default_weights = {
            "piece_size": 1.5,  # Good piece size preference
            "blocked_opponents": 1.8,  # Moderate opponent blocking
            "corner_control": 1.8,  # Moderate board control
            "compactness": 1.2,  # Good compactness
            "mobility": 1.5,  # High mobility preservation
            "opponent_restriction": 1.2,  # Moderate opponent restriction
            "endgame_optimization": 1.8,  # High endgame awareness
            "corner_path_potential": 2.5,  # High path potential for flexibility
            # NEW: Balanced opponent-aware heuristics
            "opponent_territory_pressure": 1.2,  # Moderate pressure
            "opponent_mobility_restriction": 1.5,  # Moderate restriction
            "opponent_threat_assessment": 1.5,  # Good threat awareness
            "strategic_positioning": 2.0  # Strong positioning
        }
        
        weights = weights or default_weights
        
        # Same heuristics as optimized, but with balanced weights
        self.add_heuristic("piece_size", EnhancedHeuristic.piece_size_score, weights.get("piece_size", 1.5))
        self.add_heuristic("blocked_opponents", EnhancedHeuristic.blocked_opponents_score, weights.get("blocked_opponents", 1.8))
        self.add_heuristic("corner_control", EnhancedHeuristic.corner_control_score, weights.get("corner_control", 1.8))
        self.add_heuristic("compactness", EnhancedHeuristic.compactness_score, weights.get("compactness", 1.2))
        self.add_heuristic("mobility", EnhancedHeuristic.mobility_score, weights.get("mobility", 1.5))
        self.add_heuristic("opponent_restriction", EnhancedHeuristic.opponent_restriction_score, weights.get("opponent_restriction", 1.2))
        self.add_heuristic("endgame_optimization", EnhancedHeuristic.endgame_optimization_score, weights.get("endgame_optimization", 1.8))
        self.add_heuristic("corner_path_potential", EnhancedHeuristic.corner_path_potential_score, weights.get("corner_path_potential", 2.5))
        # Opponent-aware heuristics with balanced weights
        self.add_heuristic("opponent_territory_pressure", EnhancedHeuristic.opponent_territory_pressure_score, weights.get("opponent_territory_pressure", 1.2))
        self.add_heuristic("opponent_mobility_restriction", EnhancedHeuristic.opponent_mobility_restriction_score, weights.get("opponent_mobility_restriction", 1.5))
        self.add_heuristic("opponent_threat_assessment", EnhancedHeuristic.opponent_threat_assessment_score, weights.get("opponent_threat_assessment", 1.5))
        self.add_heuristic("strategic_positioning", EnhancedHeuristic.strategic_positioning_score, weights.get("strategic_positioning", 2.0))
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with highest score"""
        if not evaluations:
            return None
        return max(evaluations, key=lambda e: e.score)


class DefensiveOptimizedStrategy(AIStrategy):
    """Defensive AI focused on self-preservation and adaptive play"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("Defensive Survivor AI")
        
        # Defensive weights - emphasize self-preservation and adaptability
        default_weights = {
            "piece_size": 2.0,  # High piece size preference (get pieces out)
            "blocked_opponents": 0.8,  # Lower opponent blocking (less aggressive)
            "corner_control": 1.2,  # Moderate board control
            "compactness": 2.0,  # High compactness (build strong territories)
            "mobility": 2.5,  # Very high mobility preservation
            "opponent_restriction": 0.6,  # Lower opponent restriction
            "endgame_optimization": 2.5,  # Very high endgame awareness
            "corner_path_potential": 4.0,  # Very high path potential for flexibility
            "opponent_territory_pressure": 0.5,  # Low pressure (defensive, not aggressive)
            "opponent_mobility_restriction": 0.8,  # Low restriction (focus on own mobility)
            "opponent_threat_assessment": 2.0,  # High threat awareness (defensive priority)
            "strategic_positioning": 2.0  # High positioning (secure key areas)
        }
        
        weights = weights or default_weights
        
        # All heuristics with defensive weights - emphasize self-preservation
        self.add_heuristic("piece_size", EnhancedHeuristic.piece_size_score, weights.get("piece_size", 2.0))
        self.add_heuristic("blocked_opponents", EnhancedHeuristic.blocked_opponents_score, weights.get("blocked_opponents", 0.8))
        self.add_heuristic("corner_control", EnhancedHeuristic.corner_control_score, weights.get("corner_control", 1.2))
        self.add_heuristic("compactness", EnhancedHeuristic.compactness_score, weights.get("compactness", 2.0))
        self.add_heuristic("mobility", EnhancedHeuristic.mobility_score, weights.get("mobility", 2.5))
        self.add_heuristic("opponent_restriction", EnhancedHeuristic.opponent_restriction_score, weights.get("opponent_restriction", 0.6))
        self.add_heuristic("endgame_optimization", EnhancedHeuristic.endgame_optimization_score, weights.get("endgame_optimization", 2.5))
        self.add_heuristic("corner_path_potential", EnhancedHeuristic.corner_path_potential_score, weights.get("corner_path_potential", 4.0))
        # Opponent-aware heuristics with defensive focus
        self.add_heuristic("opponent_territory_pressure", EnhancedHeuristic.opponent_territory_pressure_score, weights.get("opponent_territory_pressure", 0.5))
        self.add_heuristic("opponent_mobility_restriction", EnhancedHeuristic.opponent_mobility_restriction_score, weights.get("opponent_mobility_restriction", 0.8))
        self.add_heuristic("opponent_threat_assessment", EnhancedHeuristic.opponent_threat_assessment_score, weights.get("opponent_threat_assessment", 2.0))
        self.add_heuristic("strategic_positioning", EnhancedHeuristic.strategic_positioning_score, weights.get("strategic_positioning", 2.0))
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with preference for mobility and path potential"""
        if not evaluations:
            return None
        
        # Sort by total score, then by mobility and path potential for tie-breaking
        return max(evaluations, key=lambda e: (
            e.score,
            e.heuristic_breakdown.get("mobility", 0) + 
            e.heuristic_breakdown.get("corner_path_potential", 0) +
            e.heuristic_breakdown.get("endgame_optimization", 0)
        ))


class CasualAIStrategy(AIStrategy):
    """Casual AI for beginners - simple strategy with basic heuristics and low weights"""
    
    def __init__(self, weights: Optional[Dict[str, float]] = None):
        super().__init__("Casual Beginner AI")
        
        # Casual weights - very basic strategy with low values
        # Only uses simple heuristics to provide a gentle challenge
        default_weights = {
            "piece_size": 1.0,  # Basic piece size preference
            "corner_control": 0.5,  # Minimal board control
            "compactness": 0.3,  # Some territory building
            "corner_path_potential": 0.8,  # Basic path awareness
            "endgame_optimization": 0.6,  # Minimal endgame planning
            # Minimal opponent-aware behavior for a more casual experience
            "opponent_territory_pressure": 0.1,
            "strategic_positioning": 0.3
        }
        
        weights = weights or default_weights
        
        # Limited heuristic set - only basic strategies for casual play
        self.add_heuristic("piece_size", EnhancedHeuristic.piece_size_score, weights.get("piece_size", 1.0))
        self.add_heuristic("corner_control", EnhancedHeuristic.corner_control_score, weights.get("corner_control", 0.5))
        self.add_heuristic("compactness", EnhancedHeuristic.compactness_score, weights.get("compactness", 0.3))
        self.add_heuristic("corner_path_potential", EnhancedHeuristic.corner_path_potential_score, weights.get("corner_path_potential", 0.8))
        self.add_heuristic("endgame_optimization", EnhancedHeuristic.endgame_optimization_score, weights.get("endgame_optimization", 0.6))
        self.add_heuristic("opponent_territory_pressure", EnhancedHeuristic.opponent_territory_pressure_score, weights.get("opponent_territory_pressure", 0.1))
        self.add_heuristic("strategic_positioning", EnhancedHeuristic.strategic_positioning_score, weights.get("strategic_positioning", 0.3))
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select move with highest score, but with some randomness for casual play"""
        if not evaluations:
            return None
        
        # Sort evaluations by score
        sorted_evals = sorted(evaluations, key=lambda e: e.score, reverse=True)
        
        # Add some randomness - sometimes pick from top 3 moves instead of always the best
        import random
        if len(sorted_evals) >= 3 and random.random() < 0.3:  # 30% chance to not pick optimal
            return random.choice(sorted_evals[:3])
        else:
            return sorted_evals[0]


class RandomAIStrategy(AIStrategy):
    """Random AI strategy that selects moves randomly without any heuristics"""
    
    def __init__(self):
        """Initialize random AI strategy"""
        super().__init__("Random AI")  # Pass name string as required
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """Select a completely random move from available options"""
        if not evaluations:
            return None
        # Return a random move without any evaluation
        return random.choice(evaluations)
    
    def choose_move(self, game_state: GameState, color: PlayerColor) -> Optional[MoveEvaluation]:
        """Choose a random valid move"""
        possible_moves = self.get_all_possible_moves(game_state, color)
        
        if not possible_moves:
            return None
        
        # Return a random move
        return random.choice(possible_moves)
    
    def choose_move_with_visualization(self, game_state: GameState, color: PlayerColor) -> Optional[MoveEvaluation]:
        """Choose a random valid move (same as choose_move for random strategy)"""
        return self.choose_move(game_state, color)
    
    def get_all_possible_moves(self, game_state: GameState, color: PlayerColor) -> List[MoveEvaluation]:
        """Get all possible moves with random scores"""
        player = game_state.get_player(color)
        if not player:
            return []
        
        moves = []
        
        # For each available piece (these are pieces the player still has)
        for piece_obj in player.available_pieces:
            piece_type = piece_obj.piece_type
            
            # Get all valid moves for this piece
            valid_moves = game_state.get_valid_moves_for_piece(piece_type, color)
            
            # Evaluate each valid move with random score
            for row, col, oriented_piece in valid_moves:
                # Assign random score between 0 and 1
                score = random.random()
                
                moves.append(MoveEvaluation(
                    piece_type=piece_type,
                    row=row,
                    col=col,
                    piece=oriented_piece,
                    score=score,
                    heuristic_breakdown={"random": score}
                ))
        
        return moves


class AIPlayer:
    """Main AI Player class"""
    
    def __init__(self, color: PlayerColor, strategy: AIStrategy):
        self.color = color
        self.strategy = strategy
    
    def get_move(self, game_state: GameState) -> Optional[MoveEvaluation]:
        """Get the AI's chosen move"""
        return self.strategy.choose_move(game_state, self.color)
    
    def get_move_with_visualization(self, game_state: GameState) -> Optional[MoveEvaluation]:
        """Get the AI's chosen move with corner expansion visualization data"""
        return self.strategy.choose_move_with_visualization(game_state, self.color)
    
    def get_all_evaluated_moves(self, game_state: GameState) -> List[MoveEvaluation]:
        """Get all possible moves with evaluations"""
        all_moves = self.strategy.get_all_possible_moves(game_state, self.color)
        return sorted(all_moves, key=lambda e: e.score, reverse=True)


def create_ai_player(color: PlayerColor, strategy_name: str = "optimized") -> AIPlayer:
    """Factory function to create AI players with diverse optimized strategies"""
    
    # Map strategy names to new optimized strategy classes
    strategy_mapping = {
        "optimized": OptimizedAIStrategy,
        "aggressive": AggressiveOptimizedStrategy,
        "balanced": BalancedOptimizedStrategy,
        "defensive": DefensiveOptimizedStrategy,
        "casual": CasualAIStrategy,
        "random": RandomAIStrategy
    }
    
    # Handle MCTS separately
    if strategy_name.lower() == "mcts":
        try:
            from mcts_ai import MCTSAIStrategy
            strategy = MCTSAIStrategy(time_limit=15.0, max_iterations=1000)
        except ImportError:
            # Fallback to optimized if MCTS not available
            strategy = OptimizedAIStrategy()
    else:
        # Get strategy class, default to optimized
        strategy_class = strategy_mapping.get(strategy_name.lower(), OptimizedAIStrategy)
        strategy = strategy_class()
    
    return AIPlayer(color, strategy)


def difficulty_level_to_strategy(difficulty: int) -> str:
    """Convert difficulty level (1-6) to strategy name"""
    difficulty_mapping = {
        1: "random",      # Beginner
        2: "casual",      # Casual  
        3: "balanced",    # Intermediate
        4: "aggressive",  # Advanced
        5: "optimized",   # Expert
        6: "mcts"        # Master
    }
    return difficulty_mapping.get(difficulty, "balanced")


def strategy_to_difficulty_level(strategy: str) -> int:
    """Convert strategy name to difficulty level (1-6)"""
    strategy_mapping = {
        "random": 1,      # Beginner
        "casual": 2,      # Casual
        "balanced": 3,    # Intermediate  
        "aggressive": 4,  # Advanced
        "optimized": 5,   # Expert
        "mcts": 6        # Master
    }
    return strategy_mapping.get(strategy.lower(), 3)


def get_difficulty_info(difficulty: int) -> dict:
    """Get difficulty level information including name and description"""
    difficulty_info = {
        1: {
            "name": "Beginner",
            "description": "Random moves - great for learning the rules",
            "stars": "★☆☆☆☆☆"
        },
        2: {
            "name": "Casual", 
            "description": "Simple strategy with some randomness",
            "stars": "★★☆☆☆☆"
        },
        3: {
            "name": "Intermediate",
            "description": "Balanced approach with solid fundamentals", 
            "stars": "★★★☆☆☆"
        },
        4: {
            "name": "Advanced",
            "description": "Aggressive play focused on disruption",
            "stars": "★★★★☆☆"
        },
        5: {
            "name": "Expert",
            "description": "Champion-level AI with optimized strategy",
            "stars": "★★★★★☆"
        },
        6: {
            "name": "Master",
            "description": "Monte Carlo Tree Search - ultimate challenge",
            "stars": "★★★★★★"
        }
    }
    return difficulty_info.get(difficulty, difficulty_info[3])
