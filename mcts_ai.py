"""
Monte Carlo Tree Search AI for Blokus
Implements MCTS from scratch with the four core phases: selection, expansion, simulation, and backpropagation.
Uses existing heuristics for rapid evaluation and playouts.
"""

import math
import random
import time
import copy
from typing import List, Dict, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass
from abc import ABC, abstractmethod

from board import Board, PlayerColor
from pieces import Piece, PieceType
from game_state import GameState
from ai_player_enhanced import EnhancedHeuristic, MoveEvaluation, AIStrategy


@dataclass
class MCTSStats:
    """Statistics for MCTS progress tracking"""
    nodes_explored: int = 0
    simulations_run: int = 0
    current_depth: int = 0
    best_move_visits: int = 0
    time_elapsed: float = 0.0
    
    def to_dict(self):
        return {
            'nodes_explored': self.nodes_explored,
            'simulations_run': self.simulations_run,
            'current_depth': self.current_depth,
            'best_move_visits': self.best_move_visits,
            'time_elapsed': self.time_elapsed
        }


class MCTSNode:
    """Node in the Monte Carlo Tree Search tree"""
    
    def __init__(self, game_state: GameState, parent: Optional['MCTSNode'] = None, 
                 move: Optional[MoveEvaluation] = None, player_color: Optional[PlayerColor] = None):
        self.game_state = copy.deepcopy(game_state)
        self.parent = parent
        self.move = move  # The move that led to this state
        self.player_color = player_color  # Player who made the move to reach this state
        
        # MCTS statistics
        self.visits = 0
        self.wins = 0.0  # Total reward accumulated
        self.children: Dict[str, 'MCTSNode'] = {}
        self.untried_moves: List[MoveEvaluation] = []
        self.is_fully_expanded = False
        self.is_terminal = False
        
        # Initialize untried moves for current player
        self._initialize_moves()
    
    def _initialize_moves(self):
        """Initialize list of untried moves for the current player"""
        if self.game_state.is_game_over():
            self.is_terminal = True
            self.untried_moves = []
            return
        
        current_color = self.game_state.get_current_color()
        
        # Use existing heuristic evaluation to get all possible moves
        strategy = OptimizedHeuristicStrategy()
        all_moves = strategy.get_all_possible_moves(self.game_state, current_color)
        
        # Sort by heuristic score for better move ordering
        all_moves.sort(key=lambda x: x.score, reverse=True)
        
        # Limit moves for performance (top 50 moves per turn)
        self.untried_moves = all_moves[:50]
        
        if not self.untried_moves:
            # No moves available, this might be a pass situation
            self.is_terminal = True
    
    def get_move_key(self, move: MoveEvaluation) -> str:
        """Generate unique key for a move"""
        return f"{move.piece_type.value}_{move.row}_{move.col}_{hash(tuple(tuple(coord) for coord in move.piece.get_coordinates()))}"
    
    def is_leaf(self) -> bool:
        """Check if this is a leaf node (not fully expanded)"""
        return len(self.untried_moves) > 0 or not self.children
    
    def select_child(self, exploration_constant: float = 1.414) -> Optional['MCTSNode']:
        """Select child using UCB1 formula"""
        if not self.children:
            return None
        
        best_value = -float('inf')
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                ucb_value = float('inf')
            else:
                # UCB1 formula
                exploitation = child.wins / child.visits
                exploration = exploration_constant * math.sqrt(math.log(self.visits) / child.visits)
                ucb_value = exploitation + exploration
            
            if ucb_value > best_value:
                best_value = ucb_value
                best_child = child
        
        return best_child
    
    def expand(self) -> Optional['MCTSNode']:
        """Expand the tree by adding a new child node"""
        if not self.untried_moves:
            self.is_fully_expanded = True
            return None
        
        # Select next untried move
        move = self.untried_moves.pop()
        
        # Create new game state with this move
        new_game_state = copy.deepcopy(self.game_state)
        
        # Execute the move
        success = new_game_state.place_piece(
            move.piece_type,
            move.piece,
            move.row,
            move.col
        )
        
        if not success:
            # Invalid move, try next one
            return self.expand() if self.untried_moves else None
        
        # Create child node
        move_key = self.get_move_key(move)
        child = MCTSNode(new_game_state, parent=self, move=move, 
                        player_color=self.game_state.get_current_color())
        self.children[move_key] = child
        
        if not self.untried_moves:
            self.is_fully_expanded = True
        
        return child
    
    def simulate(self, max_depth: int = 10) -> float:
        """Run a random playout simulation"""
        simulation_state = copy.deepcopy(self.game_state)
        depth = 0
        original_color = self.game_state.get_current_color()
        
        # Quick heuristic strategy for simulation
        heuristic_strategy = FastHeuristicStrategy()
        
        while not simulation_state.is_game_over() and depth < max_depth:
            current_color = simulation_state.get_current_color()
            
            # Get possible moves using fast heuristic
            possible_moves = heuristic_strategy.get_all_possible_moves(simulation_state, current_color)
            
            if not possible_moves:
                # No moves, pass turn
                simulation_state.pass_turn()
            else:
                # Select move with some randomness but bias toward good moves
                if len(possible_moves) > 5:
                    # Select from top 5 moves with weighted probability
                    weights = [5, 4, 3, 2, 1]
                    selected_moves = possible_moves[:5]
                    move = random.choices(selected_moves, weights=weights)[0]
                else:
                    # If few moves, select best one
                    move = possible_moves[0]
                
                # Execute move
                success = simulation_state.place_piece(
                    move.piece_type,
                    move.piece,
                    move.row,
                    move.col
                )
                
                if not success:
                    simulation_state.pass_turn()
            
            depth += 1
        
        # Evaluate final position
        return self._evaluate_position(simulation_state, original_color)
    
    def _evaluate_position(self, game_state: GameState, player_color: PlayerColor) -> float:
        """Evaluate the final position for the given player"""
        if not game_state.is_game_over():
            # Game not over, use heuristic evaluation
            player = game_state.get_player(player_color)
            if not player:
                return 0.0
            
            # Score based on pieces remaining (fewer is better)
            pieces_remaining = len(player.available_pieces)
            total_squares_remaining = sum(piece.size() for piece in player.available_pieces)
            
            # Normalize to 0-1 range
            max_squares = 89  # Total squares in all pieces
            score = 1.0 - (total_squares_remaining / max_squares)
            
            return score
        
        # Game is over, get final ranking
        final_rankings = game_state.get_rankings()
        player_ranking = None
        
        for i, (color, score) in enumerate(final_rankings):
            if color == player_color:
                player_ranking = i
                break
        
        if player_ranking is None:
            return 0.0
        
        # Convert ranking to reward (1st = 1.0, 2nd = 0.7, 3rd = 0.3, 4th = 0.1)
        ranking_rewards = [1.0, 0.7, 0.3, 0.1]
        return ranking_rewards[min(player_ranking, len(ranking_rewards) - 1)]
    
    def backpropagate(self, reward: float):
        """Backpropagate the simulation result"""
        self.visits += 1
        self.wins += reward
        
        if self.parent:
            # Reward is from perspective of player who made the move to reach this node
            self.parent.backpropagate(reward)
    
    def get_best_child(self) -> Optional['MCTSNode']:
        """Get the child with the highest visit count (most promising)"""
        if not self.children:
            return None
        
        return max(self.children.values(), key=lambda x: x.visits)
    
    def get_win_rate(self) -> float:
        """Get the win rate for this node"""
        return self.wins / self.visits if self.visits > 0 else 0.0


class OptimizedHeuristicStrategy(AIStrategy):
    """Fast heuristic strategy for move evaluation in MCTS"""
    
    def __init__(self):
        super().__init__("Optimized Heuristic")
        
        # Balanced weights for quick evaluation
        self.add_heuristic("piece_size", EnhancedHeuristic.piece_size_score, 1.5)
        self.add_heuristic("new_paths", EnhancedHeuristic.new_paths_score, 2.0)
        self.add_heuristic("blocked_opponents", EnhancedHeuristic.blocked_opponents_score, 1.8)
        self.add_heuristic("corner_control", EnhancedHeuristic.corner_control_score, 1.2)
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        if not evaluations:
            return None
        return max(evaluations, key=lambda e: e.score)


class FastHeuristicStrategy(AIStrategy):
    """Minimal heuristic strategy for fast simulations"""
    
    def __init__(self):
        super().__init__("Fast Heuristic")
        
        # Only essential heuristics for speed
        self.add_heuristic("piece_size", EnhancedHeuristic.piece_size_score, 2.0)
        self.add_heuristic("new_paths", EnhancedHeuristic.new_paths_score, 1.5)
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        if not evaluations:
            return None
        return max(evaluations, key=lambda e: e.score)


class MCTSAIStrategy(AIStrategy):
    """Monte Carlo Tree Search AI Strategy for Blokus"""
    
    def __init__(self, time_limit: float = 5.0, max_iterations: int = 1000, 
                 exploration_constant: float = 1.414, progress_callback=None):
        super().__init__("MCTS AI")
        
        self.time_limit = time_limit
        self.max_iterations = max_iterations
        self.exploration_constant = exploration_constant
        self.progress_callback = progress_callback  # For UI updates
        
        # Statistics
        self.stats = MCTSStats()
        self.last_tree_size = 0
        
        # Global progress storage for Flask app access
        self._global_progress = None
        
    def choose_move(self, game_state: GameState, color: PlayerColor) -> Optional[MoveEvaluation]:
        """Main MCTS algorithm implementation"""
        
        # Reset statistics
        self.stats = MCTSStats()
        start_time = time.time()
        
        # Create root node
        root = MCTSNode(game_state)
        
        # If no moves available, return None
        if root.is_terminal:
            return None
        
        iteration = 0
        
        # Main MCTS loop
        while (time.time() - start_time < self.time_limit and 
               iteration < self.max_iterations):
            
            # 1. Selection: Find leaf node using tree policy
            node = self._selection(root)
            
            # 2. Expansion: Add new child if not terminal
            if not node.is_terminal and node.untried_moves:
                node = node.expand()
                if node is None:
                    iteration += 1
                    continue
            
            # 3. Simulation: Run random playout
            reward = node.simulate()
            
            # 4. Backpropagation: Update statistics
            node.backpropagate(reward)
            
            # Update statistics
            self.stats.simulations_run += 1
            self.stats.time_elapsed = time.time() - start_time
            
            # Progress callback for UI updates
            if iteration % 5 == 0:  # Update more frequently
                self.stats.nodes_explored = len(self._get_all_nodes(root))
                best_child = root.get_best_child()
                if best_child:
                    self.stats.best_move_visits = best_child.visits
                    self.stats.current_depth = self._get_tree_depth(root)
                
                # Store for global access
                self._global_progress = self.stats.to_dict()
                
                if self.progress_callback:
                    self.progress_callback(self.stats)
            
            iteration += 1
        
        # Final statistics
        self.stats.nodes_explored = len(self._get_all_nodes(root))
        self.stats.time_elapsed = time.time() - start_time
        
        # Select best move
        best_child = root.get_best_child()
        
        if best_child and best_child.move:
            self.stats.best_move_visits = best_child.visits
            if self.progress_callback:
                self.progress_callback(self.stats)
            return best_child.move
        
        return None
    
    def _selection(self, root: MCTSNode) -> MCTSNode:
        """Selection phase: traverse tree using UCB1 until leaf node"""
        node = root
        depth = 0
        
        while not node.is_leaf() and not node.is_terminal:
            node = node.select_child(self.exploration_constant)
            depth += 1
            
            if node is None:
                break
        
        self.stats.current_depth = max(self.stats.current_depth, depth)
        return node if node else root
    
    def _get_all_nodes(self, root: MCTSNode) -> List[MCTSNode]:
        """Get all nodes in the tree for statistics"""
        nodes = [root]
        for child in root.children.values():
            nodes.extend(self._get_all_nodes(child))
        return nodes
    
    def _get_tree_depth(self, root: MCTSNode) -> int:
        """Get maximum depth of the tree"""
        if not root.children:
            return 0
        return 1 + max(self._get_tree_depth(child) for child in root.children.values())
    
    def get_current_progress(self) -> Optional[Dict]:
        """Get current MCTS progress for external access"""
        return self._global_progress
    
    def select_move(self, evaluations: List[MoveEvaluation]) -> Optional[MoveEvaluation]:
        """This method is not used in MCTS - choose_move handles everything"""
        if not evaluations:
            return None
        return evaluations[0]
    
    def get_all_possible_moves(self, game_state: GameState, color: PlayerColor) -> List[MoveEvaluation]:
        """Get all possible moves (used by parent class)"""
        strategy = OptimizedHeuristicStrategy()
        return strategy.get_all_possible_moves(game_state, color)


def create_mcts_ai_player(color: PlayerColor, time_limit: float = 3.0, 
                         progress_callback: Optional[Callable] = None):
    """Factory function to create MCTS AI player"""
    # Import here to avoid circular imports
    from ai_player_enhanced import AIPlayer
    
    strategy = MCTSAIStrategy(
        time_limit=time_limit,
        max_iterations=800,
        exploration_constant=1.414,
        progress_callback=progress_callback
    )
    
    return AIPlayer(color, strategy)