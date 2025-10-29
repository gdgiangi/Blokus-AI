"""
Blokus Game State
Main game state class that manages the entire Blokus game.
"""

from typing import List, Dict, Optional, Tuple
from enum import Enum
from board import Board, PlayerColor
from player import Player
from pieces import Piece, PieceType


class GamePhase(Enum):
    """Represents the current phase of the game"""
    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    FINISHED = "finished"


class GameState:
    """
    Main game state class that manages the Blokus game.
    Handles game initialization, turn management, move validation, and scoring.
    """
    
    # Turn order is always: Blue -> Yellow -> Red -> Green
    TURN_ORDER = [PlayerColor.BLUE, PlayerColor.YELLOW, PlayerColor.RED, PlayerColor.GREEN]
    
    def __init__(self, num_players: int = 4):
        """
        Initialize a new Blokus game.
        
        Args:
            num_players: Number of players (2, 3, or 4)
        """
        if num_players < 2 or num_players > 4:
            raise ValueError("Number of players must be between 2 and 4")
        
        self.num_players = num_players
        self.board = Board()
        self.phase = GamePhase.NOT_STARTED
        
        # Initialize players based on turn order
        self.players: Dict[PlayerColor, Player] = {}
        for i in range(num_players):
            color = self.TURN_ORDER[i]
            self.players[color] = Player(color)
        
        self.current_player_index = 0
        self.turn_number = 0
        self.move_history: List[Dict] = []
        
    def reset(self, num_players: Optional[int] = None):
        """
        Reset the game to initial state.
        
        Args:
            num_players: Optional new number of players (uses current if not provided)
        """
        if num_players is not None:
            if num_players < 2 or num_players > 4:
                raise ValueError("Number of players must be between 2 and 4")
            self.num_players = num_players
            
            # Reinitialize players
            self.players = {}
            for i in range(num_players):
                color = self.TURN_ORDER[i]
                self.players[color] = Player(color)
        else:
            # Reset existing players
            for player in self.players.values():
                player.reset()
        
        self.board.reset()
        self.phase = GamePhase.NOT_STARTED
        self.current_player_index = 0
        self.turn_number = 0
        self.move_history = []
    
    def start_game(self):
        """Start the game"""
        if self.phase != GamePhase.NOT_STARTED:
            raise RuntimeError("Game has already started")
        self.phase = GamePhase.IN_PROGRESS
        self.turn_number = 1
    
    def get_current_player(self) -> Player:
        """Get the player whose turn it is"""
        active_colors = list(self.players.keys())
        return self.players[active_colors[self.current_player_index % len(active_colors)]]
    
    def get_current_color(self) -> PlayerColor:
        """Get the color of the current player"""
        return self.get_current_player().color
    
    def get_player(self, color: PlayerColor) -> Optional[Player]:
        """Get a specific player by color"""
        return self.players.get(color)
    
    def next_turn(self):
        """Advance to the next player's turn"""
        active_colors = list(self.players.keys())
        self.current_player_index = (self.current_player_index + 1) % len(active_colors)
        
        # If we've cycled back to the first player, increment turn number
        if self.current_player_index == 0:
            self.turn_number += 1
        
        # Check if game is over
        if self.is_game_over():
            self.phase = GamePhase.FINISHED
    
    def can_place_piece(self, piece: Piece, row: int, col: int, 
                       color: Optional[PlayerColor] = None) -> Tuple[bool, str]:
        """
        Check if a piece can be placed at the given position.
        
        Args:
            piece: The piece to place
            row: Row position for the piece's reference point
            col: Column position for the piece's reference point
            color: Player color (uses current player if not specified)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if color is None:
            color = self.get_current_color()
        
        # Translate piece to absolute coordinates
        coordinates = piece.translate(row, col)
        
        # Use board's validation logic
        return self.board.can_place_piece(coordinates, color)
    
    def place_piece(self, piece_type: PieceType, piece: Piece, row: int, col: int) -> bool:
        """
        Place a piece on the board for the current player.
        
        Args:
            piece_type: The type of piece being placed
            piece: The piece instance (with orientation)
            row: Row position
            col: Column position
        
        Returns:
            True if placement was successful, False otherwise
        """
        if self.phase != GamePhase.IN_PROGRESS:
            return False
        
        current_player = self.get_current_player()
        
        # Check if player has this piece
        if not current_player.has_piece(piece_type):
            return False
        
        # Validate placement
        can_place, error = self.can_place_piece(piece, row, col)
        if not can_place:
            return False
        
        # Place the piece
        coordinates = piece.translate(row, col)
        self.board.place_piece(coordinates, current_player.color)
        
        # Mark corner as used if this is first move
        if self.board.is_first_move(current_player.color):
            self.board.mark_corner_used(current_player.color)
        
        # Remove piece from player's available pieces
        current_player.play_piece(piece_type)
        
        # Record move in history
        self.move_history.append({
            'turn': self.turn_number,
            'color': current_player.color,
            'piece_type': piece_type,
            'position': (row, col),
            'coordinates': coordinates,
            'piece_shape': piece.get_coordinates()
        })
        
        # Advance to next turn
        self.next_turn()
        
        return True
    
    def pass_turn(self) -> bool:
        """
        Current player passes their turn.
        
        Returns:
            True if pass was successful
        """
        if self.phase != GamePhase.IN_PROGRESS:
            return False
        
        current_player = self.get_current_player()
        current_player.pass_turn()
        
        # Record pass in history
        self.move_history.append({
            'turn': self.turn_number,
            'color': current_player.color,
            'action': 'pass'
        })
        
        # Advance to next turn
        self.next_turn()
        
        return True
    
    def is_game_over(self) -> bool:
        """
        Check if the game is over.
        Game ends when all players have passed or no player can make a valid move.
        """
        # All players have passed
        if all(player.has_passed for player in self.players.values()):
            return True
        
        # All players are out of pieces
        if all(not player.has_pieces_remaining() for player in self.players.values()):
            return True
        
        return False
    
    def get_scores(self) -> Dict[PlayerColor, int]:
        """
        Get the current scores for all players.
        
        Returns:
            Dictionary mapping player color to score
        """
        return {color: player.calculate_score() 
                for color, player in self.players.items()}
    
    def get_winner(self) -> Optional[PlayerColor]:
        """
        Get the winner of the game.
        Returns None if game is not over or there's a tie.
        """
        if not self.is_game_over():
            return None
        
        scores = self.get_scores()
        max_score = max(scores.values())
        
        # Check for tie
        winners = [color for color, score in scores.items() if score == max_score]
        if len(winners) > 1:
            return None  # Tie
        
        return winners[0]
    
    def get_rankings(self) -> List[Tuple[PlayerColor, int]]:
        """
        Get player rankings sorted by score (highest to lowest).
        
        Returns:
            List of tuples (color, score) sorted by score
        """
        scores = self.get_scores()
        return sorted(scores.items(), key=lambda x: x[1], reverse=True)
    
    def get_valid_moves_for_piece(self, piece_type: PieceType, 
                                  color: Optional[PlayerColor] = None) -> List[Tuple[int, int, Piece]]:
        """
        Get all valid positions where a piece can be placed for a player.
        
        Args:
            piece_type: The type of piece to check
            color: Player color (uses current player if not specified)
        
        Returns:
            List of tuples (row, col, oriented_piece) representing valid placements
        """
        if color is None:
            color = self.get_current_color()
        
        player = self.get_player(color)
        if not player or not player.has_piece(piece_type):
            return []
        
        piece = player.get_piece(piece_type)
        if not piece:
            return []
        
        valid_moves = []
        
        # Try all orientations
        orientations = piece.get_all_orientations()
        unique_pieces = [Piece(piece_type, list(coords)) for coords in orientations]
        
        # Try all positions on the board
        for row in range(Board.BOARD_SIZE):
            for col in range(Board.BOARD_SIZE):
                for oriented_piece in unique_pieces:
                    can_place, _ = self.can_place_piece(oriented_piece, row, col, color)
                    if can_place:
                        valid_moves.append((row, col, oriented_piece))
        
        return valid_moves
    
    def has_valid_moves(self, color: Optional[PlayerColor] = None) -> bool:
        """
        Check if a player has any valid moves.
        
        Args:
            color: Player color (uses current player if not specified)
        
        Returns:
            True if player has at least one valid move
        """
        if color is None:
            color = self.get_current_color()
        
        player = self.get_player(color)
        if not player or not player.has_pieces_remaining():
            return False
        
        # Check each available piece
        for piece in player.available_pieces:
            moves = self.get_valid_moves_for_piece(piece.piece_type, color)
            if moves:
                return True
        
        return False
    
    def to_dict(self) -> dict:
        """
        Convert game state to dictionary for serialization.
        Useful for web interface communication.
        """
        return {
            'phase': self.phase.value,
            'num_players': self.num_players,
            'turn_number': self.turn_number,
            'current_player': self.get_current_color().value,
            'board': self.board.to_dict(),
            'players': {color.value: player.to_dict() 
                       for color, player in self.players.items()},
            'scores': {color.value: score 
                      for color, score in self.get_scores().items()},
            'is_game_over': self.is_game_over(),
            'winner': self.get_winner().value if self.get_winner() is not None else None,
            'rankings': [(color.value, score) for color, score in self.get_rankings()]
        }
    
    def __str__(self) -> str:
        """String representation of game state"""
        lines = [
            f"Blokus Game - {self.phase.value}",
            f"Players: {self.num_players}",
            f"Turn: {self.turn_number}",
            f"Current Player: {self.get_current_color().value}",
            "\nScores:"
        ]
        
        for color, score in self.get_rankings():
            player = self.players[color]
            lines.append(f"  {color.value}: {score} "
                        f"(pieces left: {len(player.available_pieces)})")
        
        return '\n'.join(lines)
