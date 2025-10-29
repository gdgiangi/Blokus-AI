"""
Blokus Player
Represents a player in the Blokus game with their pieces and state.
"""

from typing import List, Set, Optional
from dataclasses import dataclass, field
from pieces import Piece, PieceType, create_all_pieces
from board import PlayerColor


@dataclass
class Player:
    """
    Represents a player in the Blokus game.
    Tracks the player's color, available pieces, and game statistics.
    """
    color: PlayerColor
    available_pieces: List[Piece] = field(default_factory=list)
    played_pieces: List[PieceType] = field(default_factory=list)
    has_passed: bool = False
    
    def __post_init__(self):
        """Initialize player with all 21 pieces if not provided"""
        if not self.available_pieces:
            self.available_pieces = create_all_pieces()
    
    def reset(self):
        """Reset player to initial state with all pieces"""
        self.available_pieces = create_all_pieces()
        self.played_pieces = []
        self.has_passed = False
    
    def has_piece(self, piece_type: PieceType) -> bool:
        """Check if player still has a specific piece available"""
        return any(piece.piece_type == piece_type for piece in self.available_pieces)
    
    def get_piece(self, piece_type: PieceType) -> Optional[Piece]:
        """Get a specific piece if available"""
        for piece in self.available_pieces:
            if piece.piece_type == piece_type:
                return piece
        return None
    
    def play_piece(self, piece_type: PieceType) -> bool:
        """
        Mark a piece as played (remove from available pieces).
        Returns True if successful, False if piece not available.
        """
        for i, piece in enumerate(self.available_pieces):
            if piece.piece_type == piece_type:
                self.available_pieces.pop(i)
                self.played_pieces.append(piece_type)
                return True
        return False
    
    def unplay_piece(self, piece_type: PieceType) -> bool:
        """
        Return a piece to available pieces (for undo operations).
        Returns True if successful, False if piece wasn't played.
        """
        if piece_type in self.played_pieces:
            self.played_pieces.remove(piece_type)
            # Re-create the piece and add it back
            from pieces import get_piece_by_type
            self.available_pieces.append(get_piece_by_type(piece_type))
            return True
        return False
    
    def get_remaining_squares(self) -> int:
        """
        Calculate the total number of squares in remaining unplayed pieces.
        Used for scoring.
        """
        return sum(piece.size() for piece in self.available_pieces)
    
    def get_played_squares(self) -> int:
        """Calculate the total number of squares in played pieces"""
        from pieces import get_piece_by_type
        return sum(get_piece_by_type(piece_type).size() 
                  for piece_type in self.played_pieces)
    
    def has_pieces_remaining(self) -> bool:
        """Check if player has any pieces left to play"""
        return len(self.available_pieces) > 0
    
    def pass_turn(self):
        """Mark that player has passed their turn"""
        self.has_passed = True
    
    def can_play(self) -> bool:
        """Check if player can still play (hasn't passed and has pieces)"""
        return not self.has_passed and self.has_pieces_remaining()
    
    def calculate_score(self) -> int:
        """
        Calculate player's score:
        - Played squares count as positive points
        - Remaining (unplayed) squares count as negative points
        - Bonus points if all pieces were played
        """
        played = self.get_played_squares()
        remaining = self.get_remaining_squares()
        
        score = played - remaining
        
        # Bonus points for using all pieces
        if remaining == 0:
            # Check if last piece played was the monomino (1 square piece)
            if self.played_pieces and self.played_pieces[-1] == PieceType.MONO:
                score += 20  # Extra bonus for finishing with monomino
            else:
                score += 15  # Standard bonus for using all pieces
        
        return score
    
    def get_available_piece_types(self) -> List[PieceType]:
        """Get list of piece types still available to this player"""
        return [piece.piece_type for piece in self.available_pieces]
    
    def __str__(self) -> str:
        """String representation of player"""
        return (f"Player({self.color.value}, "
                f"pieces: {len(self.available_pieces)}/21, "
                f"score: {self.calculate_score()})")
    
    def to_dict(self) -> dict:
        """
        Convert player state to dictionary for serialization.
        Useful for web interface communication.
        """
        return {
            'color': self.color.value,
            'available_pieces': [piece.piece_type.value for piece in self.available_pieces],
            'played_pieces': [piece_type.value for piece_type in self.played_pieces],
            'remaining_squares': self.get_remaining_squares(),
            'played_squares': self.get_played_squares(),
            'has_passed': self.has_passed,
            'can_play': self.can_play(),
            'score': self.calculate_score()
        }
