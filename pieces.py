"""
Blokus Piece Definitions
Defines all 21 polyomino pieces used in the Blokus game.
Each piece is represented by its relative coordinates from a reference point (0,0).
"""

from enum import Enum
from typing import List, Tuple, Set
from dataclasses import dataclass, field


class PieceType(Enum):
    """Enumeration of all 21 Blokus pieces"""
    # Monomino (1 square)
    MONO = "I1"
    
    # Domino (2 squares)
    DOMINO = "I2"
    
    # Triominoes (3 squares)
    TRI_I = "I3"
    TRI_L = "L3"
    
    # Tetrominoes (4 squares)
    TETRA_I = "I4"
    TETRA_O = "O4"
    TETRA_T = "T4"
    TETRA_L = "L4"
    TETRA_Z = "Z4"
    
    # Pentominoes (12 pieces, 5 squares each)
    PENTA_F = "F"
    PENTA_I = "I5"
    PENTA_L = "L5"
    PENTA_N = "N"
    PENTA_P = "P"
    PENTA_T = "T5"
    PENTA_U = "U"
    PENTA_V = "V"
    PENTA_W = "W"
    PENTA_X = "X"
    PENTA_Y = "Y"
    PENTA_Z = "Z5"


# Define piece shapes as sets of (row, col) coordinates
# Each piece is defined in its canonical orientation
PIECE_SHAPES = {
    # Monomino (1 square)
    PieceType.MONO: [
        (0, 0)
    ],
    
    # Domino (2 squares)
    PieceType.DOMINO: [
        (0, 0), (0, 1)
    ],
    
    # Triominoes (3 squares)
    PieceType.TRI_I: [
        (0, 0), (0, 1), (0, 2)
    ],
    PieceType.TRI_L: [
        (0, 0), (0, 1), (1, 0)
    ],
    
    # Tetrominoes (4 squares)
    PieceType.TETRA_I: [
        (0, 0), (0, 1), (0, 2), (0, 3)
    ],
    PieceType.TETRA_O: [
        (0, 0), (0, 1), (1, 0), (1, 1)
    ],
    PieceType.TETRA_T: [
        (0, 0), (0, 1), (0, 2), (1, 1)
    ],
    PieceType.TETRA_L: [
        (0, 0), (1, 0), (2, 0), (2, 1)
    ],
    PieceType.TETRA_Z: [
        (0, 0), (0, 1), (1, 1), (1, 2)
    ],
    
    # Pentominoes (5 squares each)
    PieceType.PENTA_F: [
        (0, 1), (0, 2), (1, 0), (1, 1), (2, 1)
    ],
    PieceType.PENTA_I: [
        (0, 0), (0, 1), (0, 2), (0, 3), (0, 4)
    ],
    PieceType.PENTA_L: [
        (0, 0), (1, 0), (2, 0), (3, 0), (3, 1)
    ],
    PieceType.PENTA_N: [
        (0, 0), (0, 1), (1, 1), (1, 2), (1, 3)
    ],
    PieceType.PENTA_P: [
        (0, 0), (0, 1), (1, 0), (1, 1), (2, 0)
    ],
    PieceType.PENTA_T: [
        (0, 0), (0, 1), (0, 2), (1, 1), (2, 1)
    ],
    PieceType.PENTA_U: [
        (0, 0), (0, 1), (1, 0), (1, 1), (0, 2)
    ],
    PieceType.PENTA_V: [
        (0, 0), (1, 0), (2, 0), (2, 1), (2, 2)
    ],
    PieceType.PENTA_W: [
        (0, 0), (1, 0), (1, 1), (2, 1), (2, 2)
    ],
    PieceType.PENTA_X: [
        (0, 1), (1, 0), (1, 1), (1, 2), (2, 1)
    ],
    PieceType.PENTA_Y: [
        (0, 0), (1, 0), (1, 1), (2, 0), (3, 0)
    ],
    PieceType.PENTA_Z: [
        (0, 0), (0, 1), (1, 1), (2, 1), (2, 2)
    ],
}


@dataclass
class Piece:
    """
    Represents a Blokus piece with its type, shape, and transformation state.
    """
    piece_type: PieceType
    shape: List[Tuple[int, int]]
    
    def __post_init__(self):
        """Ensure shape is a list for easier manipulation"""
        if not isinstance(self.shape, list):
            self.shape = list(self.shape)
    
    def get_coordinates(self) -> List[Tuple[int, int]]:
        """Get the current coordinates of this piece"""
        return self.shape.copy()
    
    def rotate_90(self) -> 'Piece':
        """
        Rotate the piece 90 degrees clockwise.
        Returns a new Piece instance with rotated coordinates.
        """
        # Rotation formula: (row, col) -> (col, -row)
        rotated = [(col, -row) for row, col in self.shape]
        # Normalize to keep coordinates non-negative
        rotated = self._normalize(rotated)
        return Piece(self.piece_type, rotated)
    
    def flip_horizontal(self) -> 'Piece':
        """
        Flip the piece horizontally (mirror over vertical axis).
        Returns a new Piece instance with flipped coordinates.
        """
        # Flip formula: (row, col) -> (row, -col)
        flipped = [(row, -col) for row, col in self.shape]
        # Normalize to keep coordinates non-negative
        flipped = self._normalize(flipped)
        return Piece(self.piece_type, flipped)
    
    def flip_vertical(self) -> 'Piece':
        """
        Flip the piece vertically (mirror over horizontal axis).
        Returns a new Piece instance with flipped coordinates.
        """
        # Flip formula: (row, col) -> (-row, col)
        flipped = [(-row, col) for row, col in self.shape]
        # Normalize to keep coordinates non-negative
        flipped = self._normalize(flipped)
        return Piece(self.piece_type, flipped)
    
    def _normalize(self, coords: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """
        Normalize coordinates so the minimum row and column are 0.
        This ensures the piece always has a consistent reference point.
        """
        if not coords:
            return coords
        
        min_row = min(row for row, col in coords)
        min_col = min(col for row, col in coords)
        
        normalized = [(row - min_row, col - min_col) for row, col in coords]
        # Sort for consistency in comparison
        return sorted(normalized)
    
    def get_all_orientations(self) -> Set[Tuple[Tuple[int, int], ...]]:
        """
        Generate all unique orientations of this piece.
        Returns a set of tuples representing unique orientations.
        """
        orientations = set()
        current = self
        
        # Generate rotations and flips
        for _ in range(4):  # 4 rotations
            # Add current orientation
            orientations.add(tuple(sorted(current.shape)))
            # Add flipped version
            flipped = current.flip_horizontal()
            orientations.add(tuple(sorted(flipped.shape)))
            # Rotate for next iteration
            current = current.rotate_90()
        
        return orientations
    
    def translate(self, row: int, col: int) -> List[Tuple[int, int]]:
        """
        Translate (move) the piece to a specific position on the board.
        Returns the absolute coordinates of the piece at position (row, col).
        """
        return [(r + row, c + col) for r, c in self.shape]
    
    def size(self) -> int:
        """Return the number of squares in this piece"""
        return len(self.shape)
    
    def __hash__(self):
        """Make piece hashable for use in sets/dicts"""
        return hash((self.piece_type, tuple(sorted(self.shape))))
    
    def __eq__(self, other):
        """Compare pieces for equality"""
        if not isinstance(other, Piece):
            return False
        return (self.piece_type == other.piece_type and 
                sorted(self.shape) == sorted(other.shape))


def create_all_pieces() -> List[Piece]:
    """
    Create instances of all 21 Blokus pieces.
    Returns a list of Piece objects.
    """
    return [Piece(piece_type, coords.copy()) 
            for piece_type, coords in PIECE_SHAPES.items()]


def get_piece_by_type(piece_type: PieceType) -> Piece:
    """
    Get a piece instance by its type.
    """
    if piece_type not in PIECE_SHAPES:
        raise ValueError(f"Invalid piece type: {piece_type}")
    return Piece(piece_type, PIECE_SHAPES[piece_type].copy())
