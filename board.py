"""
Blokus Board
Represents the 20x20 game board and its operations.
"""

from typing import Optional, Set, Tuple, List
from enum import Enum


class PlayerColor(Enum):
    """Player colors in Blokus"""
    BLUE = "blue"
    YELLOW = "yellow"
    RED = "red"
    GREEN = "green"
    EMPTY = None
    
    def __str__(self):
        return self.value if self.value else "empty"


class Board:
    """
    Represents the 20x20 Blokus game board.
    Each cell can be empty or occupied by a player's piece.
    """
    
    BOARD_SIZE = 20
    
    # Starting corners for each player (row, col)
    STARTING_CORNERS = {
        PlayerColor.BLUE: (0, 0),           # Top-left
        PlayerColor.YELLOW: (0, 19),        # Top-right
        PlayerColor.RED: (19, 19),          # Bottom-right
        PlayerColor.GREEN: (19, 0)          # Bottom-left
    }
    
    def __init__(self):
        """Initialize an empty 20x20 board"""
        # Board grid - None means empty, otherwise contains PlayerColor
        self.grid: List[List[Optional[PlayerColor]]] = [
            [None for _ in range(self.BOARD_SIZE)] 
            for _ in range(self.BOARD_SIZE)
        ]
        
        # Track which corners have been used (for first move validation)
        self.corners_used: Set[PlayerColor] = set()
    
    def reset(self):
        """Reset the board to empty state"""
        self.grid = [
            [None for _ in range(self.BOARD_SIZE)] 
            for _ in range(self.BOARD_SIZE)
        ]
        self.corners_used = set()
    
    def is_valid_position(self, row: int, col: int) -> bool:
        """Check if a position is within board bounds"""
        return 0 <= row < self.BOARD_SIZE and 0 <= col < self.BOARD_SIZE
    
    def is_empty(self, row: int, col: int) -> bool:
        """Check if a cell is empty"""
        if not self.is_valid_position(row, col):
            return False
        return self.grid[row][col] is None
    
    def get_cell(self, row: int, col: int) -> Optional[PlayerColor]:
        """Get the occupant of a cell"""
        if not self.is_valid_position(row, col):
            return None
        return self.grid[row][col]
    
    def place_piece(self, coordinates: List[Tuple[int, int]], color: PlayerColor):
        """
        Place a piece on the board at the given coordinates.
        Does not validate placement rules - use can_place_piece for validation.
        """
        for row, col in coordinates:
            if not self.is_valid_position(row, col):
                raise ValueError(f"Invalid position: ({row}, {col})")
            if not self.is_empty(row, col):
                raise ValueError(f"Position ({row}, {col}) is already occupied")
            self.grid[row][col] = color
    
    def remove_piece(self, coordinates: List[Tuple[int, int]]):
        """Remove a piece from the board (for undo operations)"""
        for row, col in coordinates:
            if self.is_valid_position(row, col):
                self.grid[row][col] = None
    
    def get_adjacent_cells(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get cells that share an edge with the given cell (up, down, left, right)"""
        adjacent = []
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                adjacent.append((new_row, new_col))
        return adjacent
    
    def get_diagonal_cells(self, row: int, col: int) -> List[Tuple[int, int]]:
        """Get cells that touch diagonally (corner-to-corner)"""
        diagonal = []
        for dr, dc in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
            new_row, new_col = row + dr, col + dc
            if self.is_valid_position(new_row, new_col):
                diagonal.append((new_row, new_col))
        return diagonal
    
    def has_adjacent_same_color(self, coordinates: List[Tuple[int, int]], 
                                color: PlayerColor) -> bool:
        """
        Check if any of the given coordinates have an edge-adjacent cell 
        of the same color (which would violate Blokus rules).
        """
        for row, col in coordinates:
            for adj_row, adj_col in self.get_adjacent_cells(row, col):
                if self.grid[adj_row][adj_col] == color:
                    return True
        return False
    
    def has_diagonal_same_color(self, coordinates: List[Tuple[int, int]], 
                               color: PlayerColor) -> bool:
        """
        Check if any of the given coordinates have a diagonal cell 
        of the same color (required for valid placement after first move).
        """
        for row, col in coordinates:
            for diag_row, diag_col in self.get_diagonal_cells(row, col):
                if self.grid[diag_row][diag_col] == color:
                    return True
        return False
    
    def touches_corner(self, coordinates: List[Tuple[int, int]], 
                      color: PlayerColor) -> bool:
        """
        Check if the piece covers the starting corner for the given color.
        Used for validating first move.
        """
        corner = self.STARTING_CORNERS.get(color)
        if corner is None:
            return False
        return corner in coordinates
    
    def is_first_move(self, color: PlayerColor) -> bool:
        """Check if this is the first move for the given color"""
        return color not in self.corners_used
    
    def can_place_piece(self, coordinates: List[Tuple[int, int]], 
                       color: PlayerColor) -> Tuple[bool, str]:
        """
        Validate if a piece can be placed at the given coordinates.
        
        Returns:
            Tuple of (is_valid, error_message)
            - is_valid: True if placement is legal, False otherwise
            - error_message: Description of why placement is invalid (empty string if valid)
        """
        # Check all coordinates are within bounds and empty
        for row, col in coordinates:
            if not self.is_valid_position(row, col):
                return False, f"Position ({row}, {col}) is out of bounds"
            if not self.is_empty(row, col):
                return False, f"Position ({row}, {col}) is already occupied"
        
        # First move must touch the player's starting corner
        if self.is_first_move(color):
            if not self.touches_corner(coordinates, color):
                corner = self.STARTING_CORNERS[color]
                return False, f"First move must cover starting corner {corner}"
        else:
            # Subsequent moves must touch corner-to-corner with same color
            if not self.has_diagonal_same_color(coordinates, color):
                return False, "Piece must touch a corner of your own color"
        
        # Piece cannot touch edge-to-edge with same color
        if self.has_adjacent_same_color(coordinates, color):
            return False, "Piece cannot share an edge with your own color"
        
        return True, ""
    
    def mark_corner_used(self, color: PlayerColor):
        """Mark that a player has made their first move"""
        self.corners_used.add(color)
    
    def count_occupied_cells(self, color: PlayerColor) -> int:
        """Count how many cells are occupied by a specific color"""
        count = 0
        for row in self.grid:
            for cell in row:
                if cell == color:
                    count += 1
        return count
    
    def get_occupied_positions(self, color: PlayerColor) -> List[Tuple[int, int]]:
        """Get all positions occupied by a specific color"""
        positions = []
        for row in range(self.BOARD_SIZE):
            for col in range(self.BOARD_SIZE):
                if self.grid[row][col] == color:
                    positions.append((row, col))
        return positions
    
    def __str__(self) -> str:
        """String representation of the board for debugging"""
        lines = []
        color_symbols = {
            PlayerColor.BLUE: 'B',
            PlayerColor.YELLOW: 'Y',
            PlayerColor.RED: 'R',
            PlayerColor.GREEN: 'G',
            None: '.'
        }
        
        for row in self.grid:
            line = ' '.join(color_symbols.get(cell, '.') for cell in row)
            lines.append(line)
        
        return '\n'.join(lines)
    
    def to_dict(self) -> dict:
        """
        Convert board state to a dictionary for serialization.
        Useful for web interface communication.
        """
        return {
            'size': self.BOARD_SIZE,
            'grid': [
                [cell.value if cell else None for cell in row]
                for row in self.grid
            ],
            'corners_used': [color.value for color in self.corners_used]
        }
