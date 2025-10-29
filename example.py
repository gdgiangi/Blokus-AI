"""
Example usage of the Blokus game state model.
Demonstrates initialization, game setup, and basic operations.
"""

from game_state import GameState, GamePhase
from board import PlayerColor
from pieces import PieceType


def main():
    """Demonstrate the Blokus game state model"""
    
    print("=" * 60)
    print("BLOKUS GAME STATE MODEL DEMONSTRATION")
    print("=" * 60)
    
    # 1. Initialize a game with 4 players
    print("\n1. Initializing a 4-player game...")
    game = GameState(num_players=4)
    print(f"   Game created with {game.num_players} players")
    print(f"   Phase: {game.phase.value}")
    
    # 2. Start the game
    print("\n2. Starting the game...")
    game.start_game()
    print(f"   Phase: {game.phase.value}")
    print(f"   Current player: {game.get_current_color().value}")
    print(f"   Turn order: {[c.value for c in GameState.TURN_ORDER[:game.num_players]]}")
    
    # 3. Display initial player state
    print("\n3. Player initial state:")
    for color in GameState.TURN_ORDER[:game.num_players]:
        player = game.get_player(color)
        print(f"   {player}")
        print(f"      Starting corner: {game.board.STARTING_CORNERS[color]}")
    
    # 4. Display board information
    print("\n4. Board information:")
    print(f"   Size: {game.board.BOARD_SIZE}x{game.board.BOARD_SIZE}")
    print(f"   Total cells: {game.board.BOARD_SIZE * game.board.BOARD_SIZE}")
    
    # 5. Display piece information
    print("\n5. Piece information:")
    blue_player = game.get_player(PlayerColor.BLUE)
    print(f"   Total pieces per player: {len(blue_player.available_pieces)}")
    print(f"   Total squares per player: {blue_player.get_remaining_squares()}")
    
    piece_counts = {}
    for piece in blue_player.available_pieces:
        size = piece.size()
        piece_counts[size] = piece_counts.get(size, 0) + 1
    
    print(f"   Pieces by size:")
    print(f"      1 square (monomino): {piece_counts.get(1, 0)} piece")
    print(f"      2 squares (domino): {piece_counts.get(2, 0)} piece")
    print(f"      3 squares (triominoes): {piece_counts.get(3, 0)} pieces")
    print(f"      4 squares (tetrominoes): {piece_counts.get(4, 0)} pieces")
    print(f"      5 squares (pentominoes): {piece_counts.get(5, 0)} pieces")
    
    # 6. Demonstrate piece placement validation
    print("\n6. Testing piece placement (Blue's first move):")
    blue_player = game.get_player(PlayerColor.BLUE)
    
    # Get the monomino (smallest piece)
    monomino = blue_player.get_piece(PieceType.MONO)
    print(f"   Selected piece: {monomino.piece_type.value} ({monomino.size()} square)")
    
    # Try to place at Blue's corner (0, 0) - should be valid
    can_place, error = game.can_place_piece(monomino, 0, 0, PlayerColor.BLUE)
    print(f"   Can place at (0,0) [Blue's corner]: {can_place}")
    if not can_place:
        print(f"      Error: {error}")
    
    # Try to place at wrong corner - should be invalid
    can_place, error = game.can_place_piece(monomino, 19, 19, PlayerColor.BLUE)
    print(f"   Can place at (19,19) [wrong corner]: {can_place}")
    if not can_place:
        print(f"      Error: {error}")
    
    # 7. Make a move
    print("\n7. Making Blue's first move:")
    success = game.place_piece(PieceType.MONO, monomino, 0, 0)
    print(f"   Move successful: {success}")
    print(f"   Current player after move: {game.get_current_color().value}")
    print(f"   Blue's pieces remaining: {len(game.get_player(PlayerColor.BLUE).available_pieces)}")
    
    # 8. Display scoring
    print("\n8. Current scores:")
    scores = game.get_scores()
    for color, score in game.get_rankings():
        print(f"   {color.value}: {score} points")
    
    # 9. Game state serialization
    print("\n9. Game state can be serialized to dictionary:")
    state_dict = game.to_dict()
    print(f"   Keys in state dict: {list(state_dict.keys())}")
    print(f"   Can be converted to JSON for web interface")
    
    # 10. Reset demonstration
    print("\n10. Resetting the game:")
    game.reset()
    print(f"   Phase: {game.phase.value}")
    print(f"   Blue's pieces: {len(game.get_player(PlayerColor.BLUE).available_pieces)}")
    
    # 11. Demonstrate 2-player game
    print("\n11. Creating a 2-player game:")
    game2 = GameState(num_players=2)
    game2.start_game()
    print(f"   Players: {[c.value for c in game2.players.keys()]}")
    print(f"   Turn order: Blue -> Yellow -> Blue -> Yellow...")
    
    print("\n" + "=" * 60)
    print("DEMONSTRATION COMPLETE")
    print("=" * 60)
    
    # Display game rules summary
    print("\n📋 BLOKUS RULES SUMMARY (as modeled):")
    print("   • Board: 20x20 grid")
    print("   • Players: 2-4 (Blue, Yellow, Red, Green)")
    print("   • Pieces: 21 per player (89 total squares)")
    print("   • Turn order: Always Blue → Yellow → Red → Green")
    print("   • First move: Must cover player's starting corner")
    print("   • Subsequent moves:")
    print("     ✓ Must touch corner-to-corner with your color")
    print("     ✗ Cannot touch edge-to-edge with your color")
    print("   • Scoring:")
    print("     +1 for each square on board")
    print("     -1 for each unplayed square")
    print("     +15 bonus for using all pieces")
    print("     +20 bonus if monomino is the last piece played")
    print("   • Game ends: When all players pass or run out of moves")


if __name__ == "__main__":
    main()
