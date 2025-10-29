"""
Example usage of AI Players in Blokus
Demonstrates how to create and use different AI strategies programmatically.
"""

from game_state import GameState
from board import PlayerColor
from ai_player import create_ai_player, GreedyAIStrategy, BalancedAIStrategy, AggressiveAIStrategy, ExpansiveAIStrategy, AIPlayer


def play_ai_game():
    """Play a game with AI players"""
    print("=" * 60)
    print("Blokus AI Demo - 4 AI Players with Different Strategies")
    print("=" * 60)
    
    # Create a new game
    game = GameState(num_players=4)
    game.start_game()
    
    # Create AI players with different strategies
    ai_players = {
        PlayerColor.BLUE: create_ai_player(PlayerColor.BLUE, "greedy"),
        PlayerColor.YELLOW: create_ai_player(PlayerColor.YELLOW, "balanced"),
        PlayerColor.RED: create_ai_player(PlayerColor.RED, "aggressive"),
        PlayerColor.GREEN: create_ai_player(PlayerColor.GREEN, "expansive")
    }
    
    print("\nAI Players:")
    for color, ai in ai_players.items():
        print(f"  {color.value}: {ai.strategy.name}")
    print()
    
    move_count = 0
    max_moves = 100  # Safety limit
    
    # Play the game
    while not game.is_game_over() and move_count < max_moves:
        current_color = game.get_current_color()
        ai_player = ai_players[current_color]
        
        print(f"\n--- Turn {game.turn_number}, {current_color.value}'s move ({ai_player.strategy.name}) ---")
        
        # Get AI's move
        best_move = ai_player.get_move(game)
        
        if best_move is None:
            print(f"{current_color.value} has no valid moves. Passing...")
            game.pass_turn()
        else:
            # Show move details
            print(f"Piece: {best_move.piece_type.value}")
            print(f"Position: ({best_move.row}, {best_move.col})")
            print(f"Score: {best_move.score:.2f}")
            print("Heuristics:")
            for name, value in best_move.heuristic_breakdown.items():
                print(f"  - {name}: {value:.2f}")
            
            # Execute the move
            success = game.place_piece(
                best_move.piece_type,
                best_move.piece,
                best_move.row,
                best_move.col
            )
            
            if not success:
                print(f"ERROR: Failed to place piece!")
                break
        
        move_count += 1
    
    # Show final results
    print("\n" + "=" * 60)
    print("GAME OVER!")
    print("=" * 60)
    print("\nFinal Scores:")
    
    rankings = game.get_rankings()
    for i, (color, score) in enumerate(rankings, 1):
        ai = ai_players[color]
        player = game.get_player(color)
        pieces_left = len(player.available_pieces) if player else 0
        print(f"{i}. {color.value} ({ai.strategy.name}): {score} points ({pieces_left} pieces left)")
    
    winner = game.get_winner()
    if winner:
        print(f"\n🏆 Winner: {winner.value} ({ai_players[winner].strategy.name})!")
    else:
        print("\n🤝 It's a tie!")


def test_ai_strategies():
    """Test different AI strategies and show their thinking"""
    print("\n" + "=" * 60)
    print("AI Strategy Comparison - First Move Analysis")
    print("=" * 60)
    
    strategies = [
        ("Greedy", GreedyAIStrategy()),
        ("Balanced", BalancedAIStrategy()),
        ("Aggressive", AggressiveAIStrategy()),
        ("Expansive", ExpansiveAIStrategy())
    ]
    
    for strategy_name, strategy in strategies:
        print(f"\n--- {strategy_name} Strategy ---")
        
        # Create a fresh game for each test
        game = GameState(num_players=4)
        game.start_game()
        
        ai_player = AIPlayer(PlayerColor.BLUE, strategy)
        
        # Get all evaluated moves
        all_moves = ai_player.get_all_evaluated_moves(game)
        
        if all_moves:
            print(f"Total moves evaluated: {len(all_moves)}")
            print(f"\nTop 3 moves:")
            
            for i, move in enumerate(all_moves[:3], 1):
                print(f"\n{i}. {move.piece_type.value} at ({move.row}, {move.col})")
                print(f"   Total Score: {move.score:.2f}")
                print(f"   Breakdown:")
                for heuristic, value in move.heuristic_breakdown.items():
                    print(f"     - {heuristic}: {value:.2f}")


def demonstrate_custom_strategy():
    """Demonstrate creating a custom AI strategy"""
    print("\n" + "=" * 60)
    print("Custom AI Strategy Demo")
    print("=" * 60)
    
    from ai_player import AIStrategy, Heuristic
    
    class CustomAIStrategy(AIStrategy):
        """Custom AI that prioritizes blocking over expansion"""
        
        def __init__(self):
            super().__init__("Custom Blocker AI")
            
            # Custom weights
            self.add_heuristic("piece_size", Heuristic.piece_size_score, 0.5)
            self.add_heuristic("new_paths", Heuristic.new_paths_score, 1.0)
            self.add_heuristic("blocked_opponents", Heuristic.blocked_opponents_score, 5.0)
            self.add_heuristic("corner_control", Heuristic.corner_control_score, 2.5)
        
        def select_move(self, evaluations):
            """Select move with highest score"""
            if not evaluations:
                return None
            return max(evaluations, key=lambda e: e.score)
    
    # Create game with custom AI
    game = GameState(num_players=2)
    game.start_game()
    
    custom_ai = AIPlayer(PlayerColor.BLUE, CustomAIStrategy())
    balanced_ai = AIPlayer(PlayerColor.YELLOW, BalancedAIStrategy())
    
    print(f"\nCustom AI: {custom_ai.strategy.name}")
    print(f"Opponent: {balanced_ai.strategy.name}")
    
    # Show custom AI's first move
    best_move = custom_ai.get_move(game)
    
    if best_move:
        print(f"\nCustom AI's first move:")
        print(f"Piece: {best_move.piece_type.value}")
        print(f"Position: ({best_move.row}, {best_move.col})")
        print(f"Total Score: {best_move.score:.2f}")
        print(f"Heuristics:")
        for name, value in best_move.heuristic_breakdown.items():
            print(f"  - {name}: {value:.2f}")


if __name__ == "__main__":
    # Run the demos
    print("\n🎮 Blokus AI Examples\n")
    
    # Uncomment the demo you want to run:
    
    # 1. Full AI vs AI game
    play_ai_game()
    
    # 2. Compare strategies on first move
    # test_ai_strategies()
    
    # 3. Create custom strategy
    # demonstrate_custom_strategy()
