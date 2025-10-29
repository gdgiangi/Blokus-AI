"""
Flask web application for Blokus game interface.
Provides REST API endpoints for game management and state.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from game_state import GameState, GamePhase
from board import PlayerColor
from pieces import PieceType, Piece, PIECE_SHAPES
from ai_player import create_ai_player, AIPlayer
import json

app = Flask(__name__)
CORS(app)

# Global game state (in production, use session management)
game = None
ai_players = {}  # Dictionary to store AI players by color


@app.route('/')
def index():
    """Serve the main game interface"""
    return render_template('index.html')


@app.route('/api/game/new', methods=['POST'])
def new_game():
    """
    Create a new game.
    Expected JSON: {
        "num_players": 2-4,
        "ai_players": {"blue": "balanced", "red": "aggressive", ...}
    }
    """
    global game, ai_players
    
    try:
        data = request.get_json()
        num_players = data.get('num_players', 4)
        ai_config = data.get('ai_players', {})
        
        if num_players < 2 or num_players > 4:
            return jsonify({'error': 'Number of players must be between 2 and 4'}), 400
        
        game = GameState(num_players=num_players)
        game.start_game()
        
        # Initialize AI players
        ai_players = {}
        for color_str, strategy in ai_config.items():
            try:
                color = PlayerColor(color_str)
                if color in game.players:
                    ai_players[color] = create_ai_player(color, strategy)
            except ValueError:
                pass  # Invalid color, skip
        
        return jsonify({
            'success': True,
            'game_state': game.to_dict(),
            'ai_players': {color.value: ai.strategy.name for color, ai in ai_players.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    """Reset the current game"""
    global game, ai_players
    
    try:
        if game is None:
            return jsonify({'error': 'No active game'}), 400
        
        num_players = game.num_players
        game.reset(num_players)
        game.start_game()
        
        # Keep AI players but they'll work with the reset game state
        
        return jsonify({
            'success': True,
            'game_state': game.to_dict(),
            'ai_players': {color.value: ai.strategy.name for color, ai in ai_players.items()}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/state', methods=['GET'])
def get_game_state():
    """Get current game state"""
    global game
    
    if game is None:
        return jsonify({'error': 'No active game'}), 400
    
    return jsonify(game.to_dict())


@app.route('/api/game/place', methods=['POST'])
def place_piece():
    """
    Place a piece on the board.
    Expected JSON: {
        "piece_type": "I1",
        "row": 0,
        "col": 0,
        "shape": [[0,0], [0,1], ...]  // coordinates after rotation/flip
    }
    """
    global game
    
    if game is None:
        return jsonify({'error': 'No active game'}), 400
    
    try:
        data = request.get_json()
        piece_type_str = data.get('piece_type')
        row = data.get('row')
        col = data.get('col')
        shape = data.get('shape')
        
        if not all([piece_type_str, row is not None, col is not None, shape]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # Convert piece type string to enum
        piece_type = PieceType(piece_type_str)
        
        # Create piece with the provided shape (already oriented)
        piece = Piece(piece_type, shape)
        
        # Attempt to place the piece
        success = game.place_piece(piece_type, piece, row, col)
        
        if success:
            return jsonify({
                'success': True,
                'game_state': game.to_dict()
            })
        else:
            return jsonify({'error': 'Invalid move'}), 400
            
    except ValueError as e:
        return jsonify({'error': f'Invalid piece type: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/pass', methods=['POST'])
def pass_turn():
    """Current player passes their turn"""
    global game
    
    if game is None:
        return jsonify({'error': 'No active game'}), 400
    
    try:
        success = game.pass_turn()
        
        if success:
            return jsonify({
                'success': True,
                'game_state': game.to_dict()
            })
        else:
            return jsonify({'error': 'Cannot pass turn'}), 400
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/validate', methods=['POST'])
def validate_placement():
    """
    Validate if a piece can be placed without actually placing it.
    Expected JSON: {
        "piece_type": "I1",
        "row": 0,
        "col": 0,
        "shape": [[0,0], [0,1], ...]
    }
    """
    global game
    
    if game is None:
        return jsonify({'error': 'No active game'}), 400
    
    try:
        data = request.get_json()
        piece_type_str = data.get('piece_type')
        row = data.get('row')
        col = data.get('col')
        shape = data.get('shape')
        
        piece_type = PieceType(piece_type_str)
        piece = Piece(piece_type, shape)
        
        can_place, error = game.can_place_piece(piece, row, col)
        
        return jsonify({
            'valid': can_place,
            'error': error if not can_place else None
        })
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/pieces/all', methods=['GET'])
def get_all_pieces():
    """Get all piece shapes and types"""
    pieces_data = {}
    
    for piece_type, shape in PIECE_SHAPES.items():
        pieces_data[piece_type.value] = {
            'type': piece_type.value,
            'shape': shape,
            'size': len(shape)
        }
    
    return jsonify(pieces_data)


@app.route('/api/ai/move', methods=['POST'])
def ai_move():
    """
    Get AI move for the current player.
    Optionally returns all evaluated moves for visualization.
    Expected JSON: {"show_thinking": true/false}
    """
    global game, ai_players
    
    if game is None:
        return jsonify({'error': 'No active game'}), 400
    
    try:
        data = request.get_json() or {}
        show_thinking = data.get('show_thinking', True)
        
        current_color = game.get_current_color()
        
        # Check if current player is AI
        if current_color not in ai_players:
            return jsonify({'error': 'Current player is not AI'}), 400
        
        ai_player = ai_players[current_color]
        
        # Get AI's move
        best_move = ai_player.get_move(game)
        
        if best_move is None:
            # AI has no valid moves, pass turn
            game.pass_turn()
            return jsonify({
                'success': True,
                'action': 'pass',
                'game_state': game.to_dict()
            })
        
        # If requested, get all evaluated moves for visualization
        all_moves = []
        if show_thinking:
            all_moves = ai_player.get_all_evaluated_moves(game)
            # Limit to top 1000 for performance while showing comprehensive evaluation
            all_moves = [move.to_dict() for move in all_moves[:1000]]
        
        return jsonify({
            'success': True,
            'action': 'move',
            'best_move': best_move.to_dict(),
            'all_moves': all_moves,
            'thinking_enabled': show_thinking
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/execute', methods=['POST'])
def ai_execute():
    """
    Execute the AI's move on the board.
    This is separate from /ai/move to allow visualization before execution.
    """
    global game, ai_players
    
    if game is None:
        return jsonify({'error': 'No active game'}), 400
    
    try:
        current_color = game.get_current_color()
        
        # Check if current player is AI
        if current_color not in ai_players:
            return jsonify({'error': 'Current player is not AI'}), 400
        
        ai_player = ai_players[current_color]
        
        # Get AI's move
        best_move = ai_player.get_move(game)
        
        if best_move is None:
            # AI has no valid moves, pass turn
            game.pass_turn()
            return jsonify({
                'success': True,
                'action': 'pass',
                'game_state': game.to_dict()
            })
        
        # Execute the move
        success = game.place_piece(
            best_move.piece_type,
            best_move.piece,
            best_move.row,
            best_move.col
        )
        
        if success:
            return jsonify({
                'success': True,
                'action': 'move',
                'move': best_move.to_dict(),
                'game_state': game.to_dict()
            })
        else:
            return jsonify({'error': 'Failed to execute move'}), 500
            
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/ai/check', methods=['GET'])
def check_ai():
    """Check if current player is AI and if it's their turn"""
    global game, ai_players
    
    if game is None:
        return jsonify({'error': 'No active game'}), 400
    
    current_color = game.get_current_color()
    is_ai = current_color in ai_players
    
    ai_info = {}
    if is_ai:
        ai_info = {
            'color': current_color.value,
            'strategy': ai_players[current_color].strategy.name
        }
    
    return jsonify({
        'is_ai_turn': is_ai,
        'ai_info': ai_info,
        'all_ai_players': {color.value: ai.strategy.name for color, ai in ai_players.items()}
    })


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
