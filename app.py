"""
Flask web application for Blokus game interface.
Provides REST API endpoints for game management and state.
"""

from flask import Flask, render_template, jsonify, request
from flask_cors import CORS
from game_state import GameState, GamePhase
from board import PlayerColor
from pieces import PieceType, Piece, PIECE_SHAPES
import json

app = Flask(__name__)
CORS(app)

# Global game state (in production, use session management)
game = None


@app.route('/')
def index():
    """Serve the main game interface"""
    return render_template('index.html')


@app.route('/api/game/new', methods=['POST'])
def new_game():
    """
    Create a new game.
    Expected JSON: {"num_players": 2-4}
    """
    global game
    
    try:
        data = request.get_json()
        num_players = data.get('num_players', 4)
        
        if num_players < 2 or num_players > 4:
            return jsonify({'error': 'Number of players must be between 2 and 4'}), 400
        
        game = GameState(num_players=num_players)
        game.start_game()
        
        return jsonify({
            'success': True,
            'game_state': game.to_dict()
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/game/reset', methods=['POST'])
def reset_game():
    """Reset the current game"""
    global game
    
    try:
        if game is None:
            return jsonify({'error': 'No active game'}), 400
        
        num_players = game.num_players
        game.reset(num_players)
        game.start_game()
        
        return jsonify({
            'success': True,
            'game_state': game.to_dict()
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


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
