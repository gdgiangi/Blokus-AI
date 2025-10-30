# Blokus Game State Model

A comprehensive state space model for the Blokus board game, ready for web interface integration.

## Overview

This implementation provides a complete representation of the Blokus game state, including:
- **20×20 game board** with cell occupancy tracking
- **2-4 player support** (Blue, Yellow, Red, Green)
- **21 unique polyomino pieces** per player (89 unit squares per player)
- **Complete rule validation** (placement, adjacency, corners)
- **Scoring system** with bonuses
- **Game state management** (initialization, turns, win conditions)

## Project Structure

```
Blokus-AI/
├── board.py                          # Board representation and validation
├── pieces.py                         # All 21 polyomino piece definitions
├── player.py                         # Player state and piece management
├── game_state.py                     # Main game state and logic
├── app.py                            # Flask web application with AI support
├── ai_player.py                      # Modular AI player (5 heuristics)
├── ai_player_enhanced.py             # Enhanced AI (10 heuristics)
├── example.py                        # Usage demonstration
├── ai_example.py                     # AI usage demonstration
├── templates/                        # HTML templates
│   └── index.html                    # Web UI with AI controls
├── static/                           # Static assets
│   ├── css/style.css                 # Styling with AI visualization
│   └── js/app.js                     # Frontend logic with AI integration
├── tuning/                           # AI hyperparameter optimization
│   ├── hyperparameter_tuning.py      # Basic parallelized tuning
│   ├── advanced_hyperparameter_tuning.py # Advanced tuning with champion pool
│   ├── launch_tuning.py               # Easy tuning launcher
│   ├── manage_ai_weights.py           # Weight management utilities
│   └── test_advanced_tuning.py        # Testing utilities
├── docs/                             # Documentation
│   ├── AI_GUIDE.md                   # Advanced AI guide
│   ├── HYPERPARAMETER_TUNING_GUIDE.md # Basic tuning guide
│   ├── ADVANCED_HYPERPARAMETER_TUNING_GUIDE.md # Advanced tuning guide
│   └── ADVANCED_TUNING_README.md     # Advanced tuning overview
└── README.md                         # This file
```

## Core Components

### 1. Board (`board.py`)

The `Board` class represents the 20×20 game grid:

**Features:**
- 20×20 cell grid tracking occupancy by player color
- Starting corner assignment for each player:
  - Blue: (0, 0) - Top-left
  - Yellow: (0, 19) - Top-right
  - Red: (19, 19) - Bottom-right
  - Green: (19, 0) - Bottom-left
- Cell validation (bounds, occupancy)
- Adjacency detection (edge-adjacent and corner-adjacent)
- Placement rule validation

**Key Methods:**
```python
board = Board()
board.is_valid_position(row, col)  # Check if position is in bounds
board.is_empty(row, col)            # Check if cell is unoccupied
board.can_place_piece(coords, color) # Validate piece placement
board.place_piece(coords, color)    # Place piece on board
```

### 2. Pieces (`pieces.py`)

The `Piece` class and `PieceType` enum define all 21 Blokus pieces:

**Piece Distribution:**
- 1 monomino (1 square)
- 1 domino (2 squares)
- 2 triominoes (3 squares each)
- 5 tetrominoes (4 squares each)
- 12 pentominoes (5 squares each)
- **Total: 89 squares per player**

**Features:**
- Coordinate-based piece representation
- Rotation (90° clockwise)
- Flipping (horizontal/vertical)
- All unique orientations generation
- Translation to board positions

**Key Methods:**
```python
piece = Piece(PieceType.PENTA_X, coords)
piece.rotate_90()           # Rotate piece
piece.flip_horizontal()     # Mirror piece
piece.get_all_orientations() # Get all unique rotations/flips
piece.translate(row, col)   # Get absolute board coordinates
```

### 3. Player (`player.py`)

The `Player` class manages individual player state:

**Features:**
- Color assignment (Blue, Yellow, Red, Green)
- Piece inventory (21 pieces initially)
- Played piece tracking
- Score calculation
- Pass/active state

**Scoring Rules:**
- +1 point per square placed on board
- -1 point per unplayed square remaining
- +15 bonus for using all pieces
- +20 bonus if monomino is the last piece played

**Key Methods:**
```python
player = Player(PlayerColor.BLUE)
player.has_piece(piece_type)      # Check piece availability
player.play_piece(piece_type)     # Mark piece as played
player.calculate_score()          # Get current score
player.get_remaining_squares()    # Count unplayed squares
```

### 4. Game State (`game_state.py`)

The `GameState` class manages the complete game:

**Features:**
- 2-4 player game support
- Fixed turn order: Blue → Yellow → Red → Green (clockwise)
- Move validation and execution
- Turn management
- Win condition detection
- Move history tracking
- Complete state serialization (for web interface)

**Game Phases:**
- `NOT_STARTED`: Game initialized but not begun
- `IN_PROGRESS`: Active gameplay
- `FINISHED`: All players passed or no valid moves

**Key Methods:**
```python
game = GameState(num_players=4)
game.start_game()                 # Begin the game
game.get_current_player()         # Get active player
game.can_place_piece(piece, r, c) # Validate move
game.place_piece(type, piece, r, c) # Execute move
game.pass_turn()                  # Skip turn
game.is_game_over()              # Check win condition
game.get_scores()                # Get all player scores
game.to_dict()                   # Serialize state (JSON-ready)
```

## Blokus Rules (As Modeled)

### Setup
1. Board is 20×20 grid (400 cells)
2. Each player gets 21 unique pieces (containing 89 unit squares total)
3. Turn order is always: Blue → Yellow → Red → Green
4. Each player has a designated starting corner

### Gameplay Rules

**First Move:**
- Must place a piece that covers your starting corner
- At least one square of the piece must occupy the corner cell

**Subsequent Moves:**
- Piece must touch corner-to-corner with at least one of your existing pieces
- Piece CANNOT touch edge-to-edge with your own color
- Can touch edges with opponent colors
- All piece squares must be on empty cells

**Game End:**
- Game ends when all players pass or have no valid moves
- Players count scores:
  - +1 for each square on the board
  - -1 for each unplayed square
  - Bonuses for completing all pieces

### Valid Placement Example

```
Your pieces (B):
  B . .
  B B .
  . . .

Valid next placement (corner-to-corner):
  B . B    ✓ Touches diagonally
  B B B    ✗ Would touch edges
  . . .
```

## Usage Example

```python
from game_state import GameState
from pieces import PieceType
from board import PlayerColor

# Initialize a 4-player game
game = GameState(num_players=4)
game.start_game()

# Check current player
current = game.get_current_player()
print(f"Current player: {current.color.value}")

# Get a piece
piece = current.get_piece(PieceType.MONO)

# Validate placement
can_place, error = game.can_place_piece(piece, 0, 0)
if can_place:
    # Place the piece
    game.place_piece(PieceType.MONO, piece, 0, 0)
else:
    print(f"Invalid move: {error}")

# Check scores
scores = game.get_scores()
for color, score in scores.items():
    print(f"{color.value}: {score} points")

# Serialize state (for web interface)
state_dict = game.to_dict()
# Convert to JSON for frontend
import json
json_state = json.dumps(state_dict)
```

## Running the Example

```bash
python example.py
```

This will demonstrate:
- Game initialization
- Player setup
- Piece information
- Move validation
- Scoring
- State serialization

## State Space Summary

### Complete State Representation

The state space model includes:

1. **Board State**: 20×20 grid with cell occupancy
2. **Player States**: 4 players × (21 pieces + metadata)
3. **Game Metadata**: Turn, phase, history
4. **Rules Engine**: Complete validation logic

### State Space Size

- **Board configurations**: 5^400 theoretical (Empty + 4 colors)
- **Practical states**: Much smaller due to rules
- **Pieces per player**: 21 unique shapes (89 unit squares)
- **Orientations**: Each piece has 1-8 unique orientations
- **Total pieces in game**: 84 (4 players × 21 pieces)
- **Total unit squares in game**: 356 (4 players × 89 squares)

### Serializable State

All components can be serialized to dictionaries/JSON:
- `board.to_dict()` - Board state
- `player.to_dict()` - Player state
- `game.to_dict()` - Complete game state

Perfect for:
- Web APIs (REST/GraphQL)
- WebSocket communication
- State persistence
- AI agent interfaces

## Design Principles

1. **Separation of Concerns**: Board, pieces, players, and game logic are separate
2. **Immutability**: Most operations return new objects rather than mutating
3. **Validation**: All rules enforced at the board/game level
4. **Serialization**: Easy conversion to JSON for web interfaces
5. **Type Safety**: Using enums and type hints throughout
6. **Extensibility**: Easy to add AI players, move suggestions, etc.

## Next Steps

This state model is ready for:
- **Web Interface**: Flask/Django backend + React/Vue frontend ✅ **IMPLEMENTED**
- **AI Implementation**: Can plug in search algorithms (minimax, MCTS, etc.) ✅ **IMPLEMENTED**
- **Multiplayer**: State is serializable for network play
- **Move Validation**: All rules enforced ✅
- **Analysis**: Track game statistics, optimal plays, etc.

## AI Features

### 🤖 AI Players

The project includes sophisticated AI players with:
- **10 heuristics** for intelligent move evaluation
- **5 strategies**: Greedy, Balanced, Aggressive, Expansive, Optimized
- **Modular design** for easy customization
- **Visual "thinking"** process in web UI

See `AI_README.md` and `ENHANCED_AI_GUIDE.md` for details.

### ⚡ Hyperparameter Tuning (Parallelized)

Optimize AI performance with automated hyperparameter tuning:
- **Multi-core parallelization** for 8x faster optimization
- **Random search** and **grid search** algorithms
- **Real-time progress** indicators with time estimates
- **Quick tune**: 5-10 minutes (vs 40-80 min sequential)
- **Intensive tune**: 30-40 minutes (vs 4-5 hours sequential)

**Quick Start:**
```python
from tuning.hyperparameter_tuning import quick_tune

# Uses all CPU cores automatically
best_weights = quick_tune()
```

**Advanced Tuning with Champion Pool:**
```python
# Run long tuning session with champion pool system
python tuning/advanced_hyperparameter_tuning.py

# Or use the launcher for easy setup
python tuning/launch_tuning.py
```

### 🎮 Web Interface

Play Blokus in your browser with:
- Interactive board with drag-and-drop pieces
- AI opponent selection and configuration
- Visual AI "thinking" display
- Real-time score tracking

**Run the app:**
```bash
python app.py
```
Then visit `http://localhost:5000`

## License

See LICENSE file for details.

---

**Note**: This is Step 1 of the Blokus implementation - the complete state space model. The web interface will be built on top of this foundation.
