// Blokus Game Web Interface
// Main JavaScript application

// Game state
let gameState = null;
let selectedPiece = null;
let currentPieceShape = null;
let draggedPieceElement = null;
let aiPlayers = {};  // Track which players are AI
let aiThinking = false;  // Track if AI is currently "thinking"
let aiThinkingInterval = null;  // Interval for AI thinking animation
let mctsProgressInterval = null;  // Interval for MCTS progress updates
let isMCTSActive = false;  // Track if current AI is MCTS

// Color mapping
const COLORS = {
    'blue': '#3b82f6',
    'yellow': '#fbbf24',
    'red': '#ef4444',
    'green': '#10b981'
};

// DOM Elements
const elements = {
    board: null,
    piecesContainer: null,
    currentPlayer: null,
    turnNumber: null,
    scoresContainer: null,
    newGameBtn: null,
    resetGameBtn: null,
    passTurnBtn: null,
    numPlayersSelect: null,
    rotateBtn: null,
    flipHBtn: null,
    flipVBtn: null,
    gameOverModal: null,
    closeModalBtn: null,
    finalRankings: null,
    aiControlsContainer: null,
    aiThinkingPanel: null,
    aiDecisionLog: null,
    mctsProgressPanel: null,
    mctsSimulations: null,
    mctsNodes: null,
    mctsDepth: null,
    mctsBestVisits: null,
    mctsTime: null,
    mctsProgressFill: null,
    musicToggleBtn: null
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    attachEventListeners();
    updateAIControls(); // Initialize AI controls

    // Start background music
    if (typeof soundManager !== 'undefined') {
        soundManager.startBackgroundMusic();
    }

    console.log('Blokus game interface initialized');
});

function initializeElements() {
    elements.board = document.getElementById('game-board');
    elements.piecesContainer = document.getElementById('pieces-container');
    elements.currentPlayer = document.getElementById('current-player');
    elements.turnNumber = document.getElementById('turn-number');
    elements.scoresContainer = document.getElementById('scores');
    elements.newGameBtn = document.getElementById('new-game-btn');
    elements.resetGameBtn = document.getElementById('reset-game-btn');
    elements.passTurnBtn = document.getElementById('pass-turn-btn');
    elements.numPlayersSelect = document.getElementById('num-players');
    elements.rotateBtn = document.getElementById('rotate-btn');
    elements.flipHBtn = document.getElementById('flip-h-btn');
    elements.flipVBtn = document.getElementById('flip-v-btn');
    elements.gameOverModal = document.getElementById('game-over-modal');
    elements.closeModalBtn = document.getElementById('close-modal-btn');
    elements.finalRankings = document.getElementById('final-rankings');
    elements.aiControlsContainer = document.getElementById('ai-controls');
    elements.aiThinkingPanel = document.getElementById('ai-thinking-panel');
    elements.aiDecisionLog = document.getElementById('ai-decision-log');
    elements.mctsProgressPanel = document.getElementById('mcts-progress-panel');
    elements.mctsSimulations = document.getElementById('mcts-simulations');
    elements.mctsNodes = document.getElementById('mcts-nodes');
    elements.mctsDepth = document.getElementById('mcts-depth');
    elements.mctsBestVisits = document.getElementById('mcts-best-visits');
    elements.mctsTime = document.getElementById('mcts-time');
    elements.mctsProgressFill = document.getElementById('mcts-progress-fill');
    elements.musicToggleBtn = document.getElementById('music-toggle-btn');
}

function attachEventListeners() {
    elements.newGameBtn.addEventListener('click', handleNewGame);
    elements.resetGameBtn.addEventListener('click', handleResetGame);
    elements.passTurnBtn.addEventListener('click', handlePassTurn);
    elements.rotateBtn.addEventListener('click', handleRotate);
    elements.flipHBtn.addEventListener('click', handleFlipHorizontal);
    elements.flipVBtn.addEventListener('click', handleFlipVertical);
    elements.closeModalBtn.addEventListener('click', () => {
        elements.gameOverModal.classList.remove('active');
    });

    // Music toggle button
    if (elements.musicToggleBtn) {
        elements.musicToggleBtn.addEventListener('click', () => {
            if (typeof soundManager !== 'undefined') {
                const isEnabled = soundManager.toggleBackgroundMusic();
                elements.musicToggleBtn.textContent = isEnabled ? '🎵' : '🔇';
            }
        });
    }

    // Add event listener for num players change to update AI controls
    if (elements.numPlayersSelect) {
        elements.numPlayersSelect.addEventListener('change', updateAIControls);
    }
}

// API Functions
async function createNewGame(numPlayers) {
    try {
        // Get AI player selections from checkboxes
        const aiConfig = {};
        const colors = ['blue', 'yellow', 'red', 'green'];

        for (let i = 0; i < numPlayers; i++) {
            const color = colors[i];
            const checkbox = document.getElementById(`ai-${color}`);
            const strategySelect = document.getElementById(`strategy-${color}`);

            if (checkbox && checkbox.checked && strategySelect) {
                aiConfig[color] = strategySelect.value;
            }
        }

        const response = await fetch('/api/game/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                num_players: numPlayers,
                ai_players: aiConfig
            })
        });

        if (!response.ok) {
            throw new Error('Failed to create game');
        }

        const data = await response.json();
        gameState = data.game_state;
        aiPlayers = data.ai_players || {};
        updateUI();
        elements.resetGameBtn.disabled = false;
        elements.passTurnBtn.disabled = false;

        // Play game start sound
        if (typeof soundManager !== 'undefined') {
            soundManager.playGameStart();
        }

        // Check if it's AI's turn
        checkAITurn();
    } catch (error) {
        console.error('Error creating game:', error);
        alert('Failed to create game. Please try again.');
    }
}

async function resetGame() {
    try {
        const response = await fetch('/api/game/reset', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            throw new Error('Failed to reset game');
        }

        const data = await response.json();
        gameState = data.game_state;
        aiPlayers = data.ai_players || {};
        selectedPiece = null;
        currentPieceShape = null;
        updateUI();

        // Check if it's AI's turn
        checkAITurn();
    } catch (error) {
        console.error('Error resetting game:', error);
        alert('Failed to reset game. Please try again.');
    }
}

async function placePiece(pieceType, row, col, shape) {
    try {
        const response = await fetch('/api/game/place', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                piece_type: pieceType,
                row: row,
                col: col,
                shape: shape
            })
        });

        if (!response.ok) {
            const error = await response.json();

            // Play invalid move sound
            if (typeof soundManager !== 'undefined') {
                soundManager.playInvalidMove();
            }

            throw new Error(error.error || 'Invalid move');
        }

        const data = await response.json();
        gameState = data.game_state;
        selectedPiece = null;
        currentPieceShape = null;
        updateUI();

        // Play place piece sound
        if (typeof soundManager !== 'undefined') {
            soundManager.playPlacePiece();
        }

        // Check if game is over
        if (gameState.is_game_over) {
            showGameOver();
        } else {
            // Check if it's AI's turn
            checkAITurn();
        }

        return true;
    } catch (error) {
        console.error('Error placing piece:', error);
        return false;
    }
}

async function passTurn() {
    try {
        const response = await fetch('/api/game/pass', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            throw new Error('Failed to pass turn');
        }

        const data = await response.json();
        gameState = data.game_state;
        selectedPiece = null;
        currentPieceShape = null;
        updateUI();

        if (gameState.is_game_over) {
            showGameOver();
        } else {
            // Check if it's AI's turn
            checkAITurn();
        }
    } catch (error) {
        console.error('Error passing turn:', error);
        alert('Failed to pass turn. Please try again.');
    }
}

// Event Handlers
function handleNewGame() {
    const numPlayers = parseInt(elements.numPlayersSelect.value);
    createNewGame(numPlayers);
}

function handleResetGame() {
    if (confirm('Are you sure you want to reset the game?')) {
        resetGame();
    }
}

function handlePassTurn() {
    if (confirm('Are you sure you want to pass your turn?')) {
        passTurn();
    }
}

function handleRotate() {
    if (currentPieceShape) {
        currentPieceShape = rotatePiece(currentPieceShape);
        updateSelectedPieceDisplay();

        // Play rotate sound
        if (typeof soundManager !== 'undefined') {
            soundManager.playPieceRotate();
        }
    }
}

function handleFlipHorizontal() {
    if (currentPieceShape) {
        currentPieceShape = flipPieceHorizontal(currentPieceShape);
        updateSelectedPieceDisplay();

        // Play rotate sound
        if (typeof soundManager !== 'undefined') {
            soundManager.playPieceRotate();
        }
    }
}

function handleFlipVertical() {
    if (currentPieceShape) {
        currentPieceShape = flipPieceVertical(currentPieceShape);
        updateSelectedPieceDisplay();

        // Play rotate sound
        if (typeof soundManager !== 'undefined') {
            soundManager.playPieceRotate();
        }
    }
}

// Piece transformation functions
function rotatePiece(shape) {
    // Rotate 90 degrees clockwise: (row, col) -> (col, -row)
    let rotated = shape.map(([row, col]) => [col, -row]);
    return normalizePiece(rotated);
}

function flipPieceHorizontal(shape) {
    // Flip horizontally: (row, col) -> (row, -col)
    let flipped = shape.map(([row, col]) => [row, -col]);
    return normalizePiece(flipped);
}

function flipPieceVertical(shape) {
    // Flip vertically: (row, col) -> (-row, col)
    let flipped = shape.map(([row, col]) => [-row, col]);
    return normalizePiece(flipped);
}

function normalizePiece(coords) {
    if (coords.length === 0) return coords;

    const minRow = Math.min(...coords.map(c => c[0]));
    const minCol = Math.min(...coords.map(c => c[1]));

    return coords.map(([row, col]) => [row - minRow, col - minCol])
        .sort((a, b) => a[0] - b[0] || a[1] - b[1]);
}

// UI Update Functions
function updateUI() {
    if (!gameState) return;

    updateBoard();
    updatePlayerInfo();
    updateScores();
    updatePieces();
}

function updateBoard() {
    if (!gameState) return;

    // Clear board
    elements.board.innerHTML = '';

    // Create 20x20 grid
    const grid = gameState.board.grid;
    for (let row = 0; row < 20; row++) {
        for (let col = 0; col < 20; col++) {
            const cell = document.createElement('div');
            cell.className = 'cell';
            cell.dataset.row = row;
            cell.dataset.col = col;

            // Check if cell is occupied
            const cellValue = grid[row][col];
            if (cellValue) {
                cell.classList.add('occupied', cellValue);
            }

            // Mark starting corners
            if (row === 0 && col === 0) cell.classList.add('corner-blue');
            if (row === 0 && col === 19) cell.classList.add('corner-yellow');
            if (row === 19 && col === 19) cell.classList.add('corner-red');
            if (row === 19 && col === 0) cell.classList.add('corner-green');

            // Add drop handler
            cell.addEventListener('dragover', handleDragOver);
            cell.addEventListener('drop', handleDrop);
            cell.addEventListener('mouseenter', handleCellHover);
            cell.addEventListener('mouseleave', handleCellLeave);

            elements.board.appendChild(cell);
        }
    }
}

function updatePlayerInfo() {
    if (!gameState) return;

    const currentColor = gameState.current_player;
    elements.currentPlayer.textContent = currentColor;
    elements.currentPlayer.className = 'player-name ' + currentColor;
    elements.turnNumber.textContent = gameState.turn_number;
}

function updateScores() {
    if (!gameState) return;

    elements.scoresContainer.innerHTML = '';

    const currentColor = gameState.current_player;

    // Display scores in order
    gameState.rankings.forEach(([color, score]) => {
        const scoreItem = document.createElement('div');
        scoreItem.className = 'score-item';
        if (color === currentColor) {
            scoreItem.classList.add('active');
        }

        const playerInfo = gameState.players[color];
        const piecesLeft = playerInfo.available_pieces.length;

        scoreItem.innerHTML = `
            <div class="score-player">
                <div class="score-color ${color}"></div>
                <span class="score-name">${color}</span>
                <span class="score-pieces">(${piecesLeft} pieces)</span>
            </div>
            <span class="score-value">${score}</span>
        `;

        elements.scoresContainer.appendChild(scoreItem);
    });
}

function updatePieces() {
    if (!gameState) return;

    elements.piecesContainer.innerHTML = '';

    const currentColor = gameState.current_player;
    const currentPlayer = gameState.players[currentColor];

    if (!currentPlayer) return;

    // Group pieces by size
    const piecesData = {};
    currentPlayer.available_pieces.forEach(pieceType => {
        if (!piecesData[pieceType]) {
            piecesData[pieceType] = getPieceShape(pieceType);
        }
    });

    // Sort by size
    const sortedPieces = Object.entries(piecesData).sort((a, b) => {
        return a[1].length - b[1].length;
    });

    sortedPieces.forEach(([pieceType, shape]) => {
        const pieceElement = createPieceElement(pieceType, shape, currentColor);
        elements.piecesContainer.appendChild(pieceElement);
    });

    // Update piece control buttons
    updatePieceControls();
}

function createPieceElement(pieceType, shape, color) {
    const pieceItem = document.createElement('div');
    pieceItem.className = 'piece-item';
    pieceItem.dataset.pieceType = pieceType;
    pieceItem.draggable = true;

    // Calculate grid dimensions
    const maxRow = Math.max(...shape.map(c => c[0])) + 1;
    const maxCol = Math.max(...shape.map(c => c[1])) + 1;

    const pieceGrid = document.createElement('div');
    pieceGrid.className = 'piece-grid';
    pieceGrid.style.gridTemplateColumns = `repeat(${maxCol}, 10px)`;
    pieceGrid.style.gridTemplateRows = `repeat(${maxRow}, 10px)`;

    // Create grid cells
    for (let row = 0; row < maxRow; row++) {
        for (let col = 0; col < maxCol; col++) {
            const isOccupied = shape.some(([r, c]) => r === row && c === col);
            if (isOccupied) {
                const square = document.createElement('div');
                square.className = `piece-square ${color}`;
                square.style.gridColumn = col + 1;
                square.style.gridRow = row + 1;
                pieceGrid.appendChild(square);
            }
        }
    }

    pieceItem.appendChild(pieceGrid);

    // Add event listeners
    pieceItem.addEventListener('click', () => selectPiece(pieceType, shape));
    pieceItem.addEventListener('dragstart', (e) => handleDragStart(e, pieceType, shape));
    pieceItem.addEventListener('dragend', handleDragEnd);

    return pieceItem;
}

function selectPiece(pieceType, shape) {
    selectedPiece = pieceType;
    currentPieceShape = JSON.parse(JSON.stringify(shape)); // Deep copy

    // Update UI
    document.querySelectorAll('.piece-item').forEach(item => {
        item.classList.remove('selected');
    });

    const pieceElement = document.querySelector(`[data-piece-type="${pieceType}"]`);
    if (pieceElement) {
        pieceElement.classList.add('selected');
    }

    updatePieceControls();
}

function updateSelectedPieceDisplay() {
    if (!selectedPiece || !currentPieceShape) return;

    const pieceElement = document.querySelector(`[data-piece-type="${selectedPiece}"]`);
    if (!pieceElement) return;

    const color = gameState.current_player;
    const maxRow = Math.max(...currentPieceShape.map(c => c[0])) + 1;
    const maxCol = Math.max(...currentPieceShape.map(c => c[1])) + 1;

    const pieceGrid = pieceElement.querySelector('.piece-grid');
    pieceGrid.innerHTML = '';
    pieceGrid.style.gridTemplateColumns = `repeat(${maxCol}, 10px)`;
    pieceGrid.style.gridTemplateRows = `repeat(${maxRow}, 10px)`;

    for (let row = 0; row < maxRow; row++) {
        for (let col = 0; col < maxCol; col++) {
            const isOccupied = currentPieceShape.some(([r, c]) => r === row && c === col);
            if (isOccupied) {
                const square = document.createElement('div');
                square.className = `piece-square ${color}`;
                square.style.gridColumn = col + 1;
                square.style.gridRow = row + 1;
                pieceGrid.appendChild(square);
            }
        }
    }
}

function updatePieceControls() {
    const hasSelection = selectedPiece !== null;
    elements.rotateBtn.disabled = !hasSelection;
    elements.flipHBtn.disabled = !hasSelection;
    elements.flipVBtn.disabled = !hasSelection;
}

// Drag and Drop Handlers
function handleDragStart(e, pieceType, shape) {
    // If this piece is already selected, use the current transformed shape
    // Otherwise, select it with the original shape
    if (selectedPiece === pieceType && currentPieceShape) {
        // Keep the current transformed shape
    } else {
        selectedPiece = pieceType;
        currentPieceShape = JSON.parse(JSON.stringify(shape));

        // Update the visual display to match the selected piece
        document.querySelectorAll('.piece-item').forEach(item => {
            item.classList.remove('selected');
        });
        e.target.classList.add('selected');
        updateSelectedPieceDisplay();
    }

    draggedPieceElement = e.target;
    e.target.classList.add('dragging');
    e.dataTransfer.effectAllowed = 'move';
}

function handleDragEnd(e) {
    e.target.classList.remove('dragging');
    clearPreview();
}

function handleDragOver(e) {
    e.preventDefault();
    e.dataTransfer.dropEffect = 'move';

    if (selectedPiece && currentPieceShape) {
        const cell = e.target;
        const row = parseInt(cell.dataset.row);
        const col = parseInt(cell.dataset.col);

        if (!isNaN(row) && !isNaN(col)) {
            showPreview(row, col);
        }
    }
}

function handleDrop(e) {
    e.preventDefault();

    if (!selectedPiece || !currentPieceShape) return;

    const cell = e.target;
    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);

    if (!isNaN(row) && !isNaN(col)) {
        placePiece(selectedPiece, row, col, currentPieceShape);
    }

    clearPreview();
}

function handleCellHover(e) {
    if (!selectedPiece || !currentPieceShape) return;

    const cell = e.target;
    const row = parseInt(cell.dataset.row);
    const col = parseInt(cell.dataset.col);

    if (!isNaN(row) && !isNaN(col)) {
        showPreview(row, col);
    }
}

function handleCellLeave(e) {
    // Only clear if we're not entering another cell
    if (!e.relatedTarget || !e.relatedTarget.classList.contains('cell')) {
        clearPreview();
    }
}

function showPreview(row, col) {
    clearPreview();

    if (!currentPieceShape) return;

    // Calculate absolute positions
    const positions = currentPieceShape.map(([r, c]) => [row + r, col + c]);

    // Check validity
    const isValid = positions.every(([r, c]) => {
        return r >= 0 && r < 20 && c >= 0 && c < 20;
    });

    // Add preview to cells
    positions.forEach(([r, c]) => {
        const cell = document.querySelector(`[data-row="${r}"][data-col="${c}"]`);
        if (cell) {
            cell.classList.add('preview');
            if (isValid) {
                cell.classList.add('valid');
            } else {
                cell.classList.add('invalid');
            }
        }
    });
}

function clearPreview() {
    document.querySelectorAll('.cell.preview').forEach(cell => {
        cell.classList.remove('preview', 'valid', 'invalid');
    });
}

// Game Over
function showGameOver() {
    elements.finalRankings.innerHTML = '';

    gameState.rankings.forEach(([color, score], index) => {
        const rankingItem = document.createElement('div');
        rankingItem.className = 'ranking-item';
        if (index === 0) {
            rankingItem.classList.add('winner');
        }

        rankingItem.innerHTML = `
            <span class="ranking-position">${index + 1}</span>
            <div class="ranking-player">
                <div class="score-color ${color}"></div>
                <span class="score-name">${color}</span>
            </div>
            <span class="ranking-score">${score}</span>
        `;

        elements.finalRankings.appendChild(rankingItem);
    });

    elements.gameOverModal.classList.add('active');

    // Play game over sound
    if (typeof soundManager !== 'undefined') {
        soundManager.playGameOver();
    }
}

// Helper function to get piece shape (from backend data)
function getPieceShape(pieceType) {
    // This should ideally come from the backend
    // For now, we'll use a basic mapping
    const shapes = {
        'I1': [[0, 0]],
        'I2': [[0, 0], [0, 1]],
        'I3': [[0, 0], [0, 1], [0, 2]],
        'L3': [[0, 0], [0, 1], [1, 0]],
        'I4': [[0, 0], [0, 1], [0, 2], [0, 3]],
        'O4': [[0, 0], [0, 1], [1, 0], [1, 1]],
        'T4': [[0, 0], [0, 1], [0, 2], [1, 1]],
        'L4': [[0, 0], [1, 0], [2, 0], [2, 1]],
        'Z4': [[0, 0], [0, 1], [1, 1], [1, 2]],
        'F': [[0, 1], [0, 2], [1, 0], [1, 1], [2, 1]],
        'I5': [[0, 0], [0, 1], [0, 2], [0, 3], [0, 4]],
        'L5': [[0, 0], [1, 0], [2, 0], [3, 0], [3, 1]],
        'N': [[0, 0], [0, 1], [1, 1], [1, 2], [1, 3]],
        'P': [[0, 0], [0, 1], [1, 0], [1, 1], [2, 0]],
        'T5': [[0, 0], [0, 1], [0, 2], [1, 1], [2, 1]],
        'U': [[0, 0], [0, 1], [0, 2], [1, 0], [1, 2]],
        'V': [[0, 0], [1, 0], [2, 0], [2, 1], [2, 2]],
        'W': [[0, 0], [1, 0], [1, 1], [2, 1], [2, 2]],
        'X': [[0, 1], [1, 0], [1, 1], [1, 2], [2, 1]],
        'Y': [[0, 0], [1, 0], [1, 1], [2, 0], [3, 0]],
        'Z5': [[0, 0], [0, 1], [1, 1], [2, 1], [2, 2]]
    };

    return shapes[pieceType] || [[0, 0]];
}

// AI Functions
async function checkAITurn() {
    if (!gameState || gameState.is_game_over) {
        return;
    }

    try {
        const response = await fetch('/api/ai/check');
        const data = await response.json();

        if (data.is_ai_turn) {
            // It's an AI player's turn
            aiPlayers = data.all_ai_players;
            await performAIMove();
        }
    } catch (error) {
        console.error('Error checking AI turn:', error);
    }
}

async function performAIMove() {
    if (aiThinking) return;  // Already thinking

    aiThinking = true;

    try {
        // Check if current AI is MCTS
        isMCTSActive = false;

        // Get AI's evaluated moves with corner search visualization
        const response = await fetch('/api/ai/move', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                show_thinking: true,
                show_corner_search: true  // Request corner expansion visualization
            })
        });

        if (!response.ok) {
            throw new Error('Failed to get AI move');
        }

        const data = await response.json();

        // Check if this is MCTS AI
        isMCTSActive = data.is_mcts;

        if (data.action === 'pass') {
            // AI has no valid moves
            gameState = data.game_state;
            updateUI();
            hideMCTSProgress();

            if (gameState.is_game_over) {
                showGameOver();
            } else {
                setTimeout(() => {
                    aiThinking = false;
                    checkAITurn();
                }, 1000);
            }
            return;
        }

        if (isMCTSActive) {
            // Show MCTS progress
            showMCTSProgress();
            startMCTSProgressTracking();

            // Wait a bit for MCTS to finish
            setTimeout(async () => {
                await executeAIMove();
                stopMCTSProgressTracking();
                hideMCTSProgress();
            }, 3500); // Give MCTS time to complete
        } else {
            // Show traditional AI thinking process (reduced sample for speed)
            if (data.all_moves && data.all_moves.length > 0) {
                await showAIThinking(data.all_moves, data.best_move);
            }

            // Show corner search visualization if available
            if (data.best_move && data.best_move.corner_search_data) {
                await visualizeCornerExpansion(data.best_move);
            }

            // Highlight final move briefly with player color
            const playerColor = gameState ? gameState.current_player : null;
            visualizeAIMove(data.best_move, true, playerColor);
            await new Promise(resolve => setTimeout(resolve, 400));

            // Execute the move
            await executeAIMove();
        }

    } catch (error) {
        console.error('Error performing AI move:', error);
        aiThinking = false;
        stopMCTSProgressTracking();
        hideMCTSProgress();
    }
}

async function showAIThinking(allMoves, bestMove) {
    return new Promise((resolve) => {
        // Get current player color for color-coded animations
        const playerColor = gameState ? gameState.current_player : null;

        // Group moves by piece type to show variety
        const movesByPiece = {};
        allMoves.forEach(move => {
            if (!movesByPiece[move.piece_type]) {
                movesByPiece[move.piece_type] = [];
            }
            movesByPiece[move.piece_type].push(move);
        });

        // Visualize different pieces on the board (sample a few to show variety)
        const sampleMoves = [];
        Object.values(movesByPiece).forEach(moves => {
            if (moves.length > 0) {
                sampleMoves.push(moves[0]); // Take best move for each piece
            }
        });

        // Limit to top 8 pieces to visualize (reduced from 15 for faster animation)
        const movesToShow = sampleMoves.slice(0, 8);
        let currentIndex = 0;

        // Clear any existing interval
        if (aiThinkingInterval) {
            clearInterval(aiThinkingInterval);
        }

        // Animate through different pieces on the board (faster)
        aiThinkingInterval = setInterval(() => {
            if (currentIndex < movesToShow.length) {
                const move = movesToShow[currentIndex];
                visualizeAIMove(move, false, playerColor);
                currentIndex++;
            } else {
                clearInterval(aiThinkingInterval);
                aiThinkingInterval = null;

                // Clear the preview before corner visualization
                clearAIVisualization();
                resolve();
            }
        }, 80);  // Faster visualization (80ms per piece, down from 100ms)
    });
}

function addAIDecisionToLog(bestMove, totalMoves, playerColor) {
    if (!elements.aiDecisionLog) return;

    // Find the top 3 heuristics that influenced this decision
    const heuristics = Object.entries(bestMove.heuristic_breakdown || {});
    const topReasons = heuristics
        .sort((a, b) => b[1] - a[1])
        .slice(0, 3);

    // Human-readable heuristic names
    const heuristicNames = {
        'piece_size': 'larger piece',
        'new_paths': 'opens up new placement options',
        'blocked_opponents': 'blocks opponent moves',
        'corner_control': 'controls key corners',
        'compactness': 'maintains compact formation',
        'flexibility': 'preserves future options',
        'mobility': 'increases movement flexibility',
        'opponent_restriction': 'limits opponent opportunities',
        'endgame_optimization': 'optimizes for endgame',
        'territory_expansion': 'expands territory control'
    };

    const reasonsList = topReasons.map(([name, value]) =>
        `<li><strong>${heuristicNames[name] || name}</strong> (${value.toFixed(1)})</li>`
    ).join('');

    // Create decision log entry with color coding
    const decisionEntry = document.createElement('div');
    decisionEntry.className = `ai-decision-entry ${playerColor}`;
    decisionEntry.innerHTML = `
        <div class="ai-decision-header">
            <div class="ai-decision-player">
                <div class="player-color-dot ${playerColor}"></div>
                <strong>${playerColor.toUpperCase()}</strong>
            </div>
            <div class="ai-decision-move">
                <span class="piece-badge">${bestMove.piece_type}</span>
                <span class="position-badge">at (${bestMove.row}, ${bestMove.col})</span>
            </div>
            <div class="ai-decision-score">Score: ${bestMove.score.toFixed(1)}</div>
        </div>
        <div class="ai-decision-reasons">
            <strong>Why this move?</strong>
            <ul>${reasonsList}</ul>
        </div>
        <div class="ai-decision-meta">${totalMoves} positions evaluated</div>
    `;

    // Add to top of log (most recent first)
    elements.aiDecisionLog.insertBefore(decisionEntry, elements.aiDecisionLog.firstChild);

    // Auto-scroll to show latest decision
    if (elements.aiThinkingPanel) {
        elements.aiDecisionLog.scrollTop = 0;
    }
}

function visualizeAIMove(move, isFinal = false, playerColor = null) {
    // Clear previous visualization
    clearAIVisualization();

    const coordinates = move.shape.map(([r, c]) => [move.row + r, move.col + c]);

    coordinates.forEach(([r, c]) => {
        const cell = document.querySelector(`[data-row="${r}"][data-col="${c}"]`);
        if (cell) {
            cell.classList.add('ai-preview');
            // Add player-specific color class
            if (playerColor) {
                cell.classList.add(`ai-preview-${playerColor}`);
            }
            if (isFinal) {
                cell.classList.add('ai-final');
                if (playerColor) {
                    cell.classList.add(`ai-final-${playerColor}`);
                }
            }
        }
    });
}

function clearAIVisualization() {
    document.querySelectorAll('.cell.ai-preview').forEach(cell => {
        cell.classList.remove('ai-preview', 'ai-final');
        // Remove all player-specific color classes
        cell.classList.remove('ai-preview-blue', 'ai-preview-yellow', 'ai-preview-red', 'ai-preview-green');
        cell.classList.remove('ai-final-blue', 'ai-final-yellow', 'ai-final-red', 'ai-final-green');
    });
    // Also clear corner search visualization
    clearCornerSearchVisualization();
}

function clearCornerSearchVisualization() {
    document.querySelectorAll('.cell.corner-search-cell').forEach(cell => {
        cell.classList.remove(
            'corner-search-cell',
            'corner-search-radius-1',
            'corner-search-radius-2',
            'corner-search-radius-3',
            'corner-search-directional',
            'corner-marker',
            // Remove player-specific color classes
            'corner-marker-blue',
            'corner-marker-yellow',
            'corner-marker-red',
            'corner-marker-green',
            'corner-search-blue',
            'corner-search-yellow',
            'corner-search-red',
            'corner-search-green'
        );
    });
}

async function visualizeCornerExpansion(moveData) {
    const searchData = moveData.corner_search_data;
    if (!searchData || !searchData.corners || searchData.corners.length === 0) {
        return;
    }

    return new Promise((resolve) => {
        const corners = searchData.corners;
        const playerColor = gameState ? gameState.current_player : null;

        // Limit to top 3 most important corners to avoid overwhelming visualization
        const topCorners = corners
            .sort((a, b) => b.potential - a.potential)
            .slice(0, Math.min(3, corners.length));

        // Animate all corners simultaneously with staggered start
        topCorners.forEach((corner, cornerIndex) => {
            const [cornerRow, cornerCol] = corner.position;
            const startDelay = cornerIndex * 150; // Stagger each corner by 150ms

            setTimeout(() => {
                // Mark the corner itself
                const cornerCell = document.querySelector(`[data-row="${cornerRow}"][data-col="${cornerCol}"]`);
                if (cornerCell) {
                    cornerCell.classList.add('corner-search-cell', 'corner-marker');
                    if (playerColor) {
                        cornerCell.classList.add(`corner-marker-${playerColor}`);
                    }
                }

                // Group cells by radius for wave animation
                const cellsByRadius = { 1: [], 2: [], 3: [], directional: [] };

                corner.cells_examined.forEach(cellData => {
                    if (cellData.cell_type === 'expansion_search') {
                        if (!cellsByRadius[cellData.radius]) {
                            cellsByRadius[cellData.radius] = [];
                        }
                        cellsByRadius[cellData.radius].push(cellData);
                    } else if (cellData.cell_type === 'directional_search') {
                        cellsByRadius.directional.push(cellData);
                    }
                });

                // Animate waves rapidly - all radii in parallel for each corner
                [1, 2, 3].forEach(radius => {
                    const waveDelay = radius * 80; // Fast wave expansion: 80ms per radius
                    setTimeout(() => {
                        const cells = cellsByRadius[radius] || [];
                        cells.forEach(cellData => {
                            const cell = document.querySelector(`[data-row="${cellData.row}"][data-col="${cellData.col}"]`);
                            if (cell && !cell.classList.contains('occupied')) {
                                cell.classList.add('corner-search-cell', `corner-search-radius-${radius}`);
                                if (playerColor) {
                                    cell.classList.add(`corner-search-${playerColor}`);
                                }
                            }
                        });
                    }, waveDelay);
                });
            }, startDelay);
        });

        // Clear visualization after brief display
        const totalDuration = (topCorners.length * 150) + (3 * 80) + 600; // Total animation time
        setTimeout(() => {
            clearCornerSearchVisualization();
            resolve();
        }, totalDuration);
    });
}

async function executeAIMove() {
    try {
        const response = await fetch('/api/ai/execute', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });

        if (!response.ok) {
            throw new Error('Failed to execute AI move');
        }

        const data = await response.json();

        if (data.action === 'pass') {
            gameState = data.game_state;
        } else {
            gameState = data.game_state;

            // Play place piece sound for AI moves
            if (typeof soundManager !== 'undefined') {
                soundManager.playPlacePiece();
            }
        }

        // Clear AI visualization
        clearAIVisualization();
        // Keep the AI decision log visible - it's now persistent

        updateUI();

        if (gameState.is_game_over) {
            showGameOver();
        } else {
            // Wait a bit, then check if next player is also AI
            setTimeout(() => {
                aiThinking = false;
                checkAITurn();
            }, 1000);
        }

    } catch (error) {
        console.error('Error executing AI move:', error);
        aiThinking = false;
    }
}

// Update num players select to show AI controls
function updateAIControls() {
    const numPlayers = parseInt(elements.numPlayersSelect.value);
    const colors = ['blue', 'yellow', 'red', 'green'];

    if (!elements.aiControlsContainer) return;

    elements.aiControlsContainer.innerHTML = '<h3>AI Players</h3>';

    for (let i = 0; i < numPlayers; i++) {
        const color = colors[i];
        const controlDiv = document.createElement('div');
        controlDiv.className = 'ai-control';
        controlDiv.innerHTML = `
            <label class="ai-toggle">
                <input type="checkbox" id="ai-${color}">
                <span class="ai-color-badge ${color}">${color.toUpperCase()}</span>
            </label>
            <select id="strategy-${color}" class="ai-strategy-select">
                <option value="random">★☆☆☆☆ Random</option>
                <option value="defensive">★★☆☆☆ Defensive</option>
                <option value="balanced" selected>★★★☆☆ Balanced</option>
                <option value="aggressive">★★★★☆ Aggressive</option>
                <option value="optimized">★★★★★ Champion</option>
            </select>
        `;
        elements.aiControlsContainer.appendChild(controlDiv);
    }
}

// MCTS Progress Functions
function showMCTSProgress() {
    // AI thinking components are hidden - do nothing
    return;
}

function hideMCTSProgress() {
    // AI thinking components are hidden - do nothing
    return;
}

function resetMCTSProgress() {
    if (elements.mctsSimulations) elements.mctsSimulations.textContent = '0';
    if (elements.mctsNodes) elements.mctsNodes.textContent = '0';
    if (elements.mctsDepth) elements.mctsDepth.textContent = '0';
    if (elements.mctsBestVisits) elements.mctsBestVisits.textContent = '0';
    if (elements.mctsTime) elements.mctsTime.textContent = '0.0s';
    if (elements.mctsProgressFill) elements.mctsProgressFill.style.width = '0%';
}

function updateMCTSProgress(progress) {
    // AI thinking components are hidden - do nothing
    return;
}

function startMCTSProgressTracking() {
    // AI thinking components are hidden - do nothing
    return;
}

function stopMCTSProgressTracking() {
    if (mctsProgressInterval) {
        clearInterval(mctsProgressInterval);
        mctsProgressInterval = null;
    }
}

// Note: updateAIControls is called in DOMContentLoaded and attached as event listener in attachEventListeners

