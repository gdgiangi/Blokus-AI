// Blokus Game Web Interface
// Main JavaScript application

// Game state
let gameState = null;
let selectedPiece = null;
let currentPieceShape = null;
let draggedPieceElement = null;

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
    finalRankings: null
};

// Initialize on page load
document.addEventListener('DOMContentLoaded', () => {
    initializeElements();
    attachEventListeners();
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
}

// API Functions
async function createNewGame(numPlayers) {
    try {
        const response = await fetch('/api/game/new', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ num_players: numPlayers })
        });

        if (!response.ok) {
            throw new Error('Failed to create game');
        }

        const data = await response.json();
        gameState = data.game_state;
        updateUI();
        elements.resetGameBtn.disabled = false;
        elements.passTurnBtn.disabled = false;
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
        selectedPiece = null;
        currentPieceShape = null;
        updateUI();
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
            throw new Error(error.error || 'Invalid move');
        }

        const data = await response.json();
        gameState = data.game_state;
        selectedPiece = null;
        currentPieceShape = null;
        updateUI();

        // Check if game is over
        if (gameState.is_game_over) {
            showGameOver();
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
    }
}

function handleFlipHorizontal() {
    if (currentPieceShape) {
        currentPieceShape = flipPieceHorizontal(currentPieceShape);
        updateSelectedPieceDisplay();
    }
}

function handleFlipVertical() {
    if (currentPieceShape) {
        currentPieceShape = flipPieceVertical(currentPieceShape);
        updateSelectedPieceDisplay();
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
    selectedPiece = pieceType;
    currentPieceShape = JSON.parse(JSON.stringify(shape));
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
