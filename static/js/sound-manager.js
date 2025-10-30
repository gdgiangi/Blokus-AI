// Sound Manager for Blokus Game
// Handles all game sound effects

class SoundManager {
    constructor() {
        this.sounds = {};
        this.backgroundMusic = null;
        this.enabled = true;
        this.musicEnabled = true;
        this.volume = 0.5;
        this.musicVolume = 0.2; // Background music is quieter
        this.loadSounds();
        this.loadBackgroundMusic();
    }

    loadSounds() {
        // Define sound files
        const soundFiles = {
            'game-start': '/static/sounds/game-start.mp3',
            'place-piece': '/static/sounds/place-piece.mp3',
            'piece-rotate': '/static/sounds/piece-rotate.wav',
            'invalid-move': '/static/sounds/invalid-move.mp3',
            'game-over': '/static/sounds/game-over.mp3'
        };

        // Preload all sounds
        for (const [name, path] of Object.entries(soundFiles)) {
            const audio = new Audio(path);
            audio.volume = this.volume;
            audio.preload = 'auto';
            this.sounds[name] = audio;
        }
    }

    loadBackgroundMusic() {
        this.backgroundMusic = new Audio('/static/sounds/background-music.mp3');
        this.backgroundMusic.volume = this.musicVolume;
        this.backgroundMusic.loop = true;
        this.backgroundMusic.preload = 'auto';
    }

    play(soundName) {
        if (!this.enabled) return;

        const sound = this.sounds[soundName];
        if (sound) {
            // Clone the audio to allow overlapping sounds
            const soundClone = sound.cloneNode();
            soundClone.volume = this.volume;
            soundClone.play().catch(error => {
                console.warn(`Failed to play sound: ${soundName}`, error);
            });
        } else {
            console.warn(`Sound not found: ${soundName}`);
        }
    }

    playGameStart() {
        this.play('game-start');
    }

    playPlacePiece() {
        this.play('place-piece');
    }

    playPieceRotate() {
        this.play('piece-rotate');
    }

    playInvalidMove() {
        this.play('invalid-move');
    }

    playGameOver() {
        this.play('game-over');
    }

    // Background music controls
    startBackgroundMusic() {
        if (this.musicEnabled && this.backgroundMusic) {
            this.backgroundMusic.play().catch(error => {
                console.warn('Failed to play background music:', error);
            });
        }
    }

    stopBackgroundMusic() {
        if (this.backgroundMusic) {
            this.backgroundMusic.pause();
            this.backgroundMusic.currentTime = 0;
        }
    }

    toggleBackgroundMusic() {
        this.musicEnabled = !this.musicEnabled;
        if (this.musicEnabled) {
            this.startBackgroundMusic();
        } else {
            this.stopBackgroundMusic();
        }
        return this.musicEnabled;
    }

    setVolume(volume) {
        this.volume = Math.max(0, Math.min(1, volume));
        for (const sound of Object.values(this.sounds)) {
            sound.volume = this.volume;
        }
    }

    setMusicVolume(volume) {
        this.musicVolume = Math.max(0, Math.min(1, volume));
        if (this.backgroundMusic) {
            this.backgroundMusic.volume = this.musicVolume;
        }
    }

    toggle() {
        this.enabled = !this.enabled;
        return this.enabled;
    }

    setEnabled(enabled) {
        this.enabled = enabled;
    }
}

// Create global sound manager instance
const soundManager = new SoundManager();
