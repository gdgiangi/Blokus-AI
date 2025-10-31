# Hyperparameter Tuning Guide (Parallelized)

## Overview
The hyperparameter tuning system finds optimal heuristic weights through automated game simulation. **Now with multi-core parallelization for significantly faster execution!**

## Performance Improvements

### Parallelization Benefits
- **Independent game simulations** run concurrently across CPU cores
- Near-linear speedup with number of cores (e.g., 8 cores ≈ 8x faster)
- Automatically detects and uses all available CPU cores by default
- Can be configured to use specific number of processes

### Speed Comparison
Example on a typical 8-core system:
- **Sequential (old)**: 8 games in ~80 seconds
- **Parallel (new)**: 8 games in ~11 seconds (**7.3x faster**)

## Quick Start

### 1. Run Quick Tune (5-10 minutes with parallelization)
```python
from tuning.hyperparameter_tuning import quick_tune

# Use all CPU cores (default)
best_weights = quick_tune()

# Or specify number of processes
best_weights = quick_tune(n_jobs=4)
```

### 2. Run Intensive Tune (20-40 minutes with parallelization)
```python
from tuning.hyperparameter_tuning import intensive_tune

best_weights = intensive_tune()  # Uses all cores
```

### 3. Custom Tuning
```python
from tuning.hyperparameter_tuning import HyperparameterTuner

# Initialize with parallelization
tuner = HyperparameterTuner(n_jobs=None)  # None or -1 = all cores
# tuner = HyperparameterTuner(n_jobs=4)   # Use 4 cores
# tuner = HyperparameterTuner(n_jobs=1)   # Disable parallelization

base_weights = {
    "piece_size": 1.5,
    "new_paths": 2.5,
    "blocked_opponents": 2.0,
    "corner_control": 1.5,
    "compactness": 1.0,
    "flexibility": 2.0,
    "mobility": 1.0,
    "opponent_restriction": 2.0,
    "endgame_optimization": 1.0,
    "territory_expansion": 1.0
}

# Random search (parallelized)
best = tuner.random_search(
    base_weights,
    num_iterations=20,
    games_per_iteration=10,
    variation=0.3
)

# Fine-tune specific heuristic (parallelized)
best_weight = tuner.grid_search(
    "flexibility",
    best,
    (1.0, 3.0, 0.2),  # min, max, step
    games_per_weight=10
)
```

## Tuning Methods

### Random Search
Tests random variations around base weights.

**Parameters:**
- `num_iterations`: How many random configurations to test
- `games_per_iteration`: Games played per configuration
- `variation`: Variation range (0.3 = ±30%)

## Understanding Results

### Win Rate
Percentage of games where the test AI (blue player) wins against random opponents.
- **30-40%**: Baseline performance
- **40-50%**: Good improvement
- **50-60%**: Excellent performance
- **>60%**: Outstanding (may be overfitting)

### Interpreting Weights
After tuning, you'll see optimal weights for each heuristic:
- **Higher values** = more influence on move selection
- **Compare to baseline** to see which heuristics matter most
- **Consistent winners** across runs are robust

## Best Practices

### 1. Start with Quick Tune
```python
# Fast exploration (5-10 min with parallelization)
quick_tune()
```

### 2. Run Multiple Sessions
```python
# Tuning has randomness, run 3-5 times
for i in range(3):
    weights = quick_tune()
    print(f"\nRun {i+1} best weights:", weights)
```

### 3. Fine-Tune Top Heuristics
After identifying important heuristics, do targeted grid search:
```python
tuner = HyperparameterTuner()

# From quick_tune, suppose "flexibility" was crucial
best_flexibility = tuner.grid_search(
    "flexibility",
    base_weights,
    (1.5, 3.0, 0.1),
    games_per_weight=15
)
```

### 4. Save and Compare Results
```python
tuner.save_results("tuning_run1.json")
# Load later for comparison
tuner.load_results("tuning_run1.json")
```

## Using Tuned Weights

### Update ai_player_enhanced.py
```python
class OptimizedAIStrategy(AIStrategy):
    def __init__(self):
        self.weights = {
            "piece_size": 1.58,           # From tuning results
            "new_paths": 2.73,
            "blocked_opponents": 2.15,
            "corner_control": 1.62,
            "compactness": 0.98,
            "flexibility": 2.28,
            "mobility": 1.12,
            "opponent_restriction": 2.18,
            "endgame_optimization": 1.05,
            "territory_expansion": 0.95
        }
```

### Create Custom Strategy
```python
from ai_player import create_ai_player

my_weights = {
    # Your tuned weights here
}

ai = create_ai_player(
    player_color="blue",
    strategy="balanced",
    custom_weights=my_weights
)
```

## Advanced: Command Line Interface

```bash
# Run from terminal
cd c:\Repositories\Blokus-AI
.venv\Scripts\python.exe hyperparameter_tuning.py

# Choose mode:
# 1. Quick tune (~5-10 minutes with 8 cores)
# 2. Intensive tune (~30-40 minutes with 8 cores)
# 3. Custom
```

## Performance Monitoring

The system provides real-time progress information:
- **Time per iteration**: How long each configuration takes
- **Estimated remaining time**: When tuning will complete
- **Win rate trends**: Immediate feedback on improvements
- **Total elapsed time**: Final timing summary
