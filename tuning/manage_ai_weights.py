"""
Weight Management Script for Blokus AI
Easily switch between different champion weight configurations.
"""

import json
from pathlib import Path

# Weight configurations
WEIGHT_CONFIGS = {
    "previous_champion": {
        "name": "Previous Champion (33.3% win rate)",
        "weights": {
            "piece_size": 1.20,
            "new_paths": 1.89,
            "blocked_opponents": 2.42,
            "corner_control": 1.36,
            "compactness": 1.00,
            "flexibility": 1.71,
            "mobility": 0.73,
            "opponent_restriction": 0.89,
            "endgame_optimization": 1.20,
            "territory_expansion": 1.15
        }
    },
    "overnight_champion": {
        "name": "Overnight Marathon Champion (96.7% win rate)",
        "weights": {
            "piece_size": 1.46,
            "new_paths": 1.97,
            "blocked_opponents": 1.62,
            "corner_control": 1.26,
            "compactness": 1.08,
            "flexibility": 1.10,
            "mobility": 0.66,
            "opponent_restriction": 0.88,
            "endgame_optimization": 1.86,
            "territory_expansion": 1.12
        }
    }
}

def format_weights_for_code(weights):
    """Format weights dictionary for code insertion"""
    lines = []
    lines.append('        default_weights = {')
    for key, value in weights.items():
        lines.append(f'            "{key}": {value:.2f},')
    lines.append('        }')
    return '\n'.join(lines)

def update_ai_weights(config_name):
    """Update the AI weights in ai_player_enhanced.py"""
    if config_name not in WEIGHT_CONFIGS:
        print(f"Unknown configuration: {config_name}")
        return False
    
    config = WEIGHT_CONFIGS[config_name]
    weights = config["weights"]
    
    # Read the current file
    ai_file = Path("ai_player_enhanced.py")
    if not ai_file.exists():
        print("ai_player_enhanced.py not found!")
        return False
    
    content = ai_file.read_text()
    
    # Find and replace the default_weights section
    start_marker = "        # Default optimized weights"
    end_marker = "        }"
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Could not find weights section to update!")
        return False
    
    # Find the end of the weights dictionary
    brace_count = 0
    end_idx = start_idx
    in_dict = False
    
    for i, char in enumerate(content[start_idx:], start_idx):
        if char == '{':
            in_dict = True
            brace_count += 1
        elif char == '}' and in_dict:
            brace_count -= 1
            if brace_count == 0:
                end_idx = i + 1
                break
    
    if end_idx <= start_idx:
        print("Could not find end of weights dictionary!")
        return False
    
    # Create new weights section
    new_section = f"""        # Default optimized weights (updated to {config['name']})
        default_weights = {{"""
    
    for key, value in weights.items():
        new_section += f'\n            "{key}": {value:.2f},'
    
    new_section += '\n        }'
    
    # Replace the section
    new_content = content[:start_idx] + new_section + content[end_idx:]
    
    # Write back to file
    ai_file.write_text(new_content)
    
    print(f"✅ Updated AI weights to: {config['name']}")
    print("🎮 You can now play against the new AI in the web interface!")
    
    return True

def show_current_weights():
    """Show the current weights in ai_player_enhanced.py"""
    ai_file = Path("ai_player_enhanced.py")
    if not ai_file.exists():
        print("ai_player_enhanced.py not found!")
        return
    
    content = ai_file.read_text()
    
    # Extract current weights
    start_marker = "default_weights = {"
    start_idx = content.find(start_marker)
    if start_idx == -1:
        print("Could not find current weights!")
        return
    
    # Find the end of the weights dictionary
    brace_count = 0
    end_idx = start_idx
    in_dict = False
    
    for i, char in enumerate(content[start_idx:], start_idx):
        if char == '{':
            in_dict = True
            brace_count += 1
        elif char == '}' and in_dict:
            brace_count -= 1
            if brace_count == 0:
                end_idx = i
                break
    
    weights_section = content[start_idx:end_idx + 1]
    print("📊 Current AI Weights:")
    print(weights_section)

def main():
    """Interactive weight management"""
    print("🎮 Blokus AI Weight Manager")
    print("=" * 40)
    
    while True:
        print("\nAvailable configurations:")
        for i, (key, config) in enumerate(WEIGHT_CONFIGS.items(), 1):
            print(f"  {i}. {config['name']}")
        
        print("\nOptions:")
        print("  s. Show current weights")
        print("  q. Quit")
        
        choice = input("\nEnter choice: ").strip().lower()
        
        if choice == 'q':
            break
        elif choice == 's':
            show_current_weights()
        elif choice in ['1', '2']:
            config_keys = list(WEIGHT_CONFIGS.keys())
            if choice == '1':
                update_ai_weights(config_keys[0])
            elif choice == '2':
                update_ai_weights(config_keys[1])
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    main()