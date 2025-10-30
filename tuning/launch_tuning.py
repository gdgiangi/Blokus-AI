"""
Quick Launcher for Advanced Hyperparameter Tuning
Provides predefined configurations for different tuning scenarios.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from advanced_hyperparameter_tuning import run_comprehensive_tuning


def quick_exploration():
    """Quick 1-hour exploration session"""
    print("🔍 Quick Exploration (1 hour)")
    print("   Games per evaluation: 12")
    print("   Estimated total games: 2,000-4,000")
    print("   Good for: Initial weight discovery")
    
    return run_comprehensive_tuning(
        session_name="quick_exploration",
        max_hours=1.0,
        games_per_eval=12
    )


def standard_optimization():
    """Standard 4-hour optimization session"""
    print("⚡ Standard Optimization (4 hours)")
    print("   Games per evaluation: 20")
    print("   Estimated total games: 15,000-25,000")
    print("   Good for: Solid optimization runs")
    
    return run_comprehensive_tuning(
        session_name="standard_optimization",
        max_hours=4.0,
        games_per_eval=20
    )


def intensive_search():
    """Intensive 8-hour deep search"""
    print("🚀 Intensive Search (8 hours)")
    print("   Games per evaluation: 25")
    print("   Estimated total games: 40,000-60,000")
    print("   Good for: Finding excellent weights")
    
    return run_comprehensive_tuning(
        session_name="intensive_search",
        max_hours=8.0,
        games_per_eval=25
    )


def overnight_marathon():
    """Overnight 12-hour marathon session"""
    print("🌙 Overnight Marathon (12 hours)")
    print("   Games per evaluation: 30")
    print("   Estimated total games: 80,000-120,000")
    print("   Good for: Maximum optimization")
    
    return run_comprehensive_tuning(
        session_name="overnight_marathon",
        max_hours=12.0,
        games_per_eval=30
    )


def weekend_powerhouse():
    """Weekend 24-hour powerhouse session"""
    print("💪 Weekend Powerhouse (24 hours)")
    print("   Games per evaluation: 35")
    print("   Estimated total games: 200,000-300,000")
    print("   Good for: Ultimate optimization")
    
    return run_comprehensive_tuning(
        session_name="weekend_powerhouse",
        max_hours=24.0,
        games_per_eval=35
    )


def custom_session():
    """Custom session with user input"""
    print("🎛️  Custom Session")
    
    session_name = input("Session name: ").strip()
    if not session_name:
        session_name = "custom_session"
    
    while True:
        try:
            hours = float(input("Duration in hours: "))
            if hours > 0:
                break
            else:
                print("Please enter a positive number.")
        except ValueError:
            print("Please enter a valid number.")
    
    while True:
        try:
            games = int(input("Games per evaluation: "))
            if games > 0:
                break
            else:
                print("Please enter a positive integer.")
        except ValueError:
            print("Please enter a valid integer.")
    
    estimated_total = int(hours * 60 * games)
    print(f"\nEstimated total games: {estimated_total:,}")
    
    return run_comprehensive_tuning(
        session_name=session_name,
        max_hours=hours,
        games_per_eval=games
    )


def main():
    """Main launcher interface"""
    print("🎮 Advanced Blokus AI Hyperparameter Tuning Launcher")
    print("=" * 60)
    print()
    
    options = {
        "1": ("Quick Exploration (1h)", quick_exploration),
        "2": ("Standard Optimization (4h)", standard_optimization),
        "3": ("Intensive Search (8h)", intensive_search),
        "4": ("Overnight Marathon (12h)", overnight_marathon),
        "5": ("Weekend Powerhouse (24h)", weekend_powerhouse),
        "6": ("Custom Session", custom_session),
        "q": ("Quit", None)
    }
    
    print("Select tuning session type:")
    for key, (description, _) in options.items():
        if key != "q":
            print(f"  {key}. {description}")
        else:
            print(f"  {key}. {description}")
    
    while True:
        choice = input("\nEnter choice (1-6, q): ").strip().lower()
        
        if choice == "q":
            print("Goodbye!")
            return
        
        if choice in options and choice != "q":
            description, func = options[choice]
            print(f"\n{description}")
            print("-" * 40)
            
            confirm = input("Proceed with this session? (y/N): ").strip().lower()
            if confirm == "y":
                try:
                    print(f"\nStarting {description}...")
                    print("=" * 60)
                    
                    best_weights = func()
                    
                    print("\n🎉 Tuning Complete!")
                    print("=" * 60)
                    print("Best weights found:")
                    for name, weight in sorted(best_weights.items()):
                        print(f"  {name}: {weight:.3f}")
                    
                    break
                    
                except KeyboardInterrupt:
                    print("\n⏹️  Tuning interrupted by user")
                    break
                except Exception as e:
                    print(f"\n❌ Error during tuning: {e}")
                    break
            else:
                print("Session cancelled.")
        else:
            print("Invalid choice. Please enter 1-6 or q.")


if __name__ == "__main__":
    main()