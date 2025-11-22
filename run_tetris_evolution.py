"""
Simple Evolution Runner for Tetris
Can be used with or without OpenEvolve
"""
import subprocess
import json
from pathlib import Path
import shutil


def simple_evolution_loop(iterations=10, num_games_per_eval=3):
    """
    A simple evolution loop that doesn't require OpenEvolve
    Uses subprocess to modify code and evaluate
    """
    
    print("="*60)
    print("SIMPLE TETRIS EVOLUTION (No OpenEvolve Required)")
    print("="*60)
    print(f"Iterations: {iterations}")
    print(f"Games per evaluation: {num_games_per_eval}")
    print()
    
    best_score = -float('inf')
    best_code_path = "/home/claude/tetris_agent.py"
    
    # Evaluate initial program
    print("Evaluating initial program...")
    result = evaluate_program(best_code_path, num_games_per_eval)
    best_score = result['score']
    
    print(f"Initial score: {best_score:.2f}")
    print(f"  Avg game score: {result['avg_score']:.2f}")
    print(f"  Avg lines: {result['avg_lines']:.2f}")
    print()
    
    # Save best version
    shutil.copy(best_code_path, "/home/claude/best_agent.py")
    
    print("Evolution would continue here with LLM-generated modifications")
    print("To use real evolution, install OpenEvolve:")
    print("  pip install openevolve")
    print()
    print("Then run:")
    print("  python run_tetris_evolution.py --with-openevolve")
    
    return best_score, "/home/claude/best_agent.py"


def evaluate_program(program_path, num_games):
    """Evaluate a program and return metrics"""
    # Import here to avoid circular dependencies
    from tetris_evaluator import evaluate_agent
    
    results = evaluate_agent(
        program_path=program_path,
        num_games=num_games,
        max_steps=2000,
        render=False
    )
    
    return results


def run_with_openevolve(iterations=50):
    """
    Run evolution using OpenEvolve
    Requires: pip install openevolve
    """
    try:
        from openevolve import run_evolution
    except ImportError:
        print("OpenEvolve not installed. Install with: pip install openevolve")
        print("Falling back to simple evolution loop...")
        return simple_evolution_loop()
    
    print("="*60)
    print("TETRIS EVOLUTION WITH OPENEVOLVE")
    print("="*60)
    print(f"Iterations: {iterations}")
    print()
    
    # Define evaluator function for OpenEvolve
    def evaluator(program_path):
        results = evaluate_program(program_path, num_games=5)
        # Return dict with 'score' key for OpenEvolve
        return {'score': results['score']}
    
    # Read initial program
    with open('/home/claude/tetris_agent.py', 'r') as f:
        initial_code = f.read()
    
    # Run evolution
    result = run_evolution(
        initial_program=initial_code,
        evaluator=evaluator,
        iterations=iterations
    )
    
    print("\n" + "="*60)
    print("EVOLUTION COMPLETE!")
    print("="*60)
    print(f"Best score: {result.best_score:.2f}")
    print(f"Best code saved to: best_evolved_agent.py")
    
    # Save best code
    with open('/home/claude/best_evolved_agent.py', 'w') as f:
        f.write(result.best_code)
    
    return result.best_score, '/home/claude/best_evolved_agent.py'


def visualize_best_agent(agent_path, num_games=3):
    """
    Visualize the best evolved agent playing
    """
    print("\n" + "="*60)
    print(f"VISUALIZING: {agent_path}")
    print("="*60)
    
    from tetris_evaluator import evaluate_agent
    
    results = evaluate_agent(
        program_path=agent_path,
        num_games=num_games,
        max_steps=3000,
        render=True  # Show games
    )
    
    print("\n" + "="*60)
    print("VISUALIZATION COMPLETE")
    print("="*60)
    print(f"Average score: {results['avg_score']:.2f}")
    print(f"Max score: {results['max_score']:.2f}")
    print(f"Average lines: {results['avg_lines']:.2f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Evolve Tetris AI')
    parser.add_argument('--with-openevolve', action='store_true',
                        help='Use OpenEvolve for evolution')
    parser.add_argument('--iterations', type=int, default=50,
                        help='Number of evolution iterations')
    parser.add_argument('--visualize', action='store_true',
                        help='Visualize best agent after evolution')
    parser.add_argument('--visualize-only', type=str,
                        help='Just visualize a specific agent file')
    
    args = parser.parse_args()
    
    if args.visualize_only:
        visualize_best_agent(args.visualize_only)
    
    elif args.with_openevolve:
        best_score, best_path = run_with_openevolve(iterations=args.iterations)
        
        if args.visualize:
            visualize_best_agent(best_path)
    
    else:
        best_score, best_path = simple_evolution_loop(
            iterations=args.iterations,
            num_games_per_eval=3
        )
        
        if args.visualize:
            visualize_best_agent(best_path)
