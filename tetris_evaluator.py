"""
Evaluator for Tetris Agent
Runs multiple games and computes performance metrics
"""
import numpy as np
from typing import Dict, List
import sys
import importlib.util


def load_evolved_program(program_path: str):
    """Dynamically load the evolved program module"""
    spec = importlib.util.spec_from_file_location("evolved_agent", program_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["evolved_agent"] = module
    spec.loader.exec_module(module)
    return module


def evaluate_agent(program_path: str, 
                   num_games: int = 10,
                   max_steps: int = 5000,
                   render: bool = False) -> Dict[str, float]:
    """
    Evaluate evolved Tetris agent across multiple games
    
    Args:
        program_path: Path to the evolved agent code
        num_games: Number of games to play
        max_steps: Maximum steps per game
        render: Whether to render games
    
    Returns:
        Dictionary of performance metrics
    """
    # Import here to avoid circular dependencies
    from tetris_env import TetrisEnv
    
    # Load evolved code
    try:
        evolved_module = load_evolved_program(program_path)
        EvolvedAgent = evolved_module.EvolvedTetrisAgent
    except Exception as e:
        print(f"Error loading evolved program: {e}")
        return {
            'score': 0.0,
            'avg_score': 0.0,
            'max_score': 0.0,
            'avg_lines': 0.0,
            'max_lines': 0.0,
            'avg_pieces': 0.0,
            'avg_steps': 0.0,
            'success_rate': 0.0,
            'error': str(e)
        }
    
    scores = []
    lines_cleared = []
    pieces_placed = []
    steps_survived = []
    
    for game_idx in range(num_games):
        try:
            env = TetrisEnv(render_mode='human' if render else None)
            agent = EvolvedAgent()
            
            obs, info = env.reset(seed=game_idx)  # Use different seed for each game
            agent.reset()
            
            total_reward = 0
            
            for step in range(max_steps):
                action = agent.get_action(obs)
                obs, reward, done, truncated, info = env.step(action)
                total_reward += reward
                
                if render and step % 20 == 0:
                    env.render()
                
                if done or truncated:
                    break
            
            scores.append(info['score'])
            lines_cleared.append(info['lines_cleared'])
            pieces_placed.append(info['pieces_placed'])
            steps_survived.append(step + 1)
            
            env.close()
            
            if render:
                print(f"\nGame {game_idx + 1}/{num_games}")
                print(f"Score: {info['score']}, Lines: {info['lines_cleared']}, "
                      f"Pieces: {info['pieces_placed']}, Steps: {step + 1}")
        
        except Exception as e:
            print(f"Error in game {game_idx}: {e}")
            scores.append(0)
            lines_cleared.append(0)
            pieces_placed.append(0)
            steps_survived.append(0)
    
    # Compute aggregate metrics
    results = {
        'avg_score': np.mean(scores),
        'max_score': np.max(scores) if scores else 0,
        'std_score': np.std(scores) if len(scores) > 1 else 0,
        'avg_lines': np.mean(lines_cleared),
        'max_lines': np.max(lines_cleared) if lines_cleared else 0,
        'avg_pieces': np.mean(pieces_placed),
        'max_pieces': np.max(pieces_placed) if pieces_placed else 0,
        'avg_steps': np.mean(steps_survived),
        'success_rate': sum(1 for s in scores if s > 0) / len(scores) if scores else 0,
        
        # Primary score for AlphaEvolve (weighted combination)
        'score': (
            np.mean(scores) * 1.0 +  # Average score
            np.mean(lines_cleared) * 10.0 +  # Lines cleared (high weight)
            np.mean(pieces_placed) * 1.0 +  # Pieces placed
            np.mean(steps_survived) * 0.01  # Survival time
        )
    }
    
    return results


def run_evaluation(program_path: str = "/home/claude/tetris_agent.py") -> Dict[str, float]:
    """
    Main evaluation function for AlphaEvolve
    
    This is the function that AlphaEvolve will call to score each evolved program.
    """
    results = evaluate_agent(
        program_path=program_path,
        num_games=5,  # Run 5 games per evaluation
        max_steps=3000,  # Max 3000 steps per game
        render=False
    )
    
    print(f"Evaluation Results:")
    print(f"  Primary Score: {results['score']:.2f}")
    print(f"  Avg Game Score: {results['avg_score']:.2f}")
    print(f"  Avg Lines: {results['avg_lines']:.2f}")
    print(f"  Avg Pieces: {results['avg_pieces']:.2f}")
    print(f"  Avg Steps: {results['avg_steps']:.2f}")
    
    return results


if __name__ == "__main__":
    import sys
    
    program_path = sys.argv[1] if len(sys.argv) > 1 else "/home/claude/tetris_agent.py"
    
    print(f"Evaluating: {program_path}\n")
    results = run_evaluation(program_path)
    
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    for key, value in results.items():
        print(f"{key:20s}: {value:10.2f}")
