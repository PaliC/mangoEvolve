"""
Tetris AlphaEvolve Demo
Shows how to use the system without requiring OpenEvolve
"""

def demo_basic_evaluation():
    """Demo: Evaluate the basic agent"""
    print("="*70)
    print("DEMO 1: Basic Agent Evaluation")
    print("="*70)
    print("Running 3 games with the baseline agent...\n")
    
    from tetris_evaluator import evaluate_agent
    
    results = evaluate_agent(
        program_path="/home/claude/tetris_agent.py",
        num_games=3,
        max_steps=1000,
        render=False
    )
    
    print("\nResults:")
    print(f"  Primary Score: {results['score']:.2f}")
    print(f"  Avg Game Score: {results['avg_score']:.2f}")
    print(f"  Avg Lines Cleared: {results['avg_lines']:.2f}")
    print(f"  Avg Pieces Placed: {results['avg_pieces']:.2f}")
    print(f"  Avg Steps Survived: {results['avg_steps']:.2f}")
    print()


def demo_visualize_game():
    """Demo: Visualize a single game"""
    print("="*70)
    print("DEMO 2: Visualize Agent Playing")
    print("="*70)
    print("Watch the agent play one game...\n")
    
    from tetris_env import TetrisEnv
    from tetris_agent import EvolvedTetrisAgent
    
    env = TetrisEnv(render_mode='human')
    agent = EvolvedTetrisAgent()
    
    obs, info = env.reset(seed=42)
    agent.reset()
    
    for step in range(200):  # Play for 200 steps
        action = agent.get_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        
        if step % 20 == 0:
            env.render()
        
        if done:
            break
    
    env.render()
    print(f"\nGame ended after {step} steps")
    print(f"Score: {info['score']}")
    print(f"Lines: {info['lines_cleared']}")
    print(f"Pieces: {info['pieces_placed']}\n")
    env.close()


def demo_manual_modification():
    """Demo: Show how to manually improve the agent"""
    print("="*70)
    print("DEMO 3: Manual Agent Improvement")
    print("="*70)
    print("This shows what AlphaEvolve would do automatically:\n")
    
    print("Original heuristics in decide_action():")
    print("  - If complete_lines > 0: hard drop")
    print("  - If holes > 3: rotate")
    print("  - If x > width/2: move left")
    print("  - Else: move right")
    print()
    
    print("Possible improvements AlphaEvolve might discover:")
    print("  1. Better hole threshold (maybe holes > 5 is better?)")
    print("  2. Consider piece height in decisions")
    print("  3. Prefer moving toward columns with lower height")
    print("  4. Look ahead to next piece")
    print("  5. Different strategies for different pieces (I-piece vs T-piece)")
    print()
    
    print("Example improved logic:")
    print("""
    def decide_action(observation, width=10, height=20):
        # Parse observation
        board = observation[:board_size].reshape(height, width)
        current_piece = observation[board_size:board_size + piece_size].reshape(4, 4)
        
        # Compute heuristics
        heuristics = compute_heuristics(board, current_piece, next_piece, (x, y))
        
        # Evolved logic (better thresholds and conditions)
        if heuristics['complete_lines'] >= 2:
            return 5  # Hard drop for multi-line clears
        elif heuristics['holes'] > 5:
            return 2  # Rotate only if many holes
        else:
            # Move toward lower columns
            column_heights = [calculate_column_height(board, c) for c in range(width)]
            target_col = np.argmin(column_heights)
            if piece_x < target_col:
                return 1  # Move right
            else:
                return 0  # Move left
    """)
    print()


def demo_pufferlib_parallel():
    """Demo: Run parallel games with PufferLib (if available)"""
    print("="*70)
    print("DEMO 4: Parallel Execution with PufferLib")
    print("="*70)
    
    try:
        from tetris_pufferlib import PufferTetrisRunner
        from tetris_agent import EvolvedTetrisAgent
        
        print("Running 4 games in parallel...\n")
        
        runner = PufferTetrisRunner(num_envs=4, backend='serial')
        results = runner.run_parallel_games(
            agent_class=EvolvedTetrisAgent,
            num_steps=500
        )
        
        print("Results:")
        for i, result in enumerate(results):
            print(f"  Game {i+1}: Score={result['score']}, "
                  f"Lines={result['lines_cleared']}, "
                  f"Pieces={result['pieces_placed']}")
        
        runner.close()
        print()
        
    except ImportError:
        print("PufferLib not installed. Install with: pip install pufferlib")
        print("Skipping parallel execution demo.\n")


def demo_evolution_workflow():
    """Demo: Show the evolution workflow"""
    print("="*70)
    print("DEMO 5: Evolution Workflow (Conceptual)")
    print("="*70)
    print("""
Evolution Workflow with AlphaEvolve:

1. Start with initial agent code
   - evaluate_agent() → score = 36.60

2. LLM proposes modification (e.g., change hole threshold)
   - Old: if heuristics['holes'] > 3
   - New: if heuristics['holes'] > 7

3. Evaluate modified code
   - evaluate_agent() → score = 42.15 (better!)

4. Keep this version, LLM proposes another change
   - Add column height consideration

5. Evaluate again
   - evaluate_agent() → score = 38.90 (worse)

6. Discard, try different modification
   - Adjust evaluation weights

7. Repeat for N iterations (e.g., 50-100)
   - Keep evolving the best-performing code

8. Result: Agent that plays much better than baseline!

To run real evolution:
  python run_tetris_evolution.py --with-openevolve --iterations 50

Or manually with LLM API:
  - Read tetris_agent.py
  - Ask LLM to improve code in EVOLVE-BLOCK
  - Save modified version
  - Run evaluator
  - Repeat
""")


if __name__ == "__main__":
    import sys
    
    demos = {
        '1': ('Basic Evaluation', demo_basic_evaluation),
        '2': ('Visualize Game', demo_visualize_game),
        '3': ('Manual Modification', demo_manual_modification),
        '4': ('PufferLib Parallel', demo_pufferlib_parallel),
        '5': ('Evolution Workflow', demo_evolution_workflow),
    }
    
    if len(sys.argv) > 1:
        demo_num = sys.argv[1]
        if demo_num in demos:
            name, func = demos[demo_num]
            func()
        else:
            print(f"Unknown demo: {demo_num}")
    else:
        print("Tetris AlphaEvolve Demos")
        print("="*70)
        print("\nAvailable demos:")
        for num, (name, _) in demos.items():
            print(f"  {num}. {name}")
        print("\nUsage: python demo.py [1-5]")
        print("   Or: python demo.py  (runs all demos)")
        print()
        
        # Run all demos
        for num, (name, func) in demos.items():
            func()
            print("\n")
