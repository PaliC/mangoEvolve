"""
PufferLib Wrapper for Tetris Environment
Integrates our Tetris environment with PufferLib for efficient vectorization
"""
import numpy as np
try:
    import pufferlib
    import pufferlib.emulation
    PUFFERLIB_AVAILABLE = True
except ImportError:
    PUFFERLIB_AVAILABLE = False
    print("PufferLib not installed. Install with: pip install pufferlib")

from tetris_env import TetrisEnv


def make_tetris_env(render_mode=None, width=10, height=20):
    """
    Create a PufferLib-wrapped Tetris environment
    
    Args:
        render_mode: Rendering mode ('human', 'rgb_array', or None)
        width: Board width
        height: Board height
    
    Returns:
        PufferLib-wrapped environment
    """
    if not PUFFERLIB_AVAILABLE:
        # Return unwrapped env if PufferLib not available
        return TetrisEnv(width=width, height=height, render_mode=render_mode)
    
    env = TetrisEnv(width=width, height=height, render_mode=render_mode)
    puffer_env = pufferlib.emulation.GymnasiumPufferEnv(env=env)
    return puffer_env


def make_vectorized_env(num_envs=4, backend='serial', **env_kwargs):
    """
    Create vectorized Tetris environments using PufferLib
    
    Args:
        num_envs: Number of parallel environments
        backend: 'serial' or 'multiprocessing'
        **env_kwargs: Arguments passed to TetrisEnv
    
    Returns:
        Vectorized environment
    """
    if not PUFFERLIB_AVAILABLE:
        raise ImportError("PufferLib is required for vectorization")
    
    # Create environment creator function
    def env_creator():
        return make_tetris_env(**env_kwargs)
    
    # Select backend
    if backend == 'serial':
        Backend = pufferlib.vector.Serial
    elif backend == 'multiprocessing':
        Backend = pufferlib.vector.Multiprocessing
    else:
        raise ValueError(f"Unknown backend: {backend}")
    
    # Create vectorized environment
    vec_env = pufferlib.vector.make(
        env_creator,
        backend=Backend,
        num_envs=num_envs
    )
    
    return vec_env


class PufferTetrisRunner:
    """
    Runner for Tetris with PufferLib
    Handles batch execution of evolved agents
    """
    
    def __init__(self, num_envs=4, backend='serial'):
        self.num_envs = num_envs
        self.vec_env = make_vectorized_env(
            num_envs=num_envs,
            backend=backend,
            render_mode=None
        )
        
    def run_parallel_games(self, agent_class, num_steps=1000):
        """
        Run multiple games in parallel
        
        Args:
            agent_class: Class of evolved agent
            num_steps: Steps per environment
        
        Returns:
            List of results for each environment
        """
        # Create agents for each environment
        agents = [agent_class() for _ in range(self.num_envs)]
        
        # Reset all environments
        observations = self.vec_env.reset()
        
        # Reset all agents
        for agent in agents:
            agent.reset()
        
        total_rewards = np.zeros(self.num_envs)
        episode_info = [{'score': 0, 'lines_cleared': 0, 'pieces_placed': 0} 
                        for _ in range(self.num_envs)]
        
        for step in range(num_steps):
            # Get actions from all agents
            actions = np.array([
                agents[i].get_action(observations[i])
                for i in range(self.num_envs)
            ])
            
            # Step all environments
            observations, rewards, dones, infos = self.vec_env.step(actions)
            
            total_rewards += rewards
            
            # Update episode info
            for i in range(self.num_envs):
                if 'score' in infos[i]:
                    episode_info[i] = {
                        'score': infos[i].get('score', 0),
                        'lines_cleared': infos[i].get('lines_cleared', 0),
                        'pieces_placed': infos[i].get('pieces_placed', 0)
                    }
            
            # Check for done episodes
            if np.any(dones):
                break
        
        results = []
        for i in range(self.num_envs):
            results.append({
                'total_reward': total_rewards[i],
                **episode_info[i]
            })
        
        return results
    
    def close(self):
        """Close vectorized environment"""
        self.vec_env.close()


if __name__ == "__main__":
    print("Testing PufferLib Tetris Wrapper\n")
    
    # Test 1: Single environment
    print("Test 1: Single Environment")
    env = make_tetris_env()
    obs, info = env.reset()
    print(f"Observation shape: {obs.shape}")
    print(f"Action space: {env.action_space}")
    
    for _ in range(10):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        if done:
            break
    
    print(f"Game ended. Score: {info.get('score', 0)}\n")
    env.close()
    
    if PUFFERLIB_AVAILABLE:
        # Test 2: Vectorized environments
        print("Test 2: Vectorized Environments (4 parallel)")
        from tetris_agent import EvolvedTetrisAgent
        
        runner = PufferTetrisRunner(num_envs=4, backend='serial')
        results = runner.run_parallel_games(
            agent_class=EvolvedTetrisAgent,
            num_steps=500
        )
        
        print("\nResults from parallel games:")
        for i, result in enumerate(results):
            print(f"  Game {i+1}: Score={result['score']}, "
                  f"Lines={result['lines_cleared']}, "
                  f"Reward={result['total_reward']:.2f}")
        
        runner.close()
    else:
        print("\nSkipping vectorized test (PufferLib not available)")
    
    print("\n✓ All tests passed!")
