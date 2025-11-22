"""
Tetris Environment for AlphaEvolve
A simple but complete Tetris implementation using Gymnasium
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from typing import Tuple, Optional, Dict, Any


# Tetromino shapes (4x4 matrices, 1 = filled, 0 = empty)
SHAPES = {
    'I': np.array([[0, 0, 0, 0],
                   [1, 1, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]),
    'O': np.array([[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]]),
    'T': np.array([[0, 0, 0, 0],
                   [0, 1, 0, 0],
                   [1, 1, 1, 0],
                   [0, 0, 0, 0]]),
    'S': np.array([[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [1, 1, 0, 0],
                   [0, 0, 0, 0]]),
    'Z': np.array([[0, 0, 0, 0],
                   [1, 1, 0, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]]),
    'J': np.array([[0, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 1, 0, 0],
                   [1, 1, 0, 0]]),
    'L': np.array([[0, 0, 0, 0],
                   [0, 1, 0, 0],
                   [0, 1, 0, 0],
                   [0, 1, 1, 0]])
}

SHAPE_NAMES = list(SHAPES.keys())


class TetrisEnv(gym.Env):
    """
    Tetris Environment
    
    Observation: 
        - Board state: (20, 10) binary array
        - Current piece: (4, 4) binary array
        - Next piece: (4, 4) binary array
        - Current position: (x, y)
        - Total flattened: 20*10 + 4*4 + 4*4 + 2 = 234 values
    
    Actions:
        0: Move left
        1: Move right
        2: Rotate clockwise
        3: Rotate counter-clockwise
        4: Soft drop (move down faster)
        5: Hard drop (instant drop)
    """
    
    metadata = {'render_modes': ['human', 'rgb_array']}
    
    def __init__(self, width: int = 10, height: int = 20, render_mode: Optional[str] = None):
        super().__init__()
        
        self.width = width
        self.height = height
        self.render_mode = render_mode
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(6)
        
        # Observation: board + current piece + next piece + position
        obs_size = (height * width) + (4 * 4) + (4 * 4) + 2
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(obs_size,), dtype=np.float32
        )
        
        # Game state
        self.board = None
        self.current_piece = None
        self.current_shape_name = None
        self.next_piece = None
        self.next_shape_name = None
        self.piece_x = 0
        self.piece_y = 0
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.steps_since_drop = 0
        self.drop_interval = 30  # Piece falls every N steps
        
    def reset(self, seed: Optional[int] = None, options: Optional[Dict] = None) -> Tuple[np.ndarray, Dict]:
        super().reset(seed=seed)
        
        # Initialize empty board
        self.board = np.zeros((self.height, self.width), dtype=np.int8)
        
        # Spawn first pieces
        self.next_shape_name = self.np_random.choice(SHAPE_NAMES)
        self.next_piece = SHAPES[self.next_shape_name].copy()
        self._spawn_piece()
        
        self.score = 0
        self.lines_cleared = 0
        self.pieces_placed = 0
        self.game_over = False
        self.steps_since_drop = 0
        
        return self._get_observation(), {}
    
    def _spawn_piece(self):
        """Spawn a new piece at the top center"""
        self.current_shape_name = self.next_shape_name
        self.current_piece = self.next_piece.copy()
        
        self.next_shape_name = self.np_random.choice(SHAPE_NAMES)
        self.next_piece = SHAPES[self.next_shape_name].copy()
        
        self.piece_x = self.width // 2 - 2
        self.piece_y = 0
        
        # Check if spawn position is valid (game over if not)
        if not self._is_valid_position(self.piece_x, self.piece_y, self.current_piece):
            self.game_over = True
    
    def _rotate_piece(self, piece: np.ndarray, clockwise: bool = True) -> np.ndarray:
        """Rotate piece 90 degrees"""
        if clockwise:
            return np.rot90(piece, k=-1)
        else:
            return np.rot90(piece, k=1)
    
    def _is_valid_position(self, x: int, y: int, piece: np.ndarray) -> bool:
        """Check if piece can be placed at given position"""
        for i in range(4):
            for j in range(4):
                if piece[i, j]:
                    board_x = x + j
                    board_y = y + i
                    
                    # Check boundaries
                    if board_x < 0 or board_x >= self.width or board_y >= self.height:
                        return False
                    
                    # Check collision with existing pieces (but allow y < 0 for spawning)
                    if board_y >= 0 and self.board[board_y, board_x]:
                        return False
        
        return True
    
    def _lock_piece(self):
        """Lock current piece to the board"""
        for i in range(4):
            for j in range(4):
                if self.current_piece[i, j]:
                    board_x = self.piece_x + j
                    board_y = self.piece_y + i
                    if 0 <= board_y < self.height and 0 <= board_x < self.width:
                        self.board[board_y, board_x] = 1
        
        self.pieces_placed += 1
        
        # Clear lines and calculate score
        lines = self._clear_lines()
        if lines > 0:
            self.lines_cleared += lines
            # Tetris scoring: 40, 100, 300, 1200 for 1, 2, 3, 4 lines
            line_scores = [0, 40, 100, 300, 1200]
            self.score += line_scores[min(lines, 4)]
        
        # Spawn new piece
        self._spawn_piece()
    
    def _clear_lines(self) -> int:
        """Clear completed lines and return count"""
        lines_cleared = 0
        y = self.height - 1
        
        while y >= 0:
            if np.all(self.board[y]):
                # Remove line and shift down
                self.board = np.delete(self.board, y, axis=0)
                self.board = np.vstack([np.zeros((1, self.width), dtype=np.int8), self.board])
                lines_cleared += 1
            else:
                y -= 1
        
        return lines_cleared
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict]:
        """Execute one step"""
        if self.game_over:
            return self._get_observation(), 0.0, True, False, {'score': self.score}
        
        reward = 0.0
        
        # Execute action
        if action == 0:  # Move left
            if self._is_valid_position(self.piece_x - 1, self.piece_y, self.current_piece):
                self.piece_x -= 1
        
        elif action == 1:  # Move right
            if self._is_valid_position(self.piece_x + 1, self.piece_y, self.current_piece):
                self.piece_x += 1
        
        elif action == 2:  # Rotate clockwise
            rotated = self._rotate_piece(self.current_piece, clockwise=True)
            if self._is_valid_position(self.piece_x, self.piece_y, rotated):
                self.current_piece = rotated
            # Try wall kicks
            elif self._is_valid_position(self.piece_x - 1, self.piece_y, rotated):
                self.current_piece = rotated
                self.piece_x -= 1
            elif self._is_valid_position(self.piece_x + 1, self.piece_y, rotated):
                self.current_piece = rotated
                self.piece_x += 1
        
        elif action == 3:  # Rotate counter-clockwise
            rotated = self._rotate_piece(self.current_piece, clockwise=False)
            if self._is_valid_position(self.piece_x, self.piece_y, rotated):
                self.current_piece = rotated
            # Try wall kicks
            elif self._is_valid_position(self.piece_x - 1, self.piece_y, rotated):
                self.current_piece = rotated
                self.piece_x -= 1
            elif self._is_valid_position(self.piece_x + 1, self.piece_y, rotated):
                self.current_piece = rotated
                self.piece_x += 1
        
        elif action == 4:  # Soft drop
            if self._is_valid_position(self.piece_x, self.piece_y + 1, self.current_piece):
                self.piece_y += 1
                reward += 0.01  # Small reward for soft drop
            else:
                self._lock_piece()
        
        elif action == 5:  # Hard drop
            while self._is_valid_position(self.piece_x, self.piece_y + 1, self.current_piece):
                self.piece_y += 1
                reward += 0.02  # Reward for hard drop
            self._lock_piece()
        
        # Gravity: piece falls automatically
        self.steps_since_drop += 1
        if self.steps_since_drop >= self.drop_interval:
            self.steps_since_drop = 0
            if self._is_valid_position(self.piece_x, self.piece_y + 1, self.current_piece):
                self.piece_y += 1
            else:
                self._lock_piece()
        
        # Reward for staying alive and clearing lines
        reward += self.score * 0.1  # Reward proportional to score
        
        info = {
            'score': self.score,
            'lines_cleared': self.lines_cleared,
            'pieces_placed': self.pieces_placed
        }
        
        return self._get_observation(), reward, self.game_over, False, info
    
    def _get_observation(self) -> np.ndarray:
        """Get current observation as flat array"""
        # Flatten board
        board_flat = self.board.flatten()
        
        # Flatten current and next pieces
        current_piece_flat = self.current_piece.flatten()
        next_piece_flat = self.next_piece.flatten()
        
        # Position (normalized)
        position = np.array([self.piece_x / self.width, self.piece_y / self.height])
        
        # Concatenate all
        obs = np.concatenate([
            board_flat,
            current_piece_flat,
            next_piece_flat,
            position
        ]).astype(np.float32)
        
        return obs
    
    def render(self):
        """Render the game state"""
        if self.render_mode == 'human':
            # Create display board
            display = self.board.copy()
            
            # Add current piece to display
            for i in range(4):
                for j in range(4):
                    if self.current_piece[i, j]:
                        board_x = self.piece_x + j
                        board_y = self.piece_y + i
                        if 0 <= board_y < self.height and 0 <= board_x < self.width:
                            display[board_y, board_x] = 2  # Different value for current piece
            
            # Print board
            print("\n" + "=" * (self.width + 2))
            for row in display:
                print("|" + "".join("█" if cell == 1 else "▓" if cell == 2 else " " for cell in row) + "|")
            print("=" * (self.width + 2))
            print(f"Score: {self.score} | Lines: {self.lines_cleared} | Pieces: {self.pieces_placed}")
    
    def close(self):
        pass


if __name__ == "__main__":
    # Test the environment
    env = TetrisEnv(render_mode='human')
    obs, info = env.reset()
    
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    
    # Random agent test
    for _ in range(100):
        action = env.action_space.sample()
        obs, reward, done, truncated, info = env.step(action)
        env.render()
        
        if done:
            print("\nGame Over!")
            print(f"Final Score: {info['score']}")
            print(f"Lines Cleared: {info['lines_cleared']}")
            print(f"Pieces Placed: {info['pieces_placed']}")
            break
