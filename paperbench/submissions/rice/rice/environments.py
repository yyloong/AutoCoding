"""
Environment utilities and wrappers for RICE implementation.
"""

import gymnasium as gym
import numpy as np
from typing import Any, Dict, Tuple


class StateRecorderWrapper(gym.Wrapper):
    """
    Wrapper to record states during environment interaction.
    Useful for collecting trajectories for StateMask training.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.recorded_states = []
        self.current_trajectory = []
        
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """Reset environment and clear trajectory."""
        obs, info = self.env.reset(**kwargs)
        self.current_trajectory = [obs.copy()]
        return obs, info
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Step environment and record state."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.current_trajectory.append(obs.copy())
        return obs, reward, terminated, truncated, info
        
    def get_trajectory(self) -> np.ndarray:
        """Get the current trajectory."""
        return np.array(self.current_trajectory)
        
    def clear_trajectory(self):
        """Clear the current trajectory."""
        self.current_trajectory = []


class MixedInitialStateEnv(gym.Wrapper):
    """
    Environment wrapper that supports mixed initial state distribution.
    Allows resetting to either default initial states or provided critical states.
    """
    
    def __init__(self, env: gym.Env):
        super().__init__(env)
        self.critical_states = []
        self.reset_probability = 0.25  # Default p value from paper
        
    def set_critical_states(self, critical_states: list):
        """Set the list of critical states for mixed initialization."""
        self.critical_states = critical_states
        
    def set_reset_probability(self, p: float):
        """Set the probability of resetting to critical states."""
        self.reset_probability = p
        
    def reset(self, **kwargs) -> Tuple[Any, Dict]:
        """
        Reset to either default initial state or critical state based on probability.
        """
        if len(self.critical_states) > 0 and np.random.random() < self.reset_probability:
            # Reset to random critical state
            critical_state = self.critical_states[np.random.randint(len(self.critical_states))]
            
            # Try to set the environment state directly
            try:
                # For MuJoCo environments, we can set qpos and qvel
                if hasattr(self.env.unwrapped, 'set_state'):
                    qpos_size = self.env.unwrapped.model.nq
                    qvel_size = self.env.unwrapped.model.nv
                    
                    if len(critical_state) >= qpos_size + qvel_size:
                        qpos = critical_state[:qpos_size]
                        qvel = critical_state[qpos_size:qpos_size + qvel_size]
                        self.env.unwrapped.set_state(qpos, qvel)
                        obs = self.env.unwrapped._get_obs()
                        return obs, {}
                    else:
                        # State dimension mismatch, fall back to default reset
                        pass
                elif hasattr(self.env.unwrapped, 'state'):
                    # For other environments that support direct state setting
                    self.env.unwrapped.state = critical_state
                    obs = critical_state
                    return obs, {}
            except (AttributeError, ValueError, IndexError):
                # If direct state setting fails, fall back to default reset
                pass
        
        # Default reset
        return self.env.reset(**kwargs)


def create_mujoco_env(env_name: str, sparse_reward: bool = False) -> gym.Env:
    """
    Create MuJoCo environment with optional sparse reward modification.
    
    Args:
        env_name: Name of the MuJoCo environment (e.g., 'Hopper-v3')
        sparse_reward: Whether to use sparse reward version
        
    Returns:
        Configured gym environment
    """
    env = gym.make(env_name)
    
    if sparse_reward:
        # Apply sparse reward modifications based on paper descriptions
        if 'Hopper' in env_name or 'Walker2d' in env_name:
            # Sparse reward: only reward x position if x > 0.6
            env = SparseRewardWrapper(env, threshold=0.6, reward_type='x_position')
        elif 'HalfCheetah' in env_name:
            # Sparse reward: only reward x position if x > 5
            env = SparseRewardWrapper(env, threshold=5.0, reward_type='x_position')
        # Note: Reacher doesn't have sparse reward version in the paper
    
    return env


class SparseRewardWrapper(gym.Wrapper):
    """
    Wrapper to convert dense rewards to sparse rewards for MuJoCo environments.
    """
    
    def __init__(self, env: gym.Env, threshold: float = 0.6, reward_type: str = 'x_position'):
        super().__init__(env)
        self.threshold = threshold
        self.reward_type = reward_type
        
    def step(self, action: Any) -> Tuple[Any, float, bool, bool, Dict]:
        """Apply sparse reward logic."""
        obs, reward, terminated, truncated, info = self.env.step(action)
        
        # Extract relevant information for sparse reward
        if self.reward_type == 'x_position':
            # For Hopper, Walker2d, HalfCheetah - use x position
            x_position = obs[0]  # First element is usually x position in MuJoCo
            if x_position > self.threshold:
                sparse_reward = x_position
            else:
                sparse_reward = 0.0
        else:
            sparse_reward = reward
            
        return obs, sparse_reward, terminated, truncated, info