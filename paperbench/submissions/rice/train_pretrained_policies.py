"""
Script to train pre-trained policies for MuJoCo environments.
These policies will be used as the starting point for RICE refinement.
"""

import os
import gymnasium as gym
from stable_baselines3 import PPO
from rice.environments import create_mujoco_env


def train_pretrained_policy(env_name: str, sparse_reward: bool = False, total_timesteps: int = 1000000):
    """
    Train a pre-trained PPO policy for a given environment.
    
    Args:
        env_name: Name of the MuJoCo environment
        sparse_reward: Whether to use sparse reward version
        total_timesteps: Total training timesteps
    """
    print(f"Training pre-trained policy for {env_name} (sparse={sparse_reward})...")
    
    # Create environment
    env = create_mujoco_env(env_name, sparse_reward=sparse_reward)
    
    # Create PPO agent
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5
    )
    
    # Train the agent
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model
    suffix = "_sparse" if sparse_reward else "_dense"
    model_path = f"pretrained_policies/{env_name}{suffix}.zip"
    os.makedirs("pretrained_policies", exist_ok=True)
    model.save(model_path)
    
    print(f"Saved pre-trained policy to {model_path}")
    return model_path


if __name__ == "__main__":
    # Train policies for all four MuJoCo environments (using v5/v4 versions compatible with mujoco package)
    environments = [
        "Hopper-v5",
        "Walker2d-v5", 
        "Reacher-v4",
        "HalfCheetah-v5"
    ]
    
    # Train both dense and sparse reward versions where applicable
    for env_name in environments:
        # Dense reward version (all environments)
        train_pretrained_policy(env_name, sparse_reward=False, total_timesteps=500000)
        
        # Sparse reward version (except Reacher, which doesn't have sparse version in paper)
        if env_name != "Reacher-v4":
            train_pretrained_policy(env_name, sparse_reward=True, total_timesteps=500000)
