"""
Script to run RICE experiments on a single environment.
This script loads a pre-trained policy and applies the RICE refinement algorithm.
"""

import argparse
import json
import os
import gymnasium as gym
from stable_baselines3 import PPO

from rice.rice_algorithm import RICEAlgorithm
from rice.baselines import PPOFineTuning, StateMaskR, SimplifiedJSRL
from rice.environments import create_mujoco_env


def load_pretrained_policy(env_name: str, sparse_reward: bool = False):
    """Load pre-trained policy from disk."""
    suffix = "_sparse" if sparse_reward else "_dense"
    model_path = f"pretrained_policies/{env_name}{suffix}.zip"
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Pre-trained policy not found at {model_path}")
    
    # Create environment to load policy
    env = create_mujoco_env(env_name, sparse_reward=sparse_reward)
    model = PPO.load(model_path, env=env)
    
    return model, env


def run_rice_experiment(env_name: str, sparse_reward: bool, output_file: str):
    """Run complete RICE experiment including baselines."""
    print(f"Running RICE experiment for {env_name} (sparse={sparse_reward})")
    
    # Load pre-trained policy
    pretrained_policy, env = load_pretrained_policy(env_name, sparse_reward)
    
    # Determine hyperparameters based on environment (from Table 3 in paper)
    if env_name == "Hopper-v5":
        reset_prob = 0.25
        rnd_lambda = 0.001
    elif env_name == "Walker2d-v5":
        reset_prob = 0.25
        rnd_lambda = 0.01
    elif env_name == "Reacher-v4":
        reset_prob = 0.50
        rnd_lambda = 0.001
    elif env_name == "HalfCheetah-v5":
        reset_prob = 0.50
        rnd_lambda = 0.01
    else:
        reset_prob = 0.25
        rnd_lambda = 0.01
    
    results = {
        'environment': env_name,
        'sparse_reward': sparse_reward,
        'hyperparameters': {
            'reset_probability': reset_prob,
            'rnd_lambda': rnd_lambda,
            'statemask_alpha': 0.0001
        }
    }
    
    # Run RICE algorithm
    print("Running RICE algorithm...")
    rice_algo = RICEAlgorithm(
        pretrained_policy=pretrained_policy,
        env=env,
        statemask_alpha=0.0001,
        rnd_lambda=rnd_lambda,
        reset_probability=reset_prob,
        statemask_kwargs={'ppo_epochs': 4, 'batch_size': 32},
        rnd_kwargs={'lr': 1e-4},
        ppo_kwargs={'learning_rate': 3e-4, 'n_steps': 2048, 'batch_size': 64}
    )
    
    rice_results = rice_algo.run_full_rice_pipeline(
        statemask_iterations=50,  # Reduced for faster execution
        critical_trajectories=50,  # Reduced for faster execution
        refinement_timesteps=20000,  # Reduced for faster execution
        evaluation_episodes=5  # Reduced for faster execution
    )
    
    results['rice'] = rice_results
    
    # Run baseline comparisons
    print("Running baseline comparisons...")
    
    # PPO Fine-tuning
    ppo_finetune = PPOFineTuning(pretrained_policy, env, lr=1e-5)
    ppo_agent = ppo_finetune.train(total_timesteps=20000)
    ppo_reward, ppo_std = ppo_finetune.evaluate(n_episodes=5)
    results['ppo_finetune'] = {'mean': ppo_reward, 'std': ppo_std}
    
    # StateMask-R
    # Note: This requires the statemask_trainer to be trained first
    statemask_r = StateMaskR(pretrained_policy, rice_algo.statemask_trainer, env)
    critical_states = rice_algo.critical_states  # Use critical states from RICE
    if len(critical_states) > 0:
        statemask_r_agent = statemask_r.train_from_critical_states(critical_states, total_timesteps=20000)
        statemask_r_reward, statemask_r_std = statemask_r.evaluate(n_episodes=5)
        results['statemask_r'] = {'mean': statemask_r_reward, 'std': statemask_r_std}
    else:
        results['statemask_r'] = {'mean': 0.0, 'std': 0.0}
    
    # Simplified JSRL
    jsrl = SimplifiedJSRL(pretrained_policy, env, exploration_horizon=50)
    jsrl_agent = jsrl.train_with_curriculum(total_timesteps=20000)
    jsrl_reward, jsrl_std = jsrl.evaluate(n_episodes=5)
    results['jsrl'] = {'mean': jsrl_reward, 'std': jsrl_std}
    
    # Save results
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"Results saved to {output_file}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RICE experiment")
    parser.add_argument("--env", type=str, required=True, help="Environment name")
    parser.add_argument("--sparse", type=bool, default=False, help="Use sparse reward")
    parser.add_argument("--output", type=str, required=True, help="Output file path")
    
    args = parser.parse_args()
    
    run_rice_experiment(args.env, args.sparse, args.output)
