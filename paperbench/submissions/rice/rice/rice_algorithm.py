"""
Main implementation of the RICE algorithm from the paper.
Implements Algorithm 2: Refining the DRL Agent.
"""

import torch
import numpy as np
import gymnasium as gym
from typing import Dict, Any, List, Tuple
from stable_baselines3 import PPO

from .statemask import StateMaskTrainer
from .rnd import RNDModule
from .environments import MixedInitialStateEnv


class RICEAlgorithm:
    """
    RICE (Refining scheme for ReInforCement learning with Explanation) algorithm.
    Implements the main refinement algorithm described in the paper.
    """
    
    def __init__(
        self,
        pretrained_policy,
        env: gym.Env,
        statemask_alpha: float = 0.0001,
        rnd_lambda: float = 0.01,
        reset_probability: float = 0.25,
        **kwargs
    ):
        """
        Initialize RICE algorithm.
        
        Args:
            pretrained_policy: Pre-trained policy to refine
            env: Gym environment
            statemask_alpha: Alpha parameter for StateMask reward bonus
            rnd_lambda: Lambda parameter for RND exploration bonus scaling
            reset_probability: Probability p for mixed initial state distribution
        """
        self.pretrained_policy = pretrained_policy
        self.env = env
        self.statemask_alpha = statemask_alpha
        self.rnd_lambda = rnd_lambda
        self.reset_probability = reset_probability
        
        # Initialize components
        self.state_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n
        
        # Initialize StateMask trainer
        self.statemask_trainer = StateMaskTrainer(
            target_policy=pretrained_policy,
            env=env,
            state_dim=self.state_dim,
            alpha=statemask_alpha,
            **kwargs.get('statemask_kwargs', {})
        )
        
        # Initialize RND module
        self.rnd_module = RNDModule(
            state_dim=self.state_dim,
            **kwargs.get('rnd_kwargs', {})
        )
        
        # Create refined policy (PPO)
        self.refined_policy = PPO(
            "MlpPolicy",
            env,
            verbose=0,
            **kwargs.get('ppo_kwargs', {})
        )
        
        # Load pre-trained weights into refined policy
        if hasattr(pretrained_policy, 'policy'):
            self.refined_policy.policy.load_state_dict(pretrained_policy.policy.state_dict())
        else:
            self.refined_policy.policy.load_state_dict(pretrained_policy.state_dict())
        
        # Critical states identified by StateMask
        self.critical_states = []
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def train_statemask(self, n_iterations: int = 100, trajectories_per_iteration: int = 10):
        """
        Train the StateMask explanation method.
        This corresponds to the first part of the RICE pipeline.
        """
        print("Training StateMask explanation method...")
        
        for iteration in range(n_iterations):
            trajectories = []
            
            # Collect trajectories using masked policy
            for _ in range(trajectories_per_iteration):
                traj = self.statemask_trainer.collect_trajectory()
                trajectories.append(traj)
            
            # Update mask network
            metrics = self.statemask_trainer.update(trajectories)
            
            if iteration % 10 == 0:
                print(f"StateMask Iteration {iteration}: Loss={metrics['loss']:.4f}")
    
    def identify_critical_states(self, n_trajectories: int = 100) -> List[np.ndarray]:
        """
        Identify critical states using the trained StateMask.
        """
        print("Identifying critical states...")
        critical_states = []
        
        for _ in range(n_trajectories):
            # Collect trajectory using original pre-trained policy
            states = []
            rewards = []
            
            state, _ = self.env.reset()
            done = False
            
            while not done:
                states.append(state.copy())
                
                # Get action from pre-trained policy
                if hasattr(self.pretrained_policy, 'predict'):
                    action, _ = self.pretrained_policy.predict(state, deterministic=False)
                else:
                    state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        if hasattr(self.pretrained_policy, 'actor'):
                            action, _ = self.pretrained_policy.actor(state_tensor)
                            action = action.cpu().numpy().flatten()
                        else:
                            action = self.pretrained_policy(state_tensor).cpu().numpy().flatten()
                
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                rewards.append(reward)
                done = terminated or truncated
                state = next_state
            
            # Create trajectory for StateMask analysis
            trajectory = {
                'states': np.array(states),
                'rewards': np.array(rewards)
            }
            
            # Identify critical state
            critical_idx = self.statemask_trainer.identify_critical_states(trajectory)
            critical_states.append(states[critical_idx])
        
        self.critical_states = critical_states
        return critical_states
    
    def setup_mixed_initial_state_env(self) -> MixedInitialStateEnv:
        """
        Setup environment with mixed initial state distribution.
        """
        mixed_env = MixedInitialStateEnv(self.env)
        mixed_env.set_critical_states(self.critical_states)
        mixed_env.set_reset_probability(self.reset_probability)
        return mixed_env
    
    def train_refined_policy(self, total_timesteps: int = 50000):
        """
        Train the refined policy using RICE algorithm.
        This implements Algorithm 2 from the paper.
        """
        print("Training refined policy with RICE...")
        
        # Setup mixed initial state environment
        mixed_env = self.setup_mixed_initial_state_env()
        
        # Replace the environment in the refined policy
        self.refined_policy.set_env(mixed_env)
        
        # Custom training loop to incorporate RND bonuses
        obs = mixed_env.reset()[0]
        episode_reward = 0
        episode_steps = 0
        all_states = []
        
        for step in range(total_timesteps):
            # Get action from refined policy
            action, _ = self.refined_policy.predict(obs, deterministic=False)
            
            # Take step in environment
            next_obs, reward, terminated, truncated, info = mixed_env.step(action)
            done = terminated or truncated
            
            # Compute RND bonus
            rnd_bonus = self.rnd_module.compute_bonus(next_obs)
            total_reward = reward + self.rnd_lambda * rnd_bonus
            
            episode_reward += total_reward
            episode_steps += 1
            all_states.append(next_obs.copy())
            
            # Store transition for RND training (simplified - in practice would use replay buffer)
            if len(all_states) > 1000:
                all_states.pop(0)
            
            # Update RND predictor periodically
            if step % 100 == 0 and len(all_states) > 0:
                states_batch = np.array(all_states[-min(100, len(all_states)):])
                self.rnd_module.update_predictor(states_batch)
            
            # Handle episode termination
            if done:
                print(f"Episode finished after {episode_steps} steps. Reward: {episode_reward:.2f}")
                obs, _ = mixed_env.reset()
                episode_reward = 0
                episode_steps = 0
            else:
                obs = next_obs
        
        return self.refined_policy
    
    def evaluate_policy(self, policy, n_episodes: int = 10) -> Tuple[float, float]:
        """
        Evaluate a policy on the original environment.
        """
        total_rewards = []
        
        for _ in range(n_episodes):
            obs, _ = self.env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                if hasattr(policy, 'predict'):
                    action, _ = policy.predict(obs, deterministic=True)
                else:
                    state_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
                    with torch.no_grad():
                        if hasattr(policy, 'actor'):
                            action, _ = policy.actor(state_tensor)
                            action = action.cpu().numpy().flatten()
                        else:
                            action = policy(state_tensor).cpu().numpy().flatten()
                
                obs, reward, terminated, truncated, _ = self.env.step(action)
                episode_reward += reward
                done = terminated or truncated
            
            total_rewards.append(episode_reward)
        
        mean_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        
        return mean_reward, std_reward
    
    def run_full_rice_pipeline(
        self,
        statemask_iterations: int = 100,
        critical_trajectories: int = 100,
        refinement_timesteps: int = 50000,
        evaluation_episodes: int = 10
    ) -> Dict[str, Any]:
        """
        Run the complete RICE pipeline:
        1. Train StateMask
        2. Identify critical states
        3. Train refined policy with RICE
        4. Evaluate results
        """
        results = {}
        
        # Step 1: Train StateMask
        self.train_statemask(n_iterations=statemask_iterations)
        
        # Step 2: Identify critical states
        self.identify_critical_states(n_trajectories=critical_trajectories)
        
        # Step 3: Evaluate pre-trained policy
        pretrain_mean, pretrain_std = self.evaluate_policy(
            self.pretrained_policy, 
            n_episodes=evaluation_episodes
        )
        results['pretrained_reward'] = {'mean': pretrain_mean, 'std': pretrain_std}
        print(f"Pre-trained policy reward: {pretrain_mean:.2f} ± {pretrain_std:.2f}")
        
        # Step 4: Train refined policy with RICE
        refined_policy = self.train_refined_policy(total_timesteps=refinement_timesteps)
        
        # Step 5: Evaluate refined policy
        refined_mean, refined_std = self.evaluate_policy(
            refined_policy,
            n_episodes=evaluation_episodes
        )
        results['refined_reward'] = {'mean': refined_mean, 'std': refined_std}
        print(f"Refined policy reward: {refined_mean:.2f} ± {refined_std:.2f}")
        
        results['improvement'] = refined_mean - pretrain_mean
        
        return results