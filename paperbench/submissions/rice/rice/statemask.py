"""
Implementation of the StateMask explanation method from the RICE paper.
This implements Algorithm 1: Training the Mask Network.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict, Any
import gymnasium as gym


class MaskNetwork(nn.Module):
    """
    Mask network that outputs binary actions (0 or 1) to determine whether
    to blind the target agent at each time step.
    
    The mask network takes the current state as input and outputs the probability
    of blinding (outputting 1).
    """
    
    def __init__(self, state_dim: int, hidden_dim: int = 64):
        super(MaskNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid()  # Output probability of blinding (1)
        )
        
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the mask network.
        
        Args:
            state: Input state tensor of shape (batch_size, state_dim)
            
        Returns:
            Probability of blinding (outputting 1) of shape (batch_size, 1)
        """
        return self.network(state)
    
    def get_action(self, state: torch.Tensor, deterministic: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample action from the mask network.
        
        Args:
            state: Input state tensor
            deterministic: If True, always output the most likely action
            
        Returns:
            Tuple of (action, log_prob) where action is 0 or 1
        """
        prob_blind = self.forward(state)
        
        if deterministic:
            # Always choose the action with highest probability
            action = (prob_blind > 0.5).float()
            log_prob = torch.log(prob_blind + 1e-8) * action + torch.log(1 - prob_blind + 1e-8) * (1 - action)
        else:
            # Sample from Bernoulli distribution
            dist = torch.distributions.Bernoulli(probs=prob_blind)
            action = dist.sample()
            log_prob = dist.log_prob(action)
            
        return action, log_prob


class StateMaskTrainer:
    """
    Trainer for the StateMask explanation method.
    Implements Algorithm 1 from the paper.
    """
    
    def __init__(
        self,
        target_policy,
        env: gym.Env,
        state_dim: int,
        alpha: float = 0.0001,
        hidden_dim: int = 64,
        lr: float = 3e-4,
        ppo_epochs: int = 10,
        batch_size: int = 64,
        gamma: float = 0.99,
        clip_epsilon: float = 0.2
    ):
        """
        Initialize the StateMask trainer.
        
        Args:
            target_policy: The pre-trained policy to explain (callable that takes state and returns action)
            env: Gym environment
            state_dim: Dimension of the state space
            alpha: Bonus coefficient for blinding actions (α in the paper)
            hidden_dim: Hidden dimension for the mask network
            lr: Learning rate for the mask network
            ppo_epochs: Number of PPO epochs per iteration
            batch_size: Batch size for training
            gamma: Discount factor
            clip_epsilon: PPO clipping parameter
        """
        self.target_policy = target_policy
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.ppo_epochs = ppo_epochs
        self.batch_size = batch_size
        self.clip_epsilon = clip_epsilon
        
        # Initialize mask network
        self.mask_net = MaskNetwork(state_dim, hidden_dim)
        self.optimizer = optim.Adam(self.mask_net.parameters(), lr=lr)
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.mask_net.to(self.device)
        
    def _get_target_action(self, state: np.ndarray) -> np.ndarray:
        """Get action from the target policy."""
        # Convert to tensor if needed
        if isinstance(state, np.ndarray):
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        else:
            state_tensor = state.unsqueeze(0).to(self.device)
            
        with torch.no_grad():
            if hasattr(self.target_policy, 'actor'):
                # Handle stable-baselines3 policies
                action, _ = self.target_policy.actor(state_tensor)
                return action.cpu().numpy().flatten()
            else:
                # Handle custom policies
                action = self.target_policy(state_tensor)
                return action.cpu().numpy().flatten()
    
    def _apply_mask(self, target_action: np.ndarray, mask_action: float) -> np.ndarray:
        """
        Apply the mask to the target action.
        
        Args:
            target_action: Action from the target policy
            mask_action: Mask action (0 or 1)
            
        Returns:
            Final action after applying mask
        """
        if mask_action == 0:
            # Use target action
            return target_action
        else:
            # Use random action
            action_space = self.env.action_space
            if isinstance(action_space, gym.spaces.Box):
                return np.random.uniform(action_space.low, action_space.high)
            elif isinstance(action_space, gym.spaces.Discrete):
                return np.random.randint(action_space.n)
            else:
                raise NotImplementedError(f"Action space {type(action_space)} not supported")
    
    def collect_trajectory(self) -> Dict[str, np.ndarray]:
        """
        Collect a trajectory by running the masked policy.
        
        Returns:
            Dictionary containing states, mask_actions, rewards, etc.
        """
        states = []
        mask_actions = []
        target_actions = []
        final_actions = []
        rewards = []
        next_states = []
        log_probs = []
        
        state, _ = self.env.reset()
        done = False
        
        while not done:
            states.append(state.copy())
            
            # Get target action
            target_action = self._get_target_action(state)
            target_actions.append(target_action.copy())
            
            # Get mask action
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            mask_action, log_prob = self.mask_net.get_action(state_tensor)
            mask_action_val = mask_action.item()
            mask_actions.append(mask_action_val)
            log_probs.append(log_prob.item())
            
            # Apply mask to get final action
            final_action = self._apply_mask(target_action, mask_action_val)
            final_actions.append(final_action.copy())
            
            # Take step in environment
            next_state, reward, terminated, truncated, _ = self.env.step(final_action)
            done = terminated or truncated
            
            # Add bonus for blinding
            total_reward = reward + self.alpha * mask_action_val
            rewards.append(total_reward)
            next_states.append(next_state.copy())
            
            state = next_state
            
        return {
            'states': np.array(states),
            'mask_actions': np.array(mask_actions),
            'target_actions': np.array(target_actions),
            'final_actions': np.array(final_actions),
            'rewards': np.array(rewards),
            'next_states': np.array(next_states),
            'log_probs': np.array(log_probs)
        }
    
    def compute_advantages(self, rewards: np.ndarray, values: np.ndarray = None) -> np.ndarray:
        """
        Compute advantages using GAE or simple returns.
        For simplicity, we use simple discounted returns here.
        """
        returns = np.zeros_like(rewards)
        G = 0
        for t in reversed(range(len(rewards))):
            G = rewards[t] + self.gamma * G
            returns[t] = G
        return returns
    
    def update(self, trajectories: list) -> Dict[str, float]:
        """
        Update the mask network using PPO.
        
        Args:
            trajectories: List of trajectory dictionaries
            
        Returns:
            Dictionary of training metrics
        """
        # Flatten all trajectories
        all_states = np.concatenate([traj['states'] for traj in trajectories])
        all_mask_actions = np.concatenate([traj['mask_actions'] for traj in trajectories])
        all_rewards = np.concatenate([traj['rewards'] for traj in trajectories])
        all_log_probs = np.concatenate([traj['log_probs'] for traj in trajectories])
        
        # Compute returns (advantages)
        returns = self.compute_advantages(all_rewards)
        
        # Convert to tensors
        states_tensor = torch.FloatTensor(all_states).to(self.device)
        mask_actions_tensor = torch.FloatTensor(all_mask_actions).unsqueeze(1).to(self.device)
        returns_tensor = torch.FloatTensor(returns).unsqueeze(1).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(all_log_probs).unsqueeze(1).to(self.device)
        
        total_loss = 0
        total_actor_loss = 0
        total_entropy = 0
        
        # PPO updates
        for epoch in range(self.ppo_epochs):
            # Shuffle indices for mini-batch training
            indices = np.random.permutation(len(states_tensor))
            
            for i in range(0, len(indices), self.batch_size):
                batch_indices = indices[i:i+self.batch_size]
                batch_states = states_tensor[batch_indices]
                batch_mask_actions = mask_actions_tensor[batch_indices]
                batch_returns = returns_tensor[batch_indices]
                batch_old_log_probs = old_log_probs_tensor[batch_indices]
                
                # Get current log probs and entropy
                prob_blind = self.mask_net(batch_states)
                dist = torch.distributions.Bernoulli(probs=prob_blind)
                current_log_probs = dist.log_prob(batch_mask_actions)
                entropy = dist.entropy().mean()
                
                # Compute ratio
                ratio = torch.exp(current_log_probs - batch_old_log_probs)
                surr1 = ratio * batch_returns
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * batch_returns
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # Total loss
                loss = actor_loss - 0.01 * entropy  # Entropy bonus
                
                # Update
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                total_loss += loss.item()
                total_actor_loss += actor_loss.item()
                total_entropy += entropy.item()
        
        num_updates = self.ppo_epochs * (len(states_tensor) // self.batch_size)
        return {
            'loss': total_loss / num_updates,
            'actor_loss': total_actor_loss / num_updates,
            'entropy': total_entropy / num_updates
        }
    
    def identify_critical_states(self, trajectory: Dict[str, np.ndarray]) -> int:
        """
        Identify the most critical state in a trajectory.
        The critical state is the one with the lowest probability of blinding (highest importance).
        
        Args:
            trajectory: Trajectory dictionary
            
        Returns:
            Index of the most critical state
        """
        states = trajectory['states']
        state_tensor = torch.FloatTensor(states).to(self.device)
        
        with torch.no_grad():
            prob_blind = self.mask_net(state_tensor).cpu().numpy().flatten()
            # Lower probability of blinding means higher importance
            critical_idx = np.argmin(prob_blind)
            
        return critical_idx