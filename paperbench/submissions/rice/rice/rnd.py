"""
Implementation of the Random Network Distillation (RND) module for exploration bonus.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple


class RNDNetwork(nn.Module):
    """
    Base network for RND - can be used for both target and predictor networks.
    """
    
    def __init__(self, input_dim: int, hidden_dim: int = 64, output_dim: int = 64):
        super(RNDNetwork, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class RNDModule:
    """
    Random Network Distillation module for computing exploration bonuses.
    Implements the RND component used in Algorithm 2 of the RICE paper.
    """
    
    def __init__(
        self,
        state_dim: int,
        hidden_dim: int = 64,
        output_dim: int = 64,
        lr: float = 1e-4,
        device: str = "cuda" if torch.cuda.is_available() else "cpu"
    ):
        """
        Initialize RND module.
        
        Args:
            state_dim: Dimension of the state space
            hidden_dim: Hidden dimension for networks
            output_dim: Output dimension for feature embedding
            lr: Learning rate for predictor network
            device: Device to run on
        """
        self.device = torch.device(device)
        
        # Target network (fixed after initialization)
        self.target_network = RNDNetwork(state_dim, hidden_dim, output_dim).to(self.device)
        # Freeze target network parameters
        for param in self.target_network.parameters():
            param.requires_grad = False
            
        # Predictor network (trained to match target)
        self.predictor_network = RNDNetwork(state_dim, hidden_dim, output_dim).to(self.device)
        self.optimizer = optim.Adam(self.predictor_network.parameters(), lr=lr)
        
        # Normalization statistics
        self.running_mean = torch.zeros(output_dim, device=self.device)
        self.running_var = torch.ones(output_dim, device=self.device)
        self.count = 0
        
    def compute_bonus(self, state: np.ndarray, update_stats: bool = True) -> float:
        """
        Compute RND bonus for a given state.
        
        Args:
            state: Input state array
            update_stats: Whether to update normalization statistics
            
        Returns:
            RND bonus value
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            target_features = self.target_network(state_tensor)
            predictor_features = self.predictor_network(state_tensor)
            
            # Compute raw bonus
            raw_bonus = torch.mean((target_features - predictor_features) ** 2).item()
            
            # Normalize bonus
            normalized_bonus = self._normalize_bonus(raw_bonus, update_stats)
            
        return normalized_bonus
    
    def _normalize_bonus(self, bonus: float, update_stats: bool = True) -> float:
        """
        Normalize the RND bonus using running statistics.
        """
        if update_stats:
            self.count += 1
            delta = bonus - self.running_mean
            self.running_mean += delta / self.count
            delta2 = bonus - self.running_mean
            self.running_var += delta * delta2
            
        std = torch.sqrt(self.running_var / max(1, self.count))
        normalized_bonus = (bonus - self.running_mean) / (std + 1e-8)
        
        return max(0.0, normalized_bonus.item())
    
    def update_predictor(self, states: np.ndarray) -> float:
        """
        Update the predictor network to match the target network.
        
        Args:
            states: Batch of states to train on
            
        Returns:
            Training loss
        """
        states_tensor = torch.FloatTensor(states).to(self.device)
        
        target_features = self.target_network(states_tensor)
        predictor_features = self.predictor_network(states_tensor)
        
        loss = torch.mean((target_features - predictor_features) ** 2)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def reset_normalization(self):
        """Reset normalization statistics."""
        self.running_mean.zero_()
        self.running_var.fill_(1.0)
        self.count = 0