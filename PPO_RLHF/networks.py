"""
	This file contains a neural network module for us to
	define our actor and critic networks in PPO.
"""

import torch
import torch.nn as nn
import torch.optim as optim

class ActorNetwork(nn.Module):
    """
    Neural network to predict the action probabilities of a state
    """
    def __init__(self, n_actions, input_dims, alpha):
        super(ActorNetwork, self).__init__()
        self.actor = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, n_actions),
            nn.Softmax(dim=-1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.actor(state)
	
class CriticNetwork(nn.Module):
    """
    Neural network to predict the value of a state
    """
    def __init__(self, input_dims, alpha):
        super(CriticNetwork, self).__init__()
        self.critic = nn.Sequential(
            nn.Linear(input_dims, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        return self.critic(state)
	
class RewardModel(nn.Module):
    """
    Neural network to predict rewards based on state and action.
    """
    def __init__(self, state_dim, action_dim, alpha):
        super(RewardModel, self).__init__()
        input_dim = state_dim + action_dim
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1)
        )
        self.apply(self._init_weights)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-4)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, state, action):
        # state: (batch, state_dim), action: (batch, action_dim)
        if len(state.shape) == 1:
            state = state.unsqueeze(0)
        if len(action.shape) == 1:
            action = action.unsqueeze(0)
        x = torch.cat([state, action], dim=-1)
        return self.network(x)