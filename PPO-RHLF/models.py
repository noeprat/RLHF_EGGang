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
	Neural network to predict rewards based on state observations
	"""
	def __init__(self, input_dims, alpha):
		super(RewardModel, self).__init__()
		self.reward_net = nn.Sequential(
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
		return self.reward_net(state)