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
    Neural network to predict rewards based on state-action observations
    Enhanced with deeper architecture, dropout, and layer normalization
    """
    
    def __init__(self, input_dims, alpha):
        super(RewardModel, self).__init__()
        
        # First layer with layer normalization
        self.layer1 = nn.Sequential(
            nn.Linear(input_dims, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Second layer with layer normalization
        self.layer2 = nn.Sequential(
            nn.Linear(128, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Third layer with layer normalization
        self.layer3 = nn.Sequential(
            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        
        # Fourth layer with layer normalization
        self.layer4 = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
        
        # Output layer
        self.output = nn.Linear(64, 1)
        
        # Initialize weights using Xavier initialization
        self.apply(self._init_weights)
        
        self.optimizer = optim.Adam(self.parameters(), lr=alpha, weight_decay=1e-5)  # Added L2 regularization
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
    
    def forward(self, state, action):
        # Handle batch dimension
        input = torch.cat(state, action, 0)
        if len(input.shape) == 1:
            input = input.unsqueeze(0)
            
        x = self.layer1(input)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return self.output(x)