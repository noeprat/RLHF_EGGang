import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.init as init
from torch.distributions import Categorical
torch.manual_seed(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Policy(nn.Module): # define the policy network
    def __init__(self, state_size=4, action_size=2, hidden_size=32):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = self.fc2(x)
        return F.softmax(x, dim=1) # we just consider 1 dimensional probability of action

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        probs = self.forward(state).cpu()
        model = Categorical(probs)
        action = model.sample()
        return action.item(), model.log_prob(action)
    

class Reward(nn.Module): # define the policy network
    def __init__(self, dim_state=4, dim_action=1, hidden_size=64):
        super(Reward, self).__init__()
        input_dim = dim_state + dim_action
        self.fc1 = nn.Linear(input_dim, hidden_size)
        self.ln1 = nn.LayerNorm(hidden_size)  # LayerNorm after ReLU
        self.fc2 = nn.Linear(hidden_size, 1)
        self._init_weights()

    def _init_weights(self):
        # He initialization for ReLU layers
        init.kaiming_normal_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        init.kaiming_normal_(self.fc2.weight, mode='fan_in', nonlinearity='relu')
        
        # Zero biases (optional: small bias for ReLU to avoid dead neurons)
        init.zeros_(self.fc1.bias)
        init.zeros_(self.fc2.bias)

    def forward(self, state, action):
        #print(state.shape)
        #print(action.unsqueeze(0).shape)
        input=torch.cat((state,action.unsqueeze(0)),dim=-1)
        #print(input.shape)
        x = F.relu(self.fc1(input))
        x = self.ln1(x) 
        x = self.fc2(x)
        return F.relu(x) 

    