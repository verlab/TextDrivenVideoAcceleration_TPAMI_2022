import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.benchmark = True

class Policy(nn.Module):
    def __init__(self, state_size=256, action_size=3):
        super(Policy, self).__init__()

        self.action_size = action_size
        self.state_size = state_size

        self.fc1 = nn.Linear(self.state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, self.action_size)

        self.apply(self.init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return F.softmax(out, dim=2)
    
    def act(self, state):
        probs = self.forward(state)
        
        m = Categorical(probs)
        action = m.sample()
        
        return action, m.log_prob(action), m.entropy(), probs

    def argmax_action(self, state):
        probs = self.forward(state)
        action = torch.argmax(probs).cpu()

        return action.item(), probs

    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)
