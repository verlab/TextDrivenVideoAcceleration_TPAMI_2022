import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, state_size):
        super(Critic, self).__init__()

        self.state_size = state_size

        self.fc1 = nn.Linear(self.state_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 1)

        self.apply(self.init_weights)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        out = self.fc3(x)
        return out
    
    def criticize(self, state):
        v = self.forward(state)
        return v
    
    def init_weights(self, m):
        if type(m) == nn.Linear:
            torch.nn.init.xavier_uniform_(m.weight)