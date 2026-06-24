import torch
import torch.nn as nn
import torch.nn.functional as F


# a residual block - just two linear layers with a skip connection
class ResBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)

    def forward(self, x):
        return F.relu(x + self.fc2(F.relu(self.fc1(x))))


# the q network. takes a cube state (144 one-hot) and gives a Q value for each
# of the 18 moves. we kept the old name ValueNetwork so the imports dont break.
class ValueNetwork(nn.Module):
    def __init__(self, hidden=768, n_res=4):
        super().__init__()
        self.embed = nn.Sequential(
            nn.Linear(144, hidden),
            nn.ReLU(),
        )
        self.res  = nn.Sequential(*[ResBlock(hidden) for _ in range(n_res)])
        self.head = nn.Linear(hidden, 18)   # 18 outputs, one Q value per move

    def forward(self, x):
        return self.head(self.res(self.embed(x)))


QNetwork = ValueNetwork
