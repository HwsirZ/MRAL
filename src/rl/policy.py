import torch
import torch.nn as nn

class Policy(nn.Module):
    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256),
            nn.ReLU(),
            nn.Linear(256, act_dim)
        )

    def forward(self, x):
        return self.net(x)
