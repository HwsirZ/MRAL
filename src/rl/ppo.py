import torch
import torch.optim as optim
from torch.distributions import Categorical

class PPO:
    def __init__(self, policy, lr=2.5e-4):
        self.policy = policy
        self.opt = optim.Adam(policy.parameters(), lr=lr)

    def act(self, obs):
        logits = self.policy(obs)
        dist = Categorical(logits=logits)
        action = dist.sample()
        return action.item(), dist.log_prob(action)

    def update(self, loss):
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
