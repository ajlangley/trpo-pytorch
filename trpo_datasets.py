import numpy as np
from torch.utils.data import Dataset

class PolicyDataset(Dataset):
    def __init__(self, states, actions, advantages):
        self.states = states
        self.actions = actions
        self.advantages = advantages

    def __getitem__(self, i):
        return self.states[i], self.actions[i], self.advantages[i]

    def __len__(self):
        return len(self.states)


class ValueFunDataset(Dataset):
    def __init__(self, states, q_vals):
        self.states = states
        self.q_vals = q_vals

    def __getitem__(self, i):
        return self.states[i], self.q_vals[i]

    def __len__(self):
        return len(self.q_vals)
