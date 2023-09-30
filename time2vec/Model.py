from torch import nn

from time2vec.periodic_activations import SineActivation, CosineActivation
import torch.nn.functional as F

class Time2Vec(nn.Module):
    def __init__(self, activation, hidden_dim):
        super(Time2Vec, self).__init__()
        if activation == "sin":
            self.l1 = SineActivation(1, hidden_dim)
        elif activation == "cos":
            self.l1 = CosineActivation(1, hidden_dim)

        self.fc1 = nn.Linear(hidden_dim, 2)

    def forward(self, x):
        # x = x.unsqueeze(1)
        x = self.l1(x)
        x = self.fc1(x)
        return x
