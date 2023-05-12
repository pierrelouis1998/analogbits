import torch.nn as nn


class AnalogBits(nn.Module):
    """Predict x0 with xt and t"""

    def __init__(self, n_features):
        super(AnalogBits, self).__init__()
        self.layer1 = nn.Linear(n_features, 32)
        self.layer2 = nn.Linear(32, n_features)

    def forward(self, x, t):
        x = self.layer1(x)  # Just an example, don't wotk
        x = self.layer2(x)
        return x
