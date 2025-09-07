import torch.nn as nn
import torch

class LogisticNet(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.fc = nn.Linear(in_feats, 1)

    def forward(self, x): return self.fc(x)

class LinearSVMNet(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.fc = nn.Linear(in_feats, 1)

    def forward(self, x): return self.fc(x)

class MLPNet(nn.Module):
    """
    Two-layer MLP  (in  → hidden → 1)
    * hidden  : int    – number of units in the hidden layer
    * dropout : float  – dropout probability after hidden layer
    """
    def __init__(self, in_feats: int, *, hidden: int = 64, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_feats, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1)          # logits
        )

    def forward(self, x):
        return self.net(x)  
