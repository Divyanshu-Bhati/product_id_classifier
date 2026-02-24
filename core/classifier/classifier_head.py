import torch
from torch import nn

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=133):
        """
        input_dim: Number of input features -> recon loss + kl loss + 3 character ratios + z_mean (64 dims) + z_log_var (64 dims)
        """
        super(ClassifierHead, self).__init__()
        
        # Layer 1: Expanding the feature space
        self.fc1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.act1 = nn.LeakyReLU(0.1)
        self.dropout1 = nn.Dropout(0.3)
        
        # Layer 2: Feature refinement
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.act2 = nn.LeakyReLU(0.1) # TODO vs ReLU
        self.dropout2 = nn.Dropout(0.2)
        
        # Layer 3: Output projection
        self.fc3 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Input x expected shape: (Batch, no. of features)
        x = self.fc1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        x = self.bn2(x)
        x = self.act2(x)
        x = self.dropout2(x)
        
        x = self.fc3(x)
        # prob = self.sigmoid(x)
        return x # prob # not returning sigmoid prob -> instead passing to BCE logits loss