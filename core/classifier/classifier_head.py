import torch
from torch import nn

class ClassifierHead(nn.Module):
    def __init__(self, input_dim=5):
        """
        input_dim: Dimension of the input score vector.
                   Expected input is 5-dimensional: [recon_error, kl_loss, char_ratio, num_ratio, spcl_char_ratio]
        """
        super(ClassifierHead, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.bn1 = nn.BatchNorm1d(64)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(64, 32)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.bn1(x)
        x = torch.relu(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = torch.relu(x)
        x = self.dropout2(x)
        x = self.fc3(x)
        prob = self.sigmoid(x)
        return prob