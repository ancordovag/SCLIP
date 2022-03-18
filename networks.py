"""
@Author: Andrés Alejandro Córdova Galleguillos
"""

# Import pytorch libraries
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use CUDA if it is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SCLIPNN(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(SCLIPNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.input_size,self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size,512)
        self.leak = nn.LeakyReLU()

    def forward(self, x):
        x = self.leak(self.linear1(x))
        logits = self.linear2(x)
        return logits