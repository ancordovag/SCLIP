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
    def __init__(self, hidden_size):
        super(SCLIPNN, self).__init__()
        self.input_size = 384
        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(self.input_size,self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size,512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        logits = self.linear2(x)
        return logits

class SCLIP_3NN(nn.Module):
    def __init__(self, hidden_size):
        super(SCLIP_3NN, self).__init__()
        self.input_size = 384
        self.hidden_size = hidden_size
        self.output_size = 512
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.linear2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.linear3 = nn.Linear(self.hidden_size, self.output_size)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.relu(self.linear2(x))
        logits = self.linear3(x)
        return logits
    
class SCLIP_DROP(nn.Module):
    def __init__(self, hidden_size):
        super(SCLIP_DROP, self).__init__()
        self.input_size = 384
        self.hidden_size = hidden_size
        self.output_size = 512
        self.linear1 = nn.Linear(self.input_size, self.hidden_size)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.25)
        self.linear2 = nn.Linear(self.hidden_size, self.output_size)
        
    def forward(self, x):
        x = self.relu(self.linear1(x))
        x = self.drop(x)
        logits = self.linear2(x)
        return logits