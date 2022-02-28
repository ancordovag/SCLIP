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
    def __init__(self):
        super(SCLIPNN, self).__init__()
        self.linear1 = nn.Linear(384,100)
        self.linear2 = nn.Linear(100,512)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.linear1(x))
        logits = self.linear2(x)
        return logits

class SCLIP_LSTM(nn.Module):
    def __init__(self):
        super(SCLIP_LSTM, self).__init__()
        self.hidden_size = 100
        self.output_size = 512
        self.lstm = nn.LSTM(384, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(1, 1, self.hidden_size, device=device)
        c0 = torch.zeros(1, 1, self.hidden_size, device=device)
        
        embedded = x.view(-1, 1, 384)
        output, _ = self.lstm(embedded, (h0, c0))
        output = output.view(1, self.hidden_size)
        logits = self.linear(output).view(self.output_size)
        return logits
    

class SCLIP_GRU(nn.Module):
    def __init__(self):
        super(SCLIP_GRU, self).__init__()
        self.hidden_size = 100
        self.output_size = 512
        self.gru = nn.GRU(384, self.hidden_size)
        self.linear = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, x):
        h0 = torch.zeros(1, 1, self.hidden_size, device=device)
        
        embedded = x.view(-1, 1, 384)
        output, _ = self.gru(embedded, h0)
        output = output.view(1, self.hidden_size)
        logits = self.linear(output).view(self.output_size)
        return logits

class SCLIP_Attn(nn.Module):
    def __init__(self, max_length=200):
        super(SCLIP_Attn, self).__init__()
        self.hidden_size = 100
        self.output_size = 512
        self.lstm = nn.LSTM(384, self.hidden_size)
        
        # Two linear layers for Attention are created.
        self.max_length = max_length
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 2, self.hidden_size)

        # Initialize LSTM 
        self.recurrent = nn.LSTM(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input):
        hidden = (torch.zeros(1, 1, self.hidden_size, device=device),
                  torch.zeros(1, 1, self.hidden_size, device=device))
        
        encoder_outputs, _ = self.lstm(input.view(1, 1, 384), hidden)
        # It takes the embedded and the hidden to calculate the attention layers        
        attn_weights = F.softmax(
            self.attn(torch.cat((input, hidden[0][0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(0),
                                 encoder_outputs.unsqueeze(0))

        # Put the embedding and the attention layers together
        output = torch.cat((input, attn_applied[0]), 1)
        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.recurrent(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        
        return output
