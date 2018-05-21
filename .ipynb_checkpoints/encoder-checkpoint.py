# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class EncoderRNN(nn.Module):
    def __init__(self, input_size, hidden_size, pretrained_weight = None):
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(input_size, hidden_size)
        if pretrained_weight:
            self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
            #self.embedding.weight.requires_grad = False

        self.gru = nn.GRU(hidden_size, hidden_size)



    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)

        output = embedded
        output, hidden = self.gru(output, hidden)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

