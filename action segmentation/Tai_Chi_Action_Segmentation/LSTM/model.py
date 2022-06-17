import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np

"""
Neural Networks model : Bidirection LSTM
"""


class BiLSTM(nn.Module):
    
    def __init__(self, args):
        super(BiLSTM, self).__init__()
        self.args = args
        self.hidden_dim = args.lstm_hidden_dim
        self.num_layers = args.lstm_num_layers
        D = args.embed_dim
        C = args.num_classes
        self.bilstm = nn.LSTM(D, self.hidden_dim // 2, num_layers=1, dropout=args.dropout, bidirectional=True, bias=False)
        print(self.bilstm)

        self.hidden2label1 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.hidden2label2 = nn.Linear(self.hidden_dim // 2, C)

    def forward(self, x):
        bilstm_out, _ = self.bilstm(x)

        bilstm_out = torch.transpose(bilstm_out, 0, 1)
        bilstm_out = torch.transpose(bilstm_out, 1, 2)
        bilstm_out = torch.tanh(bilstm_out)
        bilstm_out = torch.max_pool1d(bilstm_out, bilstm_out.size(2)).squeeze(2)
        y = self.hidden2label1(bilstm_out)
        y = self.hidden2label2(y)
        logit = y
        return logit