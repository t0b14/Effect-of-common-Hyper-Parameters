import torch.nn as nn
import torch

class cRNN(nn.Module):
    def __init__(self, input_s=4, output_s=1, hidden_s=100, batch_first=True):
        super(cRNN, self).__init__()
        self.i_s = input_s
        self.h_s = hidden_s
        self.o_s = output_s

        self.dev = torch.device("cpu")
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")

        self.rnn = nn.RNN(input_size=self.i_s, hidden_size=self.h_s, num_layers=1, 
                          nonlinearity='tanh', bias=False, batch_first=batch_first, 
                          dropout=0., bidirectional=False, device=self.dev)
        self.fc = nn.Linear(self.h_s, self.o_s)

    def forward(self, x):

        out, _ = self.rnn(x)
        out = self.fc(out)
        return out