import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, output_size=1, batch_first=False):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.sequence_length = sequence_length
        self.rnn = nn.RNN(self, input_size, hidden_size, num_layers=1, 
                          nonlinearity='tanh', bias=False, batch_first=batch_first, 
                          dropout=0.0, bidirectional=False, device=None, dtype=None)
        self.fc = nn.Linear(self.hidden_size * self.sequence_length, output_size)

    def forward(self, x):

        out = self.rnn(x)
        out = self.fc(out)
        return out