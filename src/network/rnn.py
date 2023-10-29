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
        
        self.batchnorm = nn.BatchNorm1d(100)
        self.fc_out = nn.Linear(self.h_s, self.o_s)



    def forward(self, x, h_1=None):

        out, h_1 = self.rnn(x, h_1) if h_1 != None else self.rnn(x)

        # batchnorm wants (batch, channels, timestep) instead of (batch,timestep,channels)
        out = self.batchnorm(out.permute(0,2,1)).permute(0,2,1)
        out = self.fc_out(out)

        return out, h_1
    
    def get_weight_matrices(self):

        w_in = self.rnn.weight_ih_l0.view(-1).detach().cpu().numpy()
        w_rr = self.rnn.weight_hh_l0.view(-1).detach().cpu().numpy() 
        w_out = self.fc_out.weight.data.view(-1).detach().cpu().numpy()
        
        return w_in, w_rr, w_out
        