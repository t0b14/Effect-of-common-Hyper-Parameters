import torch.nn as nn
import torch
import cython
from time import time
import numpy as np
import torch.multiprocessing as mp

class RNNlayer(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, bias=False, dt=1, tau = 10, noise_var = 0.1):
        super().__init__()
        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, hidden_size
        self.W_in = torch.zeros(self.input_size, self.hidden_size)
        self.W_hidden = torch.zeros(self.hidden_size, self.out_size)
        self.W_in = self.init_W_in(self.W_in)
        self.W_hidden = self.init_W_hidden(self.W_hidden)

        self.noise_var = noise_var
        self.tau = tau
        self.dt = dt

        self.bias = bias
        self.bias_hidden = None
        if(self.bias):
            self.bias_hidden = nn.Parameter(torch.zeros(self.hidden_size))
    
    def forward(self, x, h_0 = None):
        
        batch_size = x.shape[0]
        timesteps = x.shape[1] 

        out = torch.zeros((batch_size, timesteps, self.out_size))

        if(h_0 == None):
            h_0 = torch.zeros((self.out_size, batch_size), dtype=torch.float)
        
        sigma_all = torch.normal(torch.zeros((self.out_size, timesteps, batch_size)), torch.ones((self.out_size, timesteps, batch_size))*self.noise_var)# (16,time,100)
        
        w_input = torch.matmul(self.W_in, x.permute((0,2,1))).permute((1,2,0))

        c_1 = np.float32(1.0 - (self.dt / self.tau))
        c_2 = np.float32(self.dt / self.tau) 

        if self.bias:
            out, h_0 = self.compute_foward_with(out,h_0,sigma_all,w_input,c_1,c_2,timesteps, batch_size)
        else:
            out, h_0 = self.compute_foward_no(out,h_0,sigma_all,w_input,c_1,c_2,timesteps)
        return out, h_0

    def compute_foward_with(self,out,h_0,sigma_all,w_input,c_1,c_2,timesteps, batch_size):
        for t in range(timesteps):

            w_h = torch.mm(self.W_hidden, torch.tanh(h_0)) + self.bias_hidden.unsqueeze(1).expand(-1, batch_size)
            h_0 =  c_1 * h_0 + c_2 * (w_input[:,t,:] + w_h  + sigma_all[:,t,:])
            out[:,t,:] = h_0.T

        return out, h_0

    def compute_foward_no(self,out,h_0,sigma_all,w_input,c_1,c_2,timesteps): 
        for t in range(timesteps):
            w_h = torch.mm(self.W_hidden, torch.tanh(h_0)) 
            h_0 =  c_1 * h_0 + c_2 * (w_input[:,t,:] + w_h  + sigma_all[:,t,:])
            out[:,t,:] = h_0.T

        return out, h_0

    
    def init_W_hidden(self, weights):
        rows = weights.shape[0]
        cols = weights.shape[1]
        weights = torch.randn(size=(rows, cols)) / torch.sqrt(torch.tensor([rows]))
        weights = weights / torch.max(torch.abs(torch.linalg.eigvals(weights)))
            
        return nn.Parameter(weights)
    
    def init_W_in(self, weights):
        rows = weights.shape[0]
        cols = weights.shape[1]

        for i in range(rows):
            idxs = torch.ceil( (cols - 1) * torch.rand( size=(cols, 1) ) ).to(torch.int).reshape(-1)
            weights[i, idxs] = torch.randn(cols)
            n = torch.norm(weights[i, :], p=2)
            weights[i, idxs] = weights[i, idxs] / n
        
        return nn.Parameter(weights)

class cRNN(nn.Module):
    def __init__(self, params, input_s=4, output_s=1, hidden_s=100, hidden_noise=0, bias=False):
        super(cRNN, self).__init__()
        self.i_s = input_s
        self.h_s = hidden_s
        self.o_s = output_s

        self.dev = torch.device("cpu")
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")

        self.rnn = RNNlayer(input_size=self.i_s, tau=params["tau"], noise_var=hidden_noise, bias=bias)
        
        self.batchnorm = nn.BatchNorm1d(100)
        self.fc_out = nn.Linear(self.h_s, self.o_s)
        with torch.no_grad():
            self.fc_out.weight.copy_(self.init_W_out(self.fc_out.weight)) 

    def forward(self, x, h_0=None):
        out, h_0 = self.rnn(x, h_0) 
        # batchnorm wants (batch, channels, timestep) instead of (batch,timestep,channels)
        #out = self.batchnorm(out.permute(0,2,1)).permute(0,2,1)
        out = torch.tanh(out)
        out = self.fc_out(out)
        return out, h_0
    
    def get_weight_matrices(self):
        w_in = self.rnn.W_in.view(-1).detach().cpu().numpy()
        w_rr = self.rnn.W_hidden.view(-1).detach().cpu().numpy() 
        w_out = self.fc_out.weight.data.view(-1).detach().cpu().numpy()
 
        return w_in, w_rr, w_out
    
    def get_activity(self, x, h_0=None):
        out, h_0 = self.rnn(x, h_0) 
        return out, h_0
    
    def set_weight_matrices(self, w_in, w_rr, w_out, bias_hidden=None, bias_out=None):

        if not torch.is_tensor(w_rr):
            w_rr = torch.tensor(w_rr, dtype=torch.float32, requires_grad=True)

        with torch.no_grad():
            self.rnn.W_in = nn.Parameter(w_in)
            self.rnn.W_hidden = nn.Parameter(w_rr)
            self.fc_out.weight = nn.Parameter(w_out)
            if self.rnn.bias:
                self.rnn.bias_hidden = nn.Parameter(bias_hidden)
                self.fc_out.bias = nn.Parameter(bias_out)

    def get_all_weight_matrices(self):
        w_in = self.rnn.W_in
        w_rr = self.rnn.W_hidden
        w_out = self.fc_out.weight.data
        
        return w_in, w_rr, w_out
    
    def get_all_weight_matrices_with_bias(self):
        w_in = self.rnn.W_in
        w_rr = self.rnn.W_hidden
        w_out = self.fc_out.weight.data
        bias_hidden = self.rnn.bias_hidden
        bias_out = self.fc_out.bias
 
        return w_in, w_rr, w_out, bias_hidden, bias_out
    
    def init_W_out(self, weights):
        rows = weights.shape[0]
        cols = weights.shape[1]

        for i in range(rows):
            idxs = torch.ceil( (cols - 1) * torch.rand( size=(cols, 1) ) ).to(torch.int).reshape(-1)
            weights[i, idxs] = torch.randn(cols)
            n = torch.norm(weights[i, :], p=2)
            weights[i, idxs] = weights[i, idxs] / n
        
        return nn.Parameter(weights)
        
    def get_hidden_noise(self):
        return self.rnn.noise_var
    
    def set_hidden_noise(self, val):
        self.rnn.noise_var = val