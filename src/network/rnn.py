import torch.nn as nn
import torch

class RNNlayer(nn.Module):
    def __init__(self, input_size=4, hidden_size=100, bias=False, dt=1, tau = 10, noise_var = 0.1):
        super().__init__()
        self.input_size, self.hidden_size, self.out_size = input_size, hidden_size, hidden_size
        self.W_in = torch.Tensor(self.input_size, self.hidden_size)
        self.W_hidden = torch.Tensor(self.hidden_size, self.out_size)
        self.W_in = self.init_W_in(self.W_in)
        self.W_hidden = self.init_W_hidden(self.W_hidden)

        self.noise_var = noise_var
        self.tau = tau
        self.dt = dt

        self.bias = bias
        self.bias_hidden = None
        if(self.bias):
            self.bias_hidden = nn.Parameter(torch.Tensor(self.hidden_size))
        
    def forward(self, x, h_0 = None):
        batch_size = x.shape[0]
        timesteps = x.shape[1] 

        out = torch.zeros((batch_size, timesteps, self.out_size))

        if(h_0 == None):
            h_0 = torch.zeros((batch_size, self.out_size)) 
        
        sigma_all = torch.normal(torch.zeros((batch_size, timesteps, self.out_size)), torch.ones((batch_size, timesteps, self.out_size))*self.noise_var)# (16,time,100)

        for t in range(timesteps-1):
            sigma_t = sigma_all[:,t,:] 

            w_input = torch.mm(x[:,t,:], self.W_in) 
            w_h = torch.mm(h_0, self.W_hidden) 
            if self.bias:
                h_0 = (1.0 - (self.dt / self.tau)) * out[:,t,:] + (self.dt / self.tau) * (w_input + w_h + self.bias_hidden + sigma_t)
            else:
                h_0 = (1.0 - (self.dt / self.tau)) * out[:,t,:] + (self.dt / self.tau) * (w_input + w_h + sigma_t)

            out[:,t+1,:] = torch.tanh(h_0)
        
        return out, h_0
    
    def init_W_hidden(self, weights):
        rows = weights.shape[0]
        cols = weights.shape[1]
        weights = torch.randn(size=(rows, cols)) / torch.sqrt(torch.tensor([rows]))
        weights = weights / torch.max(torch.abs(torch.linalg.eigvals(weights)))
            
        return weights
    
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
    def __init__(self, params, input_s=4, output_s=1, hidden_s=100, batch_first=True):
        super(cRNN, self).__init__()
        self.i_s = input_s
        self.h_s = hidden_s
        self.o_s = output_s

        self.dev = torch.device("cpu")
        if torch.cuda.is_available():
            self.dev = torch.device("cuda")

        self.rnn = RNNlayer(tau=params["tau"])
        
        self.batchnorm = nn.BatchNorm1d(100)
        self.fc_out = nn.Linear(self.h_s, self.o_s)
        with torch.no_grad():
            self.fc_out.weight.copy_(self.init_W_out(self.fc_out.weight)) 

    def forward(self, x, h_1=None):

        out, h_1 = self.rnn(x, h_1) 
        # batchnorm wants (batch, channels, timestep) instead of (batch,timestep,channels)
        #out = self.batchnorm(out.permute(0,2,1)).permute(0,2,1)
        out = self.fc_out(out)

        return out, h_1
    
    def get_weight_matrices(self):
        w_in = self.rnn.W_in.view(-1).detach().cpu().numpy()
        w_rr = self.rnn.W_hidden.view(-1).detach().cpu().numpy() 
        w_out = self.fc_out.weight.data.view(-1).detach().cpu().numpy()
        
        return w_in, w_rr, w_out
    
    def init_W_out(self, weights):
        rows = weights.shape[0]
        cols = weights.shape[1]

        for i in range(rows):
            idxs = torch.ceil( (cols - 1) * torch.rand( size=(cols, 1) ) ).to(torch.int).reshape(-1)
            weights[i, idxs] = torch.randn(cols)
            n = torch.norm(weights[i, :], p=2)
            weights[i, idxs] = weights[i, idxs] / n
        
        return nn.Parameter(weights)
        