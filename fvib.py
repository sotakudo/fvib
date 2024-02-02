import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

def calc_K(d, device=device):
    """
    calculate the matrix K from the number of classes d.
    Note that L = [K 0] as in Appendix B.1
    """
    Gamma_inv = d*(np.ones([d-1, d-1])+ np.diag(np.ones(d-1)))
    D, V = np.linalg.eig(Gamma_inv)   # eigen value and eigen vector is resolved.
    D = D.astype(np.float64)  
    V = V.astype(np.float64) 
    V,_=np.linalg.qr(V)         # Gram-Schmidt orthonormalization
    K = (V@np.diag(np.sqrt(D))).T
    K = torch.tensor(K).to(device=device, dtype=torch.float)
    return K

class FVIB(nn.Module):
    def __init__(self,K, beta, n_samples=12, conf_after_ts=0.9995, device=device):
        """
        K: matrix calculated by calc_K
        n_samples: number of samples at z for calculating output probability
        conf_after_ts: value of c in Confidence Tuning
        """
        super().__init__()
        self.L=torch.concat([K, torch.zeros(K.shape[0], 1,device=device)], dim=1)
        self.n_samples = n_samples
        self.beta = beta
        self.device = device
        self.sm = nn.Softmax(dim=2)
        if conf_after_ts == None:
            self.temperature  =  1
        else:
            d = K.shape[0]+1
            self.temperature  =  (d/ (np.log(d-1)+np.log(conf_after_ts/(1-conf_after_ts)))).item()
    def reparametrizaion(self, mean, sigma):
        epsilon = torch.randn((self.n_samples, mean.shape[0], mean.shape[1]), device=self.device)
        return mean.unsqueeze(0) + epsilon*(sigma.unsqueeze(0))
    def forward(self,h_x):
        b=self.beta
        # if type(self.beta) == torch.nn.parameter.Parameter:
        #     b = torch.clamp(input=self.beta, min=0., max=1.) #can be added for beta-optimization
        var = torch.ones(h_x.shape, device=device)*b
        mean = (1-b)**(1/2)*h_x
        z= self.reparametrizaion(mean,torch.sqrt(var))
        logits = (1-b)**(1/2)*torch.bmm(z, self.L.repeat(self.n_samples, 1, 1))
        logits = logits/self.temperature
        probs = torch.mean(self.sm(logits), dim=0)
        return logits[0], mean, var, probs 
    
class VIB(nn.Module):
    def __init__(self, input_dim, out_dim, sigma_bias = 0.57, test_n_samples=12, device=device):
        super().__init__()
        self.input_dim = input_dim
        self.sigma_bias = sigma_bias
        self.device = device
        self.test_n_samples = test_n_samples
        self.fc = nn.Linear(input_dim//2, out_dim).to(device)
        self.sm = nn.Softmax(dim=2)
    def reparametrizaion(self, mean, sigma, n_samples):
        epsilon = torch.randn((n_samples, mean.shape[0], mean.shape[1]), device=self.device)
        return mean.unsqueeze(0) + epsilon*(sigma.unsqueeze(0))
    def forward(self, x):
        if self.training:
            n_samples = 1
        else:
            n_samples = self.test_n_samples
            
        mean, s = x[:,:self.input_dim//2], x[:,self.input_dim//2:]
        sigma = F.softplus(s+self.sigma_bias)
        var = sigma*sigma+1e-8
        z = self.reparametrizaion(mean,sigma, n_samples)
        outputs = self.fc(z.view(-1, self.fc.in_features)).view(n_samples, -1, self.fc.out_features)
        
        if self.training:
            y_hat = None
        else:
            y_hat = torch.mean(self.sm(outputs), dim=0)
            
        return outputs[0], mean, var, y_hat
    
class Deterministic(nn.Module):
    #baseline model
    def __init__(self, input_dim, out_dim, device=device):
        super().__init__()
        self.fc = nn.Linear(input_dim, out_dim)
        self.device =device
        self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        outputs = self.fc(x) 
        if self.training:
            y_hat = None
        else:
            y_hat = self.sm(outputs)
        return  outputs, x, torch.zeros(x.shape, device=self.device), y_hat

    
