import torch
import torch.nn as nn
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class FVIBLoss(nn.Module):
    def __init__(self,K, device=device):
        super().__init__()
        self.d=K.shape[0]+1
        self.K = K
        self.device = device
        self.mse = nn.MSELoss()

    def forward(self,outputs, labels):
        labels = F.one_hot(labels, num_classes=self.d)[:,:-1].to(dtype=torch.float)-(torch.ones(labels.shape[0], self.d-1, device=self.device)/self.d)
        h_hat = labels@(self.K.T)
        loss = self.mse(outputs, h_hat)
        return loss


class KLLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, mean, var):
        KL = (-0.5/mean.shape[0]) * torch.sum(1+torch.log(var) - mean**2 - var)
        return KL


class DistortionTaylorApproxLoss(nn.Module):
#Expectation of CE Taylor approx
    def __init__(self, d, device=device):
        super().__init__()
        self.device = device
        self.d = d
        self.ce = nn.CrossEntropyLoss(reduction="none")
        self.Gamma = (1/d**2)*(torch.diag(torch.ones(d)*d)-torch.ones(d,d)).to(device=device)
    def forward(self, W, b, mean, var, labels):
        outputs = torch.zeros(labels.size(0), W.shape[1], device = self.device)@(W.T) +b.unsqueeze(0)
        logp_theta = -self.ce(outputs, labels)
        grad_f = (F.one_hot(labels, num_classes=self.d)-1/self.d)@W #W^t(y-1/d) (batch, kappa)
        hessian_f = (-W.T@self.Gamma@W)
        loss = -logp_theta -(grad_f*mean).sum(1)-0.5*(hessian_f.diagonal().unsqueeze(0)*var).sum(1)-0.5*(mean*(mean@hessian_f.T)).sum(1)
        loss = loss.mean()
        return loss
