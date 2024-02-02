import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

class CifarFeatureExtractor(nn.Module):
    def __init__(self,first_ch, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(first_ch,96,kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96,kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 96,kernel_size=3, stride=2,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(96,192,kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3, stride=2,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192,kernel_size=3, stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192,192, kernel_size=1, stride=1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, out_dim, kernel_size=1, stride=1, padding=0),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
    def forward(self,x):
        x = self.features(x)
        return x
    
class MnistFeatureExtractor(nn.Module):
    def __init__(self,hidden_dim=1024, out_dim=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Flatten(),
            nn.Linear(784,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim,out_dim)
        )
    def forward(self,x):
        x = self.features(x)
        return x
    
class LtafFeatureExtractor(nn.Module):
    def __init__(self, drop=0, last_layer= 50, hidden_layer=200, num_layers=1):
        super().__init__()
        self.hidden_layer = hidden_layer
        self.rnn = nn.LSTM(1, self.hidden_layer, num_layers=num_layers, bias=True, batch_first=True, bidirectional=True)
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.do = nn.Dropout(p=drop)
        self.fc1 = nn.Linear(self.hidden_layer*2, 50)
        self.fc2 = nn.Linear(50, last_layer)
    def forward(self, x): 
        out, hiden = self.rnn(x)
        out =self.pool(out.transpose(1,2)).view(-1,self.hidden_layer*2)
        out = self.do(F.relu(self.fc1(out)))
        out = self.fc2(out)
        return out   