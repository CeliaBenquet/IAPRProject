import torch 
from torch import Tensor 
from torch import nn
from torch.nn import functional as F
from torch import optim


#define the net 
class Net(nn.Module):
    def __init__(self, n_input=784, n_hidden=100, n_output=10):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(n_input, n_hidden)
        self.fc2 = nn.Linear(n_hidden,n_hidden)
        self.fc3 = nn.Linear(n_hidden, n_output)
        # dropout prevents overfitting of data
        self.dropout = nn.Dropout(0.2)


    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


