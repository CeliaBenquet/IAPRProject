import torch 
from torch import Tensor 
from torch import nn
from torch.nn import functional as F
from torch import optim


#define CNN 
class Net(nn.Module):
    def __init__(self, n_output):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(576, 200)
        self.fc2 = nn.Linear(200, n_output) #depends on the model
        self.dropout = nn.Dropout(0.25)
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=3, stride=3))
        x = self.dropout(F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2)))
        x = self.dropout(F.relu(self.fc1(x.view(-1, 576))))
        x = self.fc2(x)
        return x


