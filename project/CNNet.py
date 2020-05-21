import torch 
from torch import Tensor 
from torch import nn
from torch.nn import functional as F
from torch import optim


#define CNN 
class CNNet(nn.Module):
    def __init__(self, n_output):
        super(CNNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(256, 200)
        self.fc2 = nn.Linear(200, n_output) #depends on the model
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=3, stride=3))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 256)))
        x = self.fc2(x)
        return x

# long to train 
class CNNet2(nn.Module):
    def __init__(self, n_output):
        super(CNNet2, self).__init__()
        self.conv1 = nn.Conv2d(1, 512, kernel_size=3)
        self.bn1 = nn.BatchNorm2d(512)
        self.conv2 = nn.Conv2d(512, 128, kernel_size=3)
        self.bn2 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(3200, 256)
        self.fc2 = nn.Linear(256, n_output)
        

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.bn1(self.conv1(x)), kernel_size=2, stride=2))
        x = F.relu(F.max_pool2d(self.bn2(self.conv2(x)), kernel_size=2, stride=2))
        x = F.relu(self.fc1(x.view(-1, 3200)))
        x = self.fc2(x)
        return x

