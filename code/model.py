import torch
import torch.nn as nn
import torch.nn.functional as F
from kinematics import check_c

class FKModel(nn.Module):

    def __init__(self, bsize):
        super(FKModel, self).__init__()
        self.input_size = 3
        self.output_size = 6
        self.b_size = bsize

        self.fc1 = nn.Linear(self.input_size, 256)
        self.fc2 = nn.Linear(256, 512)
        self.fc3 = nn.Linear(512, self.output_size)

        self.bn1 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = F.relu(self.fc1(x), inplace=True)
        x = self.bn1(x)
        x = F.relu(self.fc2(x), inplace=True)
        x = torch.sigmoid(self.fc3(x))

        return x

    def loss(self, motor_a, passive_a):
        batch_l = check_c(motor_a, passive_a, self.b_size)
        return batch_l
