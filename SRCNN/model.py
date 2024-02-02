
import torch.nn as nn


class SRCCNN(nn.Module):
    def __init__(self):
        super(SRCCNN, self).__init__()
        self.Conv1 = nn.Conv2d(1, 64, 9, 1, 4)
        self.Conv2 = nn.Conv2d(64, 32, 3, 1, 1)
        self.Conv3 = nn.Conv2d(32, 1, 5, 1, 2)
        self.Relu = nn.ReLU()
        self.sig = nn.Sigmoid()

    def forward(self, x):
        out = self.Relu(self.Conv1(x))
        out = self.Relu(self.Conv2(out))
        out = self.Conv3(out)
        return out
