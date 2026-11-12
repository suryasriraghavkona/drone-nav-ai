"""DroNet-inspired ResNet-8 with two heads (steering regression, collision logit)."""
import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self, cin, cout):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(cout)
        self.conv2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.short = nn.Conv2d(cin, cout, 1, stride=2)

    def forward(self, x):
        y = self.conv1(F.relu(self.bn1(x)))
        y = self.conv2(F.relu(self.bn2(y)))
        return self.short(x) + y


class DroNetResNet8(nn.Module):
    def __init__(self, in_ch=1, img_size=64):
        super().__init__()
        assert img_size % 32 == 0, "img_size must be divisible by 32"
        self.stem = nn.Sequential(nn.Conv2d(in_ch, 32, 5, stride=2, padding=2), nn.MaxPool2d(3, 2, 1))
        self.b1, self.b2, self.b3 = ResBlock(32, 32), ResBlock(32, 64), ResBlock(64, 128)
        self.drop = nn.Dropout(0.3)
        feat = 128 * (img_size // 32) ** 2
        self.steer = nn.Linear(feat, 1)
        self.coll = nn.Linear(feat, 1)

    def forward(self, x):
        x = self.b3(self.b2(self.b1(self.stem(x))))
        x = self.drop(F.relu(x)).flatten(1)
        return torch.tanh(self.steer(x)).squeeze(1), self.coll(x).squeeze(1)  # steering in [-1,1], collision logit
