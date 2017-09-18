import torch
import torch.nn as nn
import torch.nn.functional as F

import math


class Bottleneck(nn.Module):
    def __init__(self, nChannels, growthRate, dropout=False):
        super(Bottleneck, self).__init__()
        interChannels = 4*growthRate
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, interChannels, kernel_size=1,
                               bias=False)
        self.bn2 = nn.BatchNorm2d(interChannels)
        self.conv2 = nn.Conv2d(interChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)
        self.dropout = dropout

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout:
            out = F.dropout(out, p=0.2)
        out = self.conv2(F.relu(self.bn2(out)))
        if self.dropout:
            out = F.dropout(out, p=0.2)
        out = torch.cat((x, out), 1)
        return out


class SingleLayer(nn.Module):
    def __init__(self, nChannels, growthRate):
        super(SingleLayer, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, growthRate, kernel_size=3,
                               padding=1, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        out = torch.cat((x, out), 1)
        return out


class Transition(nn.Module):
    def __init__(self, nChannels, nOutChannels, dropout=False):
        super(Transition, self).__init__()
        self.bn1 = nn.BatchNorm2d(nChannels)
        self.conv1 = nn.Conv2d(nChannels, nOutChannels, kernel_size=1,
                               bias=False)
        self.dropout = dropout

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout:
            out = F.dropout(out, p=0.2)
        out = F.avg_pool2d(out, 2)
        return out


class DenseNet(nn.Module):
    def __init__(self, growthRate, depth, reduction, nClasses, bottleneck,
                            joint=False, probs=False, dropout=False):
        super(DenseNet, self).__init__()

        nDenseBlocks = (depth-4) // 3
        if bottleneck:
            nDenseBlocks //= 2

        nChannels = 2*growthRate
        self.conv1 = nn.Conv2d(3, nChannels, kernel_size=3, padding=1,
                               bias=False)
        self.dense1 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, dropout)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans1 = Transition(nChannels, nOutChannels, dropout)

        nChannels = nOutChannels
        self.dense2 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, dropout)
        nChannels += nDenseBlocks*growthRate
        nOutChannels = int(math.floor(nChannels*reduction))
        self.trans2 = Transition(nChannels, nOutChannels, dropout)

        nChannels = nOutChannels
        self.dense3 = self._make_dense(nChannels, growthRate, nDenseBlocks, bottleneck, dropout)
        nChannels += nDenseBlocks*growthRate

        self.bn1 = nn.BatchNorm2d(nChannels)
        self.probs = probs
        self.dropout = dropout

        self.fc = nn.Linear(nChannels, nClasses)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                # m.momentum = 0.0001 # for additional exp
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_dense(self, nChannels, growthRate, nDenseBlocks, bottleneck, dropout):
        layers = []
        for i in range(int(nDenseBlocks)):
            if bottleneck:
                layers.append(Bottleneck(nChannels, growthRate, dropout))
            else:
                layers.append(SingleLayer(nChannels, growthRate))
            nChannels += growthRate
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.trans1(self.dense1(out))
        out = self.trans2(self.dense2(out))
        out = self.dense3(out)
        out = torch.squeeze(F.avg_pool2d(F.relu(self.bn1(out)), 8))
        out = self.fc(out)
        if self.probs:
            out = nn.LogSoftmax()(out)
        return out


class DenseNetSplit(nn.Module):
    def __init__(self, num=1, nClasses=0, k=12, L=100, probs=False,
                 ensemble=False, test=False, dropout=False):
        super(DenseNetSplit, self).__init__()

        self.num = num
        self.test = test
        self.ensemble = ensemble
        self.nets = nn.ModuleList([DenseNet(growthRate=k, depth=L, reduction=0.5,
                                    bottleneck=True, nClasses=nClasses, joint=False, probs=probs, dropout=dropout)for _ in range(num)])

    def forward(self, x):
        if self.test is True or self.ensemble is False:
            out = []
            for n in range(self.num):
                out.append(self.nets[n](x))
            return out
        else:
            out = self.nets[0](x)
            for n in range(1, self.num):
                out = out + self.nets[n](x)

            return out