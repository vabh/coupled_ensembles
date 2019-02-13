import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init


class Bottleneck(nn.Module):

    def __init__(self, nInputPlane, nOutputPlane, stride):

        super(Bottleneck, self).__init__()
        self.nInputPlane = nInputPlane
        self.nOutputPlane = nOutputPlane
        self.nBottleneckPlane = nOutputPlane // 4

        self.bn1 = nn.BatchNorm2d(nInputPlane)
        self.conv1 = nn.Conv2d(nInputPlane, self.nBottleneckPlane,
                               kernel_size=1, stride=stride, padding=0)

        self.bn2 = nn.BatchNorm2d(self.nBottleneckPlane)
        self.conv2 = nn.Conv2d(self.nBottleneckPlane, self.nBottleneckPlane,
                               kernel_size=3, stride=1, padding=1)

        self.bn3 = nn.BatchNorm2d(self.nBottleneckPlane)
        self.conv3 = nn.Conv2d(self.nBottleneckPlane, self.nOutputPlane,
                               kernel_size=1, stride=1, padding=0)

        if self.nInputPlane != self.nOutputPlane:
            self.shortcut = nn.Conv2d(self.nInputPlane, self.nOutputPlane,
                                      kernel_size=1, stride=stride, padding=0)

    def forward(self, x):

        if self.nInputPlane == self.nOutputPlane:
            out = self.conv1(F.relu(self.bn1(x)))
            out = self.conv2(F.relu(self.bn2(out)))
            out = self.conv3(F.relu(self.bn3(out)))
            out = out + x
            return out
        else:
            pre_act = F.relu(self.bn1(x))

            out = self.conv1(pre_act)
            out = self.conv2(F.relu(self.bn2(out)))
            out = self.conv3(F.relu(self.bn3(out)))

            out = out + self.shortcut(pre_act)
            return out


class ResNet(nn.Module):

    def __init__(self, depth=164, num_classes=100, dropout=False):

        super(ResNet, self).__init__()

        self.num_classes = num_classes

        if ((depth - 2) % 9) != 0:
            print('Please give a depth value such that depth-2 is divisible by 9, given depth value is %d' % depth)
            raise ValueError
        self.n = (depth - 2) // 9
        print('Depth: ', self.n)

        self.n_stages = [16, 64, 128, 256]

        self.conv1 = nn.Conv2d(3, self.n_stages[0], kernel_size=3, padding=1)
        self.stage1 = self._make_layer(Bottleneck, self.n_stages[0], self.n_stages[1], self.n, 1)
        self.stage2 = self._make_layer(Bottleneck, self.n_stages[1], self.n_stages[2], self.n, 2)
        self.stage3 = self._make_layer(Bottleneck, self.n_stages[2], self.n_stages[3], self.n, 2)

        self.bn1 = nn.BatchNorm2d(self.n_stages[3])
        self.fc = nn.Linear(self.n_stages[3], self.num_classes)

        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # weight_init.constant_(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        out = self.conv1(x)

        out = self.stage1(out)
        out = self.stage2(out)
        out = self.stage3(out)

        out = F.relu(self.bn1(out))
        out = torch.squeeze(F.avg_pool2d(out, 8))
        out = self.fc(out)
        return out

    def _make_layer(self, block, nInputPlane, nOutputPlane, count, stride):

        layers = []
        layers.append(block(nInputPlane, nOutputPlane, stride))
        for i in range(1, count):
            layers.append(block(nOutputPlane, nOutputPlane, 1))

        return nn.Sequential(*layers)


def resnet(**kwargs):
    model = ResNet(**kwargs)
    return model
