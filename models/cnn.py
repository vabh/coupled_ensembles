import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as weight_init


class CNN(nn.Module):

    def __init__(self, depth=164, num_classes=100, dropout=False):

        super(CNN, self).__init__()

        self.num_classes = num_classes

        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 32, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(32)

        self.fc = nn.Linear(32, num_classes) 


        for m in self.modules():

            if isinstance(m, nn.Conv2d):
                weight_init.kaiming_normal(m.weight)
                weight_init.constant(m.bias, 0.1)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):

        out = F.max_pool2d(F.relu(self.bn1(self.conv1(x))), 2)
        out = F.max_pool2d(F.relu(self.bn2(self.conv2(out))), 2)
        out = F.relu(self.bn3(self.conv3(out)))

        out = torch.squeeze(F.avg_pool2d(out, 8))
        out = self.fc(out)
        return out


def cnn(**kwargs):
    model = CNN(**kwargs)
    return model
