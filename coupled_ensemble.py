import torch
import torch.nn as nn
import torch.nn.functional as F

from densenet import DenseNet


class CoupledEnsemble(nn.Module):
    def __init__(self, base_model, num=1, probs=False, ensemble=False, test=False, k=12, L=100, dropout=False, nClasses=100):
        super(CoupledEnsemble, self).__init__()

        self.num = num
        self.probs = probs
        self.test = test
        self.ensemble = ensemble

        self.nets = nn.ModuleList([DenseNet(growthRate=k, depth=L, reduction=0.5,
                                nClasses=nClasses, bottleneck=True, dropout=dropout)for _ in range(num)])

    def forward(self, x):
        if self.test is True or self.ensemble is False:
            out = []
            for n in range(self.num):
                out.append(self.nets[n](x))
            return out
        else:
            if self.probs is True:
                out = nn.LogSoftmax()(self.nets[0](x))
                for n in range(1, self.num):
                    out = out + nn.LogSoftmax()(self.nets[n](x))
            else:
                out = self.nets[0](x)
                for n in range(1, self.num):
                    out = out + self.nets[n](x)

            return out


if __name__ == '__main__':
    test = CoupledEnsemble(None, 2)
