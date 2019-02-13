import torch
import torch.nn as nn
import models


class CoupledEnsemble(nn.Module):
    def __init__(self, arch='densenet', num=1, probs=False, ensemble=False, test=False, **kwargs):
        super(CoupledEnsemble, self).__init__()

        self.num = num
        self.probs = probs
        self.test = test
        self.ensemble = ensemble
        print("Model config: ", kwargs)

        try:
            self.nets = nn.ModuleList([models.__dict__[arch](**kwargs) for _ in range(num)])
        except:
            import sys
            print('Error: ', sys.exc_info()[1])
            raise ValueError

    def forward(self, x):
        if self.test is True or self.ensemble is False:
            out = []
            for n in range(self.num):
                out.append(self.nets[n](x))
            return out
        else:
            if self.probs is True:
                out = nn.LogSoftmax(dim=1)(self.nets[0](x))
                for n in range(1, self.num):
                    out = out + nn.LogSoftmax(dim=1)(self.nets[n](x))
            else:
                out = self.nets[0](x)
                for n in range(1, self.num):
                    out = out + self.nets[n](x)
            out = out / self.num
            return out
