from __future__ import absolute_import

import torch.nn as nn
import torch.nn.functional as F

__all__ = ['Correlation']


def cross_correlation(z, x):
    # fast cross correlation
    nz = z.size(0)
    nx, c, h, w = x.size()
    x = x.view(-1, nz * c, h, w)
    out = F.conv2d(x, z, groups=nz)
    out = out.view(nx, -1, out.size(-2), out.size(-1))
    return out


class Correlation(nn.Module):
    def __init__(self, output_scale=0.001):
        super(Correlation, self).__init__()
        self.output_scale = output_scale

    def forward(self, z, x):
        return cross_correlation(z, x) * self.output_scale
