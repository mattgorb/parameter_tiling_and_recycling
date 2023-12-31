# https://openreview.net/forum?id=TVHS5Y4dNvM

import torch.nn as nn

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import sys
sys.path.insert(0, '/s/chopin/l/grad/mgorb/parameter_tiling_and_recycling/')
from utils.layer_type import SubnetConv1dTiledFull,SubnetConvTiledFull,SubnetLinearTiledFull

def init_conv2d(in_channel, out_channel,kernel_size,stride=1, groups=1, padding=0,  args=None):
    layer=SubnetConvTiledFull(in_channel, out_channel,kernel_size,stride=stride,groups=groups, padding=padding, )
    if layer.weight.numel()<64000:
        layer.init(args, compression_factor=1)
    else:
        layer.init(args,compression_factor=args.compression_factor)
    return layer

def init_conv1d(in_channel, out_channel,kernel_size,args):
    layer=SubnetConv1dTiledFull(in_channel, out_channel,kernel_size)
    if layer.weight.numel()<64000:
        layer.init(args, compression_factor=1)
    else:
        layer.init(args,compression_factor=args.compression_factor)
    return layer

def init_linear(in_channel, out_channel,args):
    layer=SubnetLinearTiledFull(in_channel, out_channel)
    if layer.weight.numel()<64000:
        layer.init(args, compression_factor=1)
    else:
        layer.init(args,compression_factor=args.compression_factor)
    return layer

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
        
    def forward(self, x):
        return self.fn(x) + x

def TiledConvMixer(dim, depth, kernel_size=9, patch_size=7, n_classes=1000, args=None):
    return nn.Sequential(
init_conv2d(3, dim, kernel_size=patch_size, stride=patch_size, args=args),
 nn.GELU(),
 nn.BatchNorm2d(dim),
 *[nn.Sequential(
 Residual(nn.Sequential(
 init_conv2d(dim, dim, kernel_size, groups=dim, padding="same",args=args),
 nn.GELU(),
 nn.BatchNorm2d(dim)
 )),
 init_conv2d(dim, dim, kernel_size=1,args=args),
 nn.GELU(),
 nn.BatchNorm2d(dim)
 ) for i in range(depth)],
 nn.AdaptiveAvgPool2d((1,1)),
 nn.Flatten(),
init_linear(dim, n_classes, args=args)
)
