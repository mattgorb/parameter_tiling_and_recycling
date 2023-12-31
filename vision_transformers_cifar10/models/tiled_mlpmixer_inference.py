# https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce



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
from utils.layer_type import *


def init_conv1d(in_channel, out_channel,kernel_size,args):
    layer=SubnetConv1dTiledFullInference(in_channel, out_channel,kernel_size)
    if layer.weight.numel()<64000:
        layer.init(args, compression_factor=1)
    else:
        layer.init(args,compression_factor=args.compression_factor)
    return layer

def init_linear(in_channel, out_channel,args):
    layer=SubnetLinearTiledFullInference(in_channel, out_channel)
    if layer.weight.numel()<64000:
        layer.init(args, compression_factor=1)
    if layer.weight.numel()<64000:
        layer.init(args, compression_factor=1)
    else:
        layer.init(args,compression_factor=args.compression_factor)
    return layer


pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = None, args=None):
    inner_dim = int(dim * expansion_factor)

    if dense=='conv1d':
        dense_in=init_conv1d(dim, inner_dim, 1, args)
        dense_out=init_conv1d(inner_dim,dim,  1, args)
    elif dense=='linear':
        dense_in=init_linear(dim, inner_dim,  args)
        dense_out=init_linear(inner_dim, dim,  args)

    
    return nn.Sequential(
        dense_in,
        nn.GELU(),
        nn.Dropout(dropout),
        dense_out,
        nn.Dropout(dropout)
    )

def TiledMLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0., args=None):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = 'conv1d', 'linear'

    return nn.Sequential(
        Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size),
        init_linear((patch_size ** 2) * channels, dim, args),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first,args)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last,args))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
       init_linear(dim, num_classes,args)
    )
