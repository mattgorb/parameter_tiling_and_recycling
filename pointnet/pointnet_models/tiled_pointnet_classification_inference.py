import torch.nn as nn
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

#from pointnet.pointnet_models.tiled_pointnet_util import * 

sys.path.insert(0, '/s/chopin/l/grad/mgorb/parameter_tiling_and_recycling/')

from utils.layer_type import *


def init_conv1d(in_channel, out_channel,kernel_size,args):
    layer=SubnetConv1dTiledFullInference(in_channel, out_channel,kernel_size)
    if layer.weight.numel()<int(args.min_compress_size):
        layer.init(args, compression_factor=1)
    else:
        layer.init(args,compression_factor=args.compression_factor)
    return layer

def init_linear(in_channel, out_channel,args):
    layer=SubnetLinearTiledFullInference(in_channel, out_channel)
    if layer.weight.numel()<int(args.min_compress_size):
        layer.init(args, compression_factor=1)
    else:
        layer.init(args,compression_factor=args.compression_factor)
    return layer


class get_tiled_model(nn.Module):
    def __init__(self, k=40, normal_channel=True,args=None):
        super(get_tiled_model, self).__init__()
        self.args=args

        if normal_channel:
            channel = 6
        else:
            channel = 3
        self.feat = TiledPointNetEncoder(global_feat=True, feature_transform=True, channel=channel, args=self.args)
        self.fc1 = init_linear(1024, 512,args)
        self.fc2 = init_linear(512, 256,args)
        self.fc3 = init_linear(256, k,args)
        self.dropout = nn.Dropout(p=0.4)
        self.bn1 = nn.BatchNorm1d(512)
        self.bn2 = nn.BatchNorm1d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x, trans, trans_feat = self.feat(x.float())
        x = F.relu(self.bn1(self.fc1(x)))
        x = F.relu(self.bn2(self.dropout(self.fc2(x))))
        x = self.fc3(x)
        x = F.log_softmax(x, dim=1)
        return x, trans_feat

class get_loss(torch.nn.Module):
    def __init__(self, mat_diff_loss_scale=0.001):
        super(get_loss, self).__init__()
        self.mat_diff_loss_scale = mat_diff_loss_scale

    def forward(self, pred, target, trans_feat):
        loss = F.nll_loss(pred, target)
        mat_diff_loss = feature_transform_reguliarzer(trans_feat)

        total_loss = loss + mat_diff_loss * self.mat_diff_loss_scale
        return total_loss