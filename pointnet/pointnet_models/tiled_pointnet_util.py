

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.utils.data
import torch.nn.functional as F
from pointnet.pointnet_models.pointnet_utils import STN3d, STNkd, feature_transform_reguliarzer
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


class TiledSTN3d(nn.Module):
    def __init__(self, channel, args):
        super(TiledSTN3d, self).__init__()
        self.args=args
        self.conv1 = init_conv1d(channel, 64, 1,self.args)
        self.conv2 = init_conv1d(64, 128, 1,self.args)
        self.conv3 = init_conv1d(128, 1024, 1,self.args)
        self.fc1 = init_linear(1024, 512,self.args)
        self.fc2 = init_linear(512, 256,self.args)
        self.fc3 = init_linear(256, 9,self.args)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):

        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x.cuda().float())))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.autograd.Variable(torch.from_numpy(np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]).astype(float))).view(1, 9).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, 3, 3)
        return x


class TiledSTNkd(nn.Module):
    def __init__(self,args,k=64):
        super(TiledSTNkd, self).__init__()
        self.args=args
        self.conv1 = init_conv1d(k, 64, 1,args)
        self.conv2 = init_conv1d(64, 128, 1,args)
        self.conv3 = init_conv1d(128, 1024, 1,args)
        self.fc1 = init_linear(1024, 512,args)
        self.fc2 = init_linear(512, 256,args)
        self.fc3 = init_linear(256, k * k,args)
        self.relu = nn.ReLU()

        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

        self.k = k

    def forward(self, x):
        batchsize = x.size()[0]
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)

        x = F.relu(self.bn4(self.fc1(x)))
        x = F.relu(self.bn5(self.fc2(x)))
        x = self.fc3(x)

        iden = torch.autograd.Variable(torch.from_numpy(np.eye(self.k).flatten().astype(float))).view(1, self.k * self.k).repeat(
            batchsize, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(-1, self.k, self.k)
        return x


class TiledPointNetEncoder(nn.Module):
    def __init__(self, global_feat=True, feature_transform=False, channel=3, args=None):
        super(TiledPointNetEncoder, self).__init__()
        self.args=args
        self.stn = TiledSTN3d(channel,self.args)
        self.conv1 = init_conv1d(channel, 64, 1,args)
        self.conv2 = init_conv1d(64, 128, 1,args)
        self.conv3 = init_conv1d(128, 1024, 1,args)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(128)
        self.bn3 = nn.BatchNorm1d(1024)
        self.global_feat = global_feat
        self.feature_transform = feature_transform
        if self.feature_transform:
            self.fstn = TiledSTNkd(self.args,k=64)

    def forward(self, x):
        B, D, N = x.size()
        trans = self.stn(x)
        x = x.transpose(2, 1)
        if D > 3:
            feature = x[:, :, 3:]
            x = x[:, :, :3]
        x = torch.bmm(x.double(), trans.double()).float()
        if D > 3:
            x = torch.cat([x, feature], dim=2)
        x = x.transpose(2, 1)
        x = F.relu(self.bn1(self.conv1(x)))

        if self.feature_transform:
            trans_feat = self.fstn(x)
            x = x.transpose(2, 1)
            x = torch.bmm(x.double(), trans_feat.double()).float()
            x = x.transpose(2, 1)
        else:
            trans_feat = None

        pointfeat = x
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.bn3(self.conv3(x))
        x = torch.max(x, 2, keepdim=True)[0]
        x = x.view(-1, 1024)
        if self.global_feat:
            return x, trans, trans_feat
        else:
            x = x.view(-1, 1024, 1).repeat(1, 1, N)
            return torch.cat([x, pointfeat], 1), trans, trans_feat


def feature_transform_reguliarzer(trans):
    d = trans.size()[1]
    I = torch.eye(d)[None, :, :]
    if trans.is_cuda:
        I = I.cuda()
    loss = torch.mean(torch.norm(torch.bmm(trans, trans.transpose(2, 1)) - I, dim=(1, 2)))
    return loss