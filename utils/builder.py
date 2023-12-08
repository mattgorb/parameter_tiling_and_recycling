from args import args
import math

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm as wn
import utils.layer_type
import utils.bn_type
from utils.initializations import _init_weight

class Builder(object):
    def __init__(self, conv_layer, bn_layer, first_layer=None, weight_tile=None, compression_factors=None):
        self.conv_layer = conv_layer
        self.bn_layer = bn_layer
        self.first_layer = first_layer or conv_layer
        self.weight_tile=weight_tile
        self.compression_factors=compression_factors
        self.compression_factors_ind=0

    def conv(self, kernel_size, in_planes, out_planes, stride=1, first_layer=False, compress_factor=None):
        conv_layer = self.first_layer if first_layer else self.conv_layer

        if first_layer:
            print(f"==> Building first layer with {str(self.first_layer)}")

        if kernel_size == 3:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=3,
                stride=stride,
                padding=1,
                bias=False,
            )
        elif kernel_size == 1:
            conv = conv_layer(
                in_planes, out_planes, kernel_size=1, stride=stride, bias=False
            )
        elif kernel_size == 5:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=5,
                stride=stride,
                padding=2,
                bias=False,
            )
        elif kernel_size == 7:
            conv = conv_layer(
                in_planes,
                out_planes,
                kernel_size=7,
                stride=stride,
                padding=3,
                bias=False,
            )
        else:
            return None

        if args.layer_type=='SubnetConvTiledFull':
            conv.init(args, 
                      self.compression_factors[self.compression_factors_ind])
            self.compression_factors_ind+=1            
        elif args.layer_type!='DenseConv':
            conv.init(args, )
        else:
            conv.args = args
            conv.weight = _init_weight(conv.args, conv.weight)

        #print(conv.weight)

        return conv

    def conv3x3(self, in_planes, out_planes, stride=1, first_layer=False, compress_factor=None):
        """3x3 convolution with padding"""
        c = self.conv(3, in_planes, out_planes, stride=stride, first_layer=first_layer, 
                      compress_factor=compress_factor)
        return c

    def conv1x1(self, in_planes, out_planes, stride=1, first_layer=False, compress_factor=None):
        """1x1 convolution with padding"""
        c = self.conv(1, in_planes, out_planes, stride=stride, first_layer=first_layer, 
                      compress_factor=compress_factor)
        return c

    def conv7x7(self, in_planes, out_planes, stride=1, first_layer=False, compress_factor=None):
        """7x7 convolution with padding"""
        c = self.conv(7, in_planes, out_planes, stride=stride, first_layer=first_layer, 
                      compress_factor=compress_factor)
        return c

    def conv5x5(self, in_planes, out_planes, stride=1, first_layer=False, compress_factor=None):
        """5x5 convolution with padding"""
        c = self.conv(5, in_planes, out_planes, stride=stride, first_layer=first_layer, 
                      compress_factor=compress_factor)
        return c

    def batchnorm(self, planes, last_bn=False, first_layer=False):
        return self.bn_layer(planes)

    def activation(self):
        if args.nonlinearity == "relu":
            return (lambda: nn.ReLU(inplace=True))()
        else:
            raise ValueError(f"{args.nonlinearity} is not an initialization option!")

def get_builder(weight_tile=None, compression_factors=None):

    print("==> Conv Type: {}".format(args.layer_type))
    print("==> BN Type: {}".format(args.bn_type))

    conv_layer = getattr(utils.layer_type, args.layer_type)

    bn_layer = getattr(utils.bn_type, args.bn_type)

    if args.first_layer_type is not None:
        first_layer = getattr(utils.layer_type, args.first_layer_type)
        print(f"==> First Layer Type: {args.first_layer_type}")
    else:
        first_layer = None

    builder = Builder(conv_layer=conv_layer, bn_layer=bn_layer, first_layer=first_layer, 
                      weight_tile=weight_tile, compression_factors=compression_factors)

    return builder
