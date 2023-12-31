
import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math
from utils.tile_utils import create_signed_tile

from args import args
from utils.bn_type import LearnedBatchNorm

from utils.layer_type import SubnetConvTiledFull, SubnetLinearTiledFull

def conv_layer_init(cin, cout, layer_compression_factor):
        conv=SubnetConvTiledFull(cin, cout, kernel_size=3, padding=1, bias=False)
        conv.init(args, layer_compression_factor)
        return conv

class VGG_SMALL(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL, self).__init__()


        self.num_layers=7

        self.layer_compression_factors=None



        if args.layer_compression_factors is not None:
            self.layer_compression_factors=list(args.layer_compression_factors.split(','))
            self.layer_compression_factors=[int(x) for x in self.layer_compression_factors]
            assert len(self.layer_compression_factors)==self.num_layers, f"mask compression factor must have length {self.num_layers}"
        if args.global_compression_factor is not None: 
            self.layer_compression_factors=[args.global_compression_factor for i in range(self.num_layers)]
        #self.compression_factors_ind=0


        self.conv0 = conv_layer_init(3, 128,self.layer_compression_factors[0])
        self.bn0 = LearnedBatchNorm(128)
        self.conv1 = conv_layer_init(128, 128, self.layer_compression_factors[1])
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = LearnedBatchNorm(128)
        self.nonlinear = nn.ReLU(inplace=True)
        #self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = conv_layer_init(128, 256,  self.layer_compression_factors[2])
        self.bn2 = LearnedBatchNorm(256)
        self.conv3 = conv_layer_init(256, 256,  self.layer_compression_factors[3])
        self.bn3 = LearnedBatchNorm(256)
        self.conv4 = conv_layer_init(256, 512,  self.layer_compression_factors[4])
        self.bn4 = LearnedBatchNorm(512)
        self.conv5 = conv_layer_init(512, 512,  self.layer_compression_factors[5])
        self.bn5 = LearnedBatchNorm(512)

        self.fc = SubnetLinearTiledFull(512*4*4, num_classes, bias=False)
        self.fc.init(args,  self.layer_compression_factors[6])

        print(self.layer_compression_factors)
        
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, SubnetConvTiledFull):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                #m.bias.data.zero_()
            elif isinstance(m, nn.Linear) or isinstance(m, SubnetLinearTiledFull):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                #m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x




def vgg_small(**kwargs):
    model = VGG_SMALL()
    return model































class VGG_SMALL_DENSE(nn.Module):

    def __init__(self, num_classes=10):
        super(VGG_SMALL_DENSE, self).__init__()
        self.conv0 = nn.Conv2d(3, 128, kernel_size=3, padding=1,)
        self.bn0 = LearnedBatchNorm(128)
        self.conv1 = nn.Conv2d(128, 128, kernel_size=3, padding=1,)
        self.pooling = nn.MaxPool2d(kernel_size=2, stride=2)
        self.bn1 = LearnedBatchNorm(128)
        self.nonlinear = nn.ReLU(inplace=True)
        #self.nonlinear = nn.Hardtanh(inplace=True)
        self.conv2 = nn.Conv2d(128, 256, kernel_size=3, padding=1,)
        self.bn2 = LearnedBatchNorm(256)
        self.conv3 = nn.Conv2d(256, 256, kernel_size=3, padding=1,)
        self.bn3 = LearnedBatchNorm(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=1,)
        self.bn4 = LearnedBatchNorm(512)
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, padding=1,)
        self.bn5 = LearnedBatchNorm(512)

        self.fc = nn.Linear(512*4*4, num_classes)
        

        #print(self.layer_compression_factors)
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.nonlinear(x)
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.bn1(x)
        x = self.nonlinear(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.nonlinear(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.bn3(x)
        x = self.nonlinear(x)
        x = self.conv4(x)
        x = self.bn4(x)
        x = self.nonlinear(x)
        x = self.conv5(x)
        x = self.pooling(x)
        x = self.bn5(x)
        x = self.nonlinear(x)
        # x = self.pooling(x)
        x = x.view(x.size(0), -1)

        x = self.fc(x)
        return x




def vgg_small_dense(**kwargs):
    model = VGG_SMALL_DENSE()
    return model