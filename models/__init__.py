from models.resnet import ResNet18,ResNet34, ResNet50, ResNet101, WideResNet50_2, WideResNet101_2
from models.resnet_cifar import cResNet18, cResNet50, cResNet101, cResNet34
from models.frankle import FC, Conv2, Conv4, Conv6, Conv4Wide, Conv8, Conv6Wide,Conv8Wide,Conv2Wide
from models.vgg_cifar10 import vgg_small
__all__ = [
    "ResNet18",
    "ResNet34",
    "ResNet50",
    "ResNet101",
    "cResNet18",
    "cResNet34"
    "cResNet50",
    "WideResNet50_2",
    "WideResNet101_2",
    "FC",
    "Conv2",
    "Conv4",
    "Conv6",
    "Conv4Wide",
    "Conv8",
    "Conv6Wide",
    'Conv8Wide',
    'Conv2Wide', 
    'vgg_small'
]
