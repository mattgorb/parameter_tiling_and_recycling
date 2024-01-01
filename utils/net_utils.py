from functools import partial
import os
import pathlib
import shutil
import math

import torch
import torch.nn as nn
from utils.layer_type import SubnetConvEdgePopup, SubnetConvBiprop,SubnetConvTiledFull, SubnetConv1dTiledFull, SubnetLinearTiledFull, SubnetConv1dTiledFullInference,SubnetConvTiledFullInference,SubnetLinearTiledFullInference

import torch.nn as nn


def save_checkpoint(state, is_best, filename="checkpoint.pth", save=False):
    filename = pathlib.Path(filename)

    if not filename.parent.exists():
        os.makedirs(filename.parent)

    torch.save(state, filename)

    if is_best:
        shutil.copyfile(filename, str(filename.parent / "model_best.pth"))

        if not save:
            os.remove(filename)


def get_lr(optimizer):
    return optimizer.param_groups[0]["lr"]


def rerandomize_model(model, args):
    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            if isinstance(m, SubnetConvEdgePopup) or isinstance(m, SubnetConvBiprop) or isinstance(m, SubnetConvTiledFull) :
                print(f"==> Rerandomizing weights of {n} with  {args.rerand_type}")

                m.rerandomize()
    if args.rerand_rate is not None: 
        print(f"==> Rerand_rate:  {args.rerand_rate}")
    #if args.data_type=='compression_factor':
        #model = model.to(torch.compression_factor)


def rerandomize_model_parallel(model, args):
    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            if isinstance(m, SubnetConvEdgePopup) or isinstance(m, SubnetConvBiprop) or isinstance(m, SubnetConvTiledFull) or isinstance(m, compression_factor):
                print(f"==> Rerandomizing weights of {n} with  {args.rerand_type}")
                m.rerandomize()
    if args.rerand_rate is not None:
        print(f"==> Rerand_rate:  {args.rerand_rate}")
    #if args.data_type=='compression_factor':
        #model = model.to(torch.compression_factor)
    if args.multigpu:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])

def freeze_model_weights(model):
    print("=> Freezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> No gradient to {n}.weight")
            m.weight.requires_grad = False
            if m.weight.grad is not None:
                print(f"==> Setting gradient of {n}.weight to None")
                m.weight.grad = None

            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> No gradient to {n}.bias")
                m.bias.requires_grad = False

                if m.bias.grad is not None:
                    print(f"==> Setting gradient of {n}.bias to None")
                    m.bias.grad = None


def freeze_model_subnet(model):
    print("=> Freezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            m.scores.requires_grad = False
            print(f"==> No gradient to {n}.scores")
            if m.scores.grad is not None:
                print(f"==> Setting gradient of {n}.scores to None")
                m.scores.grad = None


def unfreeze_model_weights(model):
    print("=> Unfreezing model weights")

    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            print(f"==> Gradient to {n}.weight")
            m.weight.requires_grad = True
            if hasattr(m, "bias") and m.bias is not None:
                print(f"==> Gradient to {n}.bias")
                m.bias.requires_grad = True


def unfreeze_model_subnet(model):
    print("=> Unfreezing model subnet")

    for n, m in model.named_modules():
        if hasattr(m, "scores"):
            print(f"==> Gradient to {n}.scores")
            m.scores.requires_grad = True



def accumulate(model, f):
    acc = 0.0

    for child in model.children():
        acc += accumulate(child, f)

    acc += f(model)

    return acc


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """

    def __init__(self, smoothing=0.0):
        """
        Constructor for the LabelSmoothing module.

        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()


class SubnetL1RegLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, model, temperature=1.0):
        l1_accum = 0.0
        for n, p in model.named_parameters():
            if n.endswith("scores"):
                l1_accum += (p*temperature).sigmoid().sum()

        return l1_accum



def model_stats(model):

    print(f'Total model parameters: { sum(p.numel() for p in model.parameters()) }')
    for n,m in model.named_parameters():
        print(f'{n}, {m.size()}, {m.numel()}')

        
    model_layer_params={
        'linear': 0,
        'bias': 0,
        'batchnorm2d':0,
        'batchnorm1d':0,
        'layernorm':0,
        'conv1x1':0,
        'conv_gt_1x1':0, 

    }
    '''model_layer_conv_kernels={
        'conv1x1':0,
        'conv_gt_1x1':0
    }'''
    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            if isinstance(m, nn.Linear) :
                model_layer_params['linear']+=m.weight.numel()
            elif isinstance(m, nn.BatchNorm2d) :
                model_layer_params['batchnorm2d']+=m.weight.numel()
            elif isinstance(m, nn.BatchNorm1d) :
                model_layer_params['batchnorm1d']+=m.weight.numel()
            elif isinstance(m, nn.LayerNorm) :
                model_layer_params['layernorm']+=m.weight.numel()

            elif  isinstance(m, nn.Conv2d) or isinstance(m, nn.Conv1d):
                #print(m.kernel_size)
                kernel_size=m.kernel_size[0]
                if kernel_size==1:
                    model_layer_params['conv1x1']+=m.weight.numel()
                else:
                    model_layer_params['conv_gt_1x1']+=m.weight.numel()
            else:
                if  not isinstance(m, SubnetConv1dTiledFull) and\
                    not isinstance(m, SubnetLinearTiledFull) and\
                    not isinstance(m, SubnetConvTiledFull):
                    print(f'module instance not found for module with weights: {n}')
                    #print("add instance before continuing!!!!!!!")
                    #sys.exit()

            '''elif isinstance(m, nn.Conv2d) :
                model_layer_params['conv2d']+=m.weight.numel()
            elif isinstance(m, nn.Conv1d) :
                model_layer_params['conv1d']+=m.weight.numel()'''



        if hasattr(m, "bias") and m.bias is not None:
            #print(f'adding {m.bias.numel()} bias params for module {n}')
            model_layer_params['bias']+=m.bias.numel()
    
    print('\ndense model summary:')
    for layer, params in model_layer_params.items():
        print(f'\t{layer}, {params}')
    



    model_layer_params={
        'linear': 0,
        'bias': 0,
        'batchnorm2d':0,
        'batchnorm1d':0,
        'layernorm':0,
        'conv2d':0,
        'conv1d':0
    }
    for n, m in model.named_modules():
        if hasattr(m, "weight") and m.weight is not None:
            if isinstance(m, SubnetLinearTiledFull) or isinstance(m, SubnetLinearTiledFullInference)  :
                model_layer_params['linear']+=m.weight.numel()/m.compression_factor+m.compression_factor*32
                print(f'layer {n} compression rate: {m.compression_factor}')
            elif isinstance(m, nn.BatchNorm2d) :
                model_layer_params['batchnorm2d']+=m.weight.numel()
            elif isinstance(m, nn.BatchNorm1d) :
                model_layer_params['batchnorm1d']+=m.weight.numel()
            elif isinstance(m, nn.LayerNorm) :
                model_layer_params['layernorm']+=m.weight.numel()
            elif isinstance(m, SubnetConvTiledFull) or isinstance(m, SubnetConvTiledFullInference) :
                model_layer_params['conv2d']+=m.weight.numel()/m.compression_factor+m.compression_factor*32
                print(f'layer {n} compression rate: {m.compression_factor}')
            elif isinstance(m, SubnetConv1dTiledFull)  or isinstance(m, SubnetConv1dTiledFullInference) :
                model_layer_params['conv1d']+=m.weight.numel()/m.compression_factor+m.compression_factor*32
                print(f'layer {n} compression rate: {m.compression_factor}')
            else:
                if  not isinstance(m, nn.Conv2d) and\
                    not isinstance(m, nn.Conv1d) and\
                    not isinstance(m, nn.Linear):
                    print(f'module instance not found for module with weights: {n}')
                    #print("add instance before continuing!!!!!!!")
                    #sys.exit()
        if hasattr(m, "bias") and m.bias is not None:
            #print(f'adding {m.bias.numel()} bias params for module {n}')
            model_layer_params['bias']+=m.bias.numel()
    
    print('\ntiled model summary:')
    print("(linear, conv1d, conv2d are compressed)")
    for layer, params in model_layer_params.items():
        print(f'\t{layer}, {params}')