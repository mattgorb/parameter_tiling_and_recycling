# -*- coding: utf-8 -*-

'''Some helper functions for PyTorch, including:
    - get_mean_and_std: calculate the mean and std value of dataset.
    - msr_init: net parameter initialization.
    - progress_bar: progress bar mimic xlua.progress.
'''
import os
import sys
import time
import math

import torch.nn as nn
import torch.nn.init as init
import sys



def get_mean_and_std(dataset):
    '''Compute the mean and std value of dataset.'''
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=2)
    mean = torch.zeros(3)
    std = torch.zeros(3)
    print('==> Computing mean and std..')
    for inputs, targets in dataloader:
        for i in range(3):
            mean[i] += inputs[:,i,:,:].mean()
            std[i] += inputs[:,i,:,:].std()
    mean.div_(len(dataset))
    std.div_(len(dataset))
    return mean, std

def init_params(net):
    '''Init layer parameters.'''
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            init.kaiming_normal(m.weight, mode='fan_out')
            if m.bias:
                init.constant(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            init.constant(m.weight, 1)
            init.constant(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal(m.weight, std=1e-3)
            if m.bias:
                init.constant(m.bias, 0)


try:
	_, term_width = os.popen('stty size', 'r').read().split()
except:
	term_width = 80
term_width = int(term_width)

TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()

def format_time(seconds):
    days = int(seconds / 3600/24)
    seconds = seconds - days*3600*24
    hours = int(seconds / 3600)
    seconds = seconds - hours*3600
    minutes = int(seconds / 60)
    seconds = seconds - minutes*60
    secondsf = int(seconds)
    seconds = seconds - secondsf
    millis = int(seconds*1000)

    f = ''
    i = 1
    if days > 0:
        f += str(days) + 'D'
        i += 1
    if hours > 0 and i <= 2:
        f += str(hours) + 'h'
        i += 1
    if minutes > 0 and i <= 2:
        f += str(minutes) + 'm'
        i += 1
    if secondsf > 0 and i <= 2:
        f += str(secondsf) + 's'
        i += 1
    if millis > 0 and i <= 2:
        f += str(millis) + 'ms'
        i += 1
    if f == '':
        f = '0ms'
    return f



def model_stats(model):
    for n,m in model.named_parameters():
        print(f'{n}, {m.size()}, {m.numel()}')
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
            if isinstance(m, nn.Linear) :
                model_layer_params['linear']+=m.weight.numel()
            elif isinstance(m, nn.BatchNorm2d) :
                model_layer_params['batchnorm2d']+=m.weight.numel()
            elif isinstance(m, nn.BatchNorm1d) :
                model_layer_params['batchnorm1d']+=m.weight.numel()
            elif isinstance(m, nn.LayerNorm) :
                model_layer_params['layernorm']+=m.weight.numel()
            elif isinstance(m, nn.Conv2d) :
                model_layer_params['conv2d']+=m.weight.numel()
            elif isinstance(m, nn.Conv1d) :
                model_layer_params['conv1d']+=m.weight.numel()
            else:
                if  not isinstance(m, SubnetConv1dTiledFull) and\
                    not isinstance(m, SubnetLinearTiledFull) and\
                    not isinstance(m, SubnetConvTiledFull):
                    print(f'module instance not found for module with weights: {n}')
                    print("add instance before continuing!!!!!!!")
                    sys.exit()
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
            if isinstance(m, SubnetLinearTiledFull) :
                model_layer_params['linear']+=m.weight.numel()/m.compression_factor+m.compression_factor*32
            elif isinstance(m, nn.BatchNorm2d) :
                model_layer_params['batchnorm2d']+=m.weight.numel()
            elif isinstance(m, nn.BatchNorm1d) :
                model_layer_params['batchnorm1d']+=m.weight.numel()
            elif isinstance(m, nn.LayerNorm) :
                model_layer_params['layernorm']+=m.weight.numel()
            elif isinstance(m, SubnetConvTiledFull) :
                model_layer_params['conv2d']+=m.weight.numel()/m.compression_factor+m.compression_factor*32
            elif isinstance(m, SubnetConv1dTiledFull) :
                model_layer_params['conv1d']+=m.weight.numel()/m.compression_factor+m.compression_factor*32
            else:
                if  not isinstance(m, nn.Conv2d) and\
                    not isinstance(m, nn.Conv1d) and\
                    not isinstance(m, nn.Linear):
                    print(f'module instance not found for module with weights: {n}')
                    print("add instance before continuing!!!!!!!")
                    sys.exit()
        if hasattr(m, "bias") and m.bias is not None:
            #print(f'adding {m.bias.numel()} bias params for module {n}')
            model_layer_params['bias']+=m.bias.numel()
    
    print('\ntiled model summary:')
    print("(linear, conv1d, conv2d are compressed)")
    for layer, params in model_layer_params.items():
        print(f'\t{layer}, {params}')