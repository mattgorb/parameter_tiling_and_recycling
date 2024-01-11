# -*- coding: utf-8 -*-
'''

Train CIFAR10 with PyTorch and Vision Transformers!
written by @kentaroy47, @arutema47

'''


#from __future__ import print_function

import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import numpy as np

import torchvision
import torchvision.transforms as transforms

import os

import pandas as pd
import csv
import time
from models.convmixer import ConvMixer
#from models import *
from utils_vit import progress_bar#, model_stats
from randomaug import RandAugment



import sys

sys.path.insert(0, '/s/chopin/l/grad/mgorb/parameter_tiling_and_recycling/')


from utils.net_utils import model_stats


def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate') # resnets.. 1e-3, Vit..1e-4
    parser.add_argument('--opt', default="adam")
    parser.add_argument('--resume', '-r', action='store_true', help='resume from checkpoint')
    parser.add_argument('--noaug', action='store_true', help='disable use randomaug')
    parser.add_argument('--noamp', action='store_true', help='disable mixed precision training. for older pytorch versions')
    parser.add_argument('--nowandb', action='store_true', help='disable wandb')
    parser.add_argument('--mixup', action='store_true', help='add mixup augumentations')
    parser.add_argument('--net', default='vit')
    parser.add_argument('--bs', default='128')
    parser.add_argument('--size', default="32")
    parser.add_argument('--n_epochs', type=int, default='200')
    parser.add_argument('--patch', default='4', type=int, help="patch for ViT")
    parser.add_argument('--dimhead', default="512", type=int)
    parser.add_argument('--convkernel', default='8', type=int, help="parameter for convmixer")

    #tiled model extra arguments
    parser.add_argument('--model_type', default='dense')
    parser.add_argument('--weight_init', default=None)
    parser.add_argument('--score_init', default=None)
    parser.add_argument('--compression_factor', default=None, type=int)
    parser.add_argument('--weight_seed', default=0)
    parser.add_argument('--score_seed', default=0)
    parser.add_argument('--alpha_param', default='weight')
    parser.add_argument('--alpha_type', default='multiple')
    parser.add_argument('--cos', action='store_false', help='Train with cosine annealing scheduling')
    parser.add_argument('--global_compression_factor', default=None, type=int)
    parser.add_argument('--min_compress_size', default=64000, type=int)
    parser.add_argument(
        "--layer_type", type=str, default=None, 
    )

    #print()
    return parser.parse_args()

def main(args):

    '''usewandb = ~args.nowandb
    if usewandb:
        import wandb
        watermark = "{}_lr{}".format(args.net, args.lr)
        wandb.init(project="cifar10-challange",
                name=watermark)
        wandb.config.update(args)'''

    print(args)
    bs = int(args.bs)
    imsize = int(args.size)

    use_amp = False#not args.noamp
    aug = args.noaug

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    best_acc = 0  # best test accuracy
    start_epoch = 0  # start from epoch 0 or last checkpoint epoch

    # Data
    print('==> Preparing data..')
    if args.net=="vit_timm":
        size = 384
    else:
        size = imsize

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.Resize(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.Resize(size),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    # Add RandAugment with N, M(hyperparameter)
    if aug:  
        N = 2; M = 14;
        transform_train.transforms.insert(0, RandAugment(N, M))

    # Prepare dataset
    trainset = torchvision.datasets.CIFAR10(root='/s/lovelace/c/nobackup/iray/mgorb/data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=bs, shuffle=True, num_workers=8)

    testset = torchvision.datasets.CIFAR10(root='/s/lovelace/c/nobackup/iray/mgorb/data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=8)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # Model factory..
    print('==> Building model..')


    if args.net=="convmixer":
        
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = ConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10)
    elif args.net=="mlpmixer":
        from models.mlpmixer import MLPMixer
        net = MLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = 10
    )

    elif args.net=="vit_tiny":
        from models.vit_small import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 4,
        heads = 6,
        mlp_dim = 256,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="simplevit":
        from models.simplevit import SimpleViT
        net = SimpleViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512
    )
    elif args.net=="vit":
        # ViT for cifar10
        from models.vit import ViT
        net = ViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1
    )
    elif args.net=="swin":
        from models.swin import swin_t
        net = swin_t(window_size=args.patch,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1))
    elif args.net=="tiled_vit":
        # ViT for cifar10
        from models.tiled_vit import TiledViT
        net = TiledViT(
        image_size = size,
        patch_size = args.patch,
        num_classes = 10,
        dim = int(args.dimhead),
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1, 
        args=args
    )
    elif args.net=="tiled_mlpmixer":
        from models.tiled_mlpmixer import TiledMLPMixer
        net = TiledMLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = 10, 
        args=args
    )
    elif args.net=="tiled_mlpmixer_inf":
        from models.tiled_mlpmixer_inference import TiledMLPMixer
        net = TiledMLPMixer(
        image_size = 32,
        channels = 3,
        patch_size = args.patch,
        dim = 512,
        depth = 6,
        num_classes = 10, 
        args=args
    )
    elif args.net=="tiled_convmixer":
  
        from models.tiled_convmixer import TiledConvMixer
        # from paper, accuracy >96%. you can tune the depth and dim to scale accuracy and speed.
        net = TiledConvMixer(256, 16, kernel_size=args.convkernel, patch_size=1, n_classes=10, args=args)
    elif args.net=="tiled_swin":
        from models.tiled_swin import tiled_swin_t
        net = tiled_swin_t(window_size=args.patch,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1), args=args)

    model_stats(net)


    # For Multi-GPU
    if 'cuda' in device:
        print(device)
        print("using data parallel")
        net = torch.nn.DataParallel(net) # make parallel
        cudnn.benchmark = True

    if args.resume:
        # Load checkpoint.
        print('==> Resuming from checkpoint..')
        assert os.path.isdir('checkpoint'), 'Error: no checkpoint directory found!'
        checkpoint = torch.load(f'/s/lovelace/c/nobackup/iray/mgorb/checkpoint_transformer/{args.net}_patch{args.patch}_cr{args.compression_factor}_alpha_param_{args.alpha_param}_alpha_type_{args.alpha_type}_global_{args.global_compression_factor}-ckpt.t7')
        net.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        start_epoch = checkpoint['epoch']

    # Loss is CE
    criterion = nn.CrossEntropyLoss()

    if args.opt == "adam":
        optimizer = optim.Adam(net.parameters(), lr=args.lr)
    elif args.opt == "sgd":
        optimizer = optim.SGD(net.parameters(), lr=args.lr)  
        
    # use cosine scheduling
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
    if not args.cos:
        from torch.optim import lr_scheduler
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, verbose=True, min_lr=1e-3*1e-5, factor=0.1)
        print('reduce lr on plateau')
    else:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.n_epochs)
        print('using cosine annealing')
    ##### Training
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    def train(epoch):
        print('\nEpoch: %d' % epoch)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            # Train with amp
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = net(inputs)
                loss = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                % (train_loss/(batch_idx+1), 100.*correct/total, correct, total))
        return 100.*correct/total

    ##### Validation
    def test(epoch, best_acc):
        #global best_acc
        net.eval()
        test_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_idx, (inputs, targets) in enumerate(testloader):
                inputs, targets = inputs.to(device), targets.to(device)
                outputs = net(inputs)
                loss = criterion(outputs, targets)

                test_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

                progress_bar(batch_idx, len(testloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
                    % (test_loss/(batch_idx+1), 100.*correct/total, correct, total))
        
        # Update scheduler
        if not args.cos:
            scheduler.step(test_loss)

        # Save checkpoint.
        acc = 100.*correct/total
        if acc > best_acc:
            print('Saving..')
            state = {"model": net.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict()}
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, f'/s/lovelace/c/nobackup/iray/mgorb/checkpoint_transformer/{args.net}_patch{args.patch}_cr{args.compression_factor}_alpha_param_{args.alpha_param}_alpha_type_{args.alpha_type}_global_{args.global_compression_factor}-ckpt.t7')
            best_acc = acc
        
        os.makedirs("log", exist_ok=True)
        content = time.ctime() + ' ' + f'Epoch {epoch}, lr: {optimizer.param_groups[0]["lr"]:.7f}, val loss: {test_loss:.5f}, acc: {(acc):.5f}'
        print(content)
        print(f'best test accuracy: {best_acc}')
        #with open(f'log/log_{args.net}_patch{args.patch}.txt', 'a') as appender:
            #appender.write(content + "\n")
        return test_loss, acc, best_acc

    list_loss = []
    list_acc = []
    train_acc=[]
    #if usewandb:
        #wandb.watch(net)
        
    net.cuda()
    for epoch in range(start_epoch, args.n_epochs):
        start = time.time()
        trainacc = train(epoch)
        val_loss, acc,best_acc = test(epoch,best_acc)

        if args.cos:
            scheduler.step(epoch-1)
        
        list_loss.append(val_loss)
        list_acc.append(acc)
        train_acc.append(trainacc)
        # Log training..
        #if usewandb:
            #wandb.log({'epoch': epoch, 'train_loss': trainloss, 'val_loss': val_loss, "val_acc": acc, "lr": optimizer.param_groups[0]["lr"],
            #"epoch_time": time.time()-start})

        # Write out csv..
        with open(f'/s/lovelace/c/nobackup/iray/mgorb/vision_transformer_logs/log_{args.net}_patch{args.patch}_cr{args.compression_factor}_alpha_param_{args.alpha_param}_alpha_type_{args.alpha_type}_global_{args.global_compression_factor}.csv', 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(list_loss) 
            writer.writerow(list_acc) 
            writer.writerow(train_acc)
    
if __name__ == "__main__":
    global best_acc
    best_acc=0
    args = parse_args()
    main(args)




