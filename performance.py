import os
import pathlib
import random
import time

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed


from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    rerandomize_model,model_stats
)
from utils.schedulers import get_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from args import args
import importlib
import copy
import data
import models
from utils.initializations import set_seed

from utils.layer_type import SubnetConvEdgePopup, SubnetConvBiprop, GetSubnetEdgePopup, GetQuantnet_binary
import matplotlib.pyplot as plt

import numpy as np

def main():
    print('args: {}'.format(args))
    set_seed(args.seed)

    # Simply call main_worker function
    main_worker(args,)


def main_worker(args,):
    #args.gpu = None

    model = get_model(args)
    #model = nn.DataParallel(model,device_ids = [0,1,2])

    #only run CIFAR10
    # Warm-up the model (optional)
    input_tensor=torch.randn(1,3,32,32).to(torch.float32).cuda()
    model=model.to(torch.float32).cuda()

    #if args.arch=='pointnet' or 'pointnet_tiled':
    #input_tensor=0.5*torch.ones(1,3,1024).to(torch.float).cuda()

    model.eval()

    with torch.no_grad():
        model(input_tensor)#warmup
    sys.exit()
    fps_ls=[]
    for i in range(5):
        # Measure FPS
        num_frames = 1000  # Adjust this based on the number of frames you want to process
        start_time = time.time()

        for _ in range(num_frames):
            with torch.no_grad():
                output = model(input_tensor)

        end_time = time.time()
        elapsed_time = end_time - start_time
        fps = num_frames / elapsed_time

        print(f'\nelapsed_time {elapsed_time}')
        print(f'fps {fps}')

        fps_ls.append(fps)
    
    print(f'\n\nmean fps: {np.mean(np.array(fps_ls))}')
    print(f'std fps: {np.std(np.array(fps_ls))}')
    #return fps

def get_model(args,):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"


    if args.arch=='pointnet':
        from pointnet.pointnet_models.pointnet_classification import get_model 
        model=get_model(k=40, normal_channel=False)
    elif args.arch=='tiled_pointnet':
        from pointnet.pointnet_models.tiled_pointnet_classification import get_tiled_model 
        model=get_tiled_model(k=40,normal_channel=False, args=args)

    elif args.arch=='tiled_mlpmixer':
        from vision_transformers_cifar10.models.tiled_mlpmixer_inference import TiledMLPMixer
        model = TiledMLPMixer(
            image_size = 32,
            channels = 3,
            patch_size = 4,
            dim = 512,
            depth = 6,
            num_classes = 10, 
            args=args
        )
    elif args.arch=='mlpmixer':
        from vision_transformers_cifar10.models.mlpmixer import MLPMixer
        model = MLPMixer(
            image_size = 32,
            channels = 3,
            patch_size = 4,
            dim = 512,
            depth = 6,
            num_classes = 10
        )
    elif args.arch=='tiled_swint':
        from vision_transformers_cifar10.models.tiled_swin_inference import tiled_swin_t
        model = tiled_swin_t(window_size=4,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1), args=args)
    elif args.arch=='swint':
        from vision_transformers_cifar10.models.swin import swin_t
        model = swin_t(window_size=args.patch,
                    num_classes=10,
                    downscaling_factors=(2,2,2,1))
    elif args.arch=="tiled_vit":
        # ViT for cifar10
        from vision_transformers_cifar10.models.tiled_vit_inference import TiledViT
        model = TiledViT(
        image_size = 32,
        patch_size = 4,
        num_classes = 10,
        dim = 512,
        depth = 6,
        heads = 8,
        mlp_dim = 512,
        dropout = 0.1,
        emb_dropout = 0.1, 
        args=args
    )
    else:
        print("=> Creating model '{}'".format(args.arch))
        model = models.__dict__[args.arch]()
        dense_params=sum(int(p.numel() ) for n, p in model.named_parameters() if not n.endswith('scores'))
        dense_params="{:,}".format(dense_params)
        print(
            f"=> Dense model params:\n\t{dense_params}"
        )

    

    # applying sparsity to the network
    if args.layer_type != "DenseConv" and args.layer_type!='SubnetConvTiledFull':
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

    if args.layer_type=='SubnetConvTiledFull':
        assert args.model_type=='prune' or args.model_type=='binarize',  'model type needs to be prune or binarize'
        assert args.alpha_type=='single' or args.alpha_type=='multiple', "alpha needs to be single or multiple"
        model_stats(model)

    model_stats(model)


    # freezing the weights if we are only doing subnet training
    if args.layer_type=='SubnetConvEdgePopup' or args.layer_type=='SubnetConvBiprop':
        freeze_model_weights(model)


    return model




def _run_dir_exists(run_base_dir):
    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"

    return log_base_dir.exists() or ckpt_base_dir.exists()


def get_directories(args):
    if args.config is None or args.name is None:
        raise ValueError("Must have name and config")

    config = pathlib.Path(args.config).stem
    if args.log_dir is None:
        run_base_dir = pathlib.Path(
            f"/s/babbage/b/nobackup/nblancha/public-datasets/subnetworks/runs/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    else:
        run_base_dir = pathlib.Path(
            f"{args.log_dir}/{config}/{args.name}/prune_rate={args.prune_rate}"
        )
    if args.width_mult != 1.0:
        run_base_dir = run_base_dir / "width_mult={}".format(str(args.width_mult))

    if _run_dir_exists(run_base_dir):
        rep_count = 0
        while _run_dir_exists(run_base_dir / str(rep_count)):
            rep_count += 1

        run_base_dir = run_base_dir / str(rep_count)

    log_base_dir = run_base_dir / "logs"
    ckpt_base_dir = run_base_dir / "checkpoints"
    print("Base dir {}".format(run_base_dir))
    if not run_base_dir.exists():
        print("Making directory {}".format(run_base_dir))
        os.makedirs(run_base_dir)

    (run_base_dir / "settings.txt").write_text(str(args))

    print(run_base_dir)
    print(ckpt_base_dir)
    print(log_base_dir)
    return run_base_dir, ckpt_base_dir, log_base_dir




if __name__ == "__main__":
    main()


