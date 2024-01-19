import os
import pathlib
import random
import time
import sys

from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed

import logging
import socket
from datetime import datetime, timedelta

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
import torchsummary

from torch.profiler import profile, record_function, ProfilerActivity

def main():
    print('args: {}'.format(args))
    set_seed(args.seed)

    # Simply call main_worker function
    main_worker(args,)


def main_worker(args,):
    #args.gpu = None
    if args.kernel is None:
        print("set kernel!")
        sys.exit()
    model = get_model(args)
    #model = nn.DataParallel(model,device_ids = [0,1,2])

    #only run CIFAR10
    # Warm-up the model (optional)
    
    model=model.to(torch.float32).cuda()

    if args.arch=='pointnet' or args.arch== 'pointnet_tiled':
        input_tensor=0.25*torch.randn(1,3,1024).abs().to(torch.float).cuda()
    elif args.arch=='tiled_vit_imagenet' or args.arch=='tiled_vit_imagenet_measure':
        input_tensor=torch.randn(1,3,256,256).to(torch.float32).cuda()
    else:
        input_tensor=torch.randn(1,3,32,32).to(torch.float32).cuda()

    model.eval()
    #def get_memory_usage():
        #memory_stats = torch.cuda.memory_stats()
        #print(memory_stats)
        #for key, value in memory_stats.items():
            #print(f"{key}: {value / 1024 / 1024:.2f} MB")

    # Call this function before and after forward pass to analyze memory usage

    with torch.no_grad():
        model(input_tensor)#warmup

    
    #for i, size in enumerate(model.activations_sizes):
        #print(f"Layer {i + 1}: {size}")

    #sys.exit()
    logging.basicConfig(
        format="%(levelname)s:%(asctime)s %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    logger: logging.Logger = logging.getLogger(__name__)
    logger.setLevel(level=logging.INFO)

    TIME_FORMAT_STR: str = "%b_%d_%H_%M_%S"

    def trace_handler(prof: torch.profiler.profile):
        # Prefix for file names.
        # Prefix for file names.
        host_name = socket.gethostname()
        timestamp = datetime.now().strftime(TIME_FORMAT_STR)
        file_prefix = f"{host_name}_{timestamp}"


    #print(prof.key_averages().table())
    with torch.profiler.profile(
        activities=[
            torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(wait=0, warmup=0, active=1, repeat=0),
        #on_trace_ready=torch.profiler.tensorboard_trace_handler(f'perf_log/kernel_{args.kernel}/model_{args.arch}/log_{args.log_perf}_compress{args.compression_factor}/'),
        on_trace_ready=trace_handler,
        record_shapes=True,
        profile_memory=True,
        with_stack=True) as prof:
        model(input_tensor)

    #prof.export_memory_timeline(f'perf_log/memory/kernel_{args.kernel}_model_{args.arch}_log_{args.log_perf}_compress{args.compression_factor}.json', device="cuda:0")


    # Construct the trace file.
    #prof.export_chrome_trace(f'perf_log/memory/kernel_{args.kernel}_model_{args.arch}_log_{args.log_perf}_compress{args.compression_factor}.json.gz')

    # Construct the memory timeline file.
    #prof.export_memory_timeline(f'perf_log/memory/kernel_{args.kernel}_model_{args.arch}_log_{args.log_perf}_compress{args.compression_factor}.html', device="cuda:0")


    if not args.perf_speed:
        return
    fps_ls=[]
    print('Running Speed Tests.')
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
        from pointnet.pointnet_models.tiled_pointnet_classification_inference import get_tiled_model 
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
    elif args.arch=="tiled_vit_imagenet":
        # ViT for cifar10
        from vision_transformers_cifar10.models.tiled_vit_inference import TiledViT
        model = TiledViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
        dropout = 0.1,
        emb_dropout = 0.1, 
        args=args
    )
    elif args.arch=="tiled_vit_imagenet_measure":
        # ViT for cifar10
        from vision_transformers_cifar10.models.tiled_vit_measure import TiledViT
        model = TiledViT(
        image_size = 256,
        patch_size = 32,
        num_classes = 1000,
        dim = 1024,
        depth = 6,
        heads = 16,
        mlp_dim = 2048,
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


