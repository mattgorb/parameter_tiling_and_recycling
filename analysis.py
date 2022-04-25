import os
import pathlib
import random
import time
import pandas as pd
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
from utils.conv_type import SubnetConvEdgePopup, SubnetConvBiprop, GetSubnetEdgePopup, GetQuantnet_binary

from utils.logging import AverageMeter, ProgressMeter
from utils.net_utils import (
    freeze_model_weights,
    save_checkpoint,
    get_lr,
    LabelSmoothing,
    rerandomize_model
)
from utils.schedulers import get_policy
import itertools

from args import args
import importlib

import data
import models
from utils.initializations import set_seed

def main():

    print('args: {}'.format(args))

    set_seed(args.seed)

    # Simply call main_worker function
    main_worker(args)


def main_worker(args):
    #args.gpu = None
    train, validate, modifier,validate_pretrained = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.rerand_iter_freq is not None and args.rerand_epoch_freq is not None:
        print('Use only one of rerand_iter_freq and rerand_epoch_freq')

    set_seed(args.seed)
    data = get_dataset(args)

    print(args.config)

    results_df=None
    count=0

    # create model and optimizer
    weightfile='/s/luffy/b/nobackup/mgorb/runs/conv8-sc-epu/baseline/prune_rate=0.5/0/checkpoints/model_best.pth'

    model = get_model(args)
    model,device = set_gpu(args, model)



    criterion = nn.CrossEntropyLoss().to(device)

    pretrained(weightfile, model)
    acc1, acc5 = validate(data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch )

    model.eval()

    for name,mod in model.named_modules():

        if isinstance(mod, SubnetConvEdgePopup) or isinstance(mod,SubnetConvBiprop):
            if isinstance(mod, SubnetConvBiprop):
                mask1 = GetQuantnet_binary.apply(mod.clamped_scores, mod.weight, mod.prune_rate)
                #y = torch.ones_like(mask1)
                #mask1 = torch.where(mask1 > 0, y, mask1)
                #mask2 = torch.where(mask2 > 0, y, mask2)
            else:
                mask1 = GetSubnetEdgePopup.apply(mod.clamped_scores, mod.prune_rate)
            print(mod.weight.size())
            mask1=torch.nonzero(mask1)
            weights_with_mask=mod.weight[mask1]
            print(weights_with_mask.size())
            sys.exit()


    if results_df is None:
        results_df=pd.DataFrame([vals], columns = cols)
    else:
        df = pd.DataFrame([vals], columns=cols)
        results_df=results_df.append(df)

    print(results_df)

    count+=1
    print(f"Finished {count} of {len(combos)} combinations ")
    results_df.to_csv(f'model_combos_{args.config.split("/")[-1].split(".")[0]}_{args.prune_rate}.csv')



def set_gpu(args, model):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    if args.gpu is not None:
        device=torch.device('cuda:{}'.format(args.gpu))
        model = model.to(device)

        #elif args.multigpu:
        #model = torch.nn.DataParallel(model, device_ids=[4, 5, 6, 7])


    cudnn.benchmark = True

    return model, device

def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier, trainer.validate_pretrained


def pretrained(weight_file, model):
    if os.path.isfile(weight_file):
        print("=> loading pretrained weights from '{}'".format(weight_file))
        pretrained = torch.load(
            weight_file,
            #map_location=torch.device("cuda:{}".format(args.multigpu[0])),
        )["state_dict"]


        model_state_dict = model.state_dict()
        for k, v in pretrained.items():
            if k not in model_state_dict or v.size() != model_state_dict[k].size():
                print("IGNORE:", k)
        pretrained = {
            k: v
            for k, v in pretrained.items()
            if (k in model_state_dict and v.size() == model_state_dict[k].size())
        }
        model_state_dict.update(pretrained)
        model.load_state_dict(model_state_dict)

    else:
        print("=> no pretrained weights found at '{}'".format(weight_file))
        sys.exit()
    #for n, m in model.named_modules():
        #if isinstance(m, FixedSubnetConv):
            #m.set_subnet()

def get_dataset(args):
    print(f"=> Getting {args.set} dataset")
    dataset = getattr(data, args.set)(args)

    return dataset


def get_model(args,):
    if args.first_layer_dense:
        args.first_layer_type = "DenseConv"


    print("=> Creating model '{}'".format(args.arch))
    model = models.__dict__[args.arch]()

    # applying sparsity to the network
    if args.conv_type != "DenseConv":
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

        #set_model_prune_rate(model, prune_rate=args.prune_rate)
        print(
            f"=> Rough estimate model params {sum(int(p.numel() * (1-args.prune_rate)) for n, p in model.named_parameters() if not n.endswith('scores'))}"
        )

    # freezing the weights if we are only doing subnet training
    if args.conv_type=='SubnetConvEdgePopup' or args.conv_type=='SubnetConvBiprop' or args.conv_type=='SubnetConvSSTL':
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
            f"/s/luffy/b/nobackup/mgorb/runs/{config}/{args.name}/prune_rate={args.prune_rate}"
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

