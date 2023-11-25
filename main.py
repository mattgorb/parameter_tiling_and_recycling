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
    rerandomize_model
)
from utils.schedulers import get_policy
from torch.nn.parallel import DistributedDataParallel as DDP

from args import args
import importlib
import copy
import data
import models
from utils.initializations import set_seed

from utils.conv_type import SubnetConvEdgePopup, SubnetConvBiprop, GetSubnetEdgePopup, GetQuantnet_binary
import matplotlib.pyplot as plt

def main():

    print('args: {}'.format(args))


    set_seed(args.seed)

    # Simply call main_worker function
    main_worker(args,)


def main_worker(args,):
    #args.gpu = None
    train, validate, modifier,validate_pretrained = get_trainer(args)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.rerand_iter_freq is not None and args.rerand_epoch_freq is not None:
        print('Use only one of rerand_iter_freq and rerand_epoch_freq')

    model = get_model(args)
    #model = nn.DataParallel(model,device_ids = [0,1,2])

    if args.data_type=='float16':
        model = model.to(torch.float16)
    model,device = set_gpu(args, model,)

    set_seed(args.seed)

    data = get_dataset(args)

    optimizer = get_optimizer(args, model)

    lr_policy = get_policy(args.lr_policy)(optimizer, args)

    if args.label_smoothing is None:
        criterion = nn.CrossEntropyLoss().to(device)
    else:
        criterion = LabelSmoothing(smoothing=args.label_smoothing)

    if args.pretrained:
        pretrained(args, model)

        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )
        print(f'acc: {acc1}')


    # optionally resume from a checkpoint
    best_acc1 = 0.0
    best_acc5 = 0.0
    best_train_acc1 = 0.0
    best_train_acc5 = 0.0

    if args.resume:
        best_acc1 = resume(args, model, optimizer)

    # Data loading code
    if args.evaluate:
        acc1, acc5 = validate(
            data.val_loader, model, criterion, args, writer=None, epoch=args.start_epoch
        )

        #checkpoint=torch.load(args.pretrained)
        print()
        print("EPOCH: {}".format(checkpoint['epoch']))
        print("ACC: {}".format(checkpoint['best_acc1']))

        return

    # Set up directories
    run_base_dir, ckpt_base_dir, log_base_dir = get_directories(args)
    args.ckpt_base_dir = ckpt_base_dir


    writer = SummaryWriter(log_dir=log_base_dir)


    epoch_time = AverageMeter("epoch_time", ":.4f", write_avg=False)
    validation_time = AverageMeter("validation_time", ":.4f", write_avg=False)
    train_time = AverageMeter("train_time", ":.4f", write_avg=False)


    progress_overall = ProgressMeter(
        1, [epoch_time, validation_time, train_time], prefix="Overall Timing"
    )

    end_epoch = time.time()
    args.start_epoch = args.start_epoch or 0
    acc1 = None

    # Save the initial state

    save_checkpoint(
        {
            "epoch": 0,
            "arch": args.arch,
            "state_dict": model.state_dict(),
            "best_acc1": best_acc1,
            "best_acc5": best_acc5,
            "best_train_acc1": best_train_acc1,
            "best_train_acc5": best_train_acc5,
            "optimizer": optimizer.state_dict(),
            "curr_acc1": acc1 if acc1 else "Not evaluated",
        },
        False,
        filename=ckpt_base_dir / f"initial.state",
        save=False,
    )

    # Start training
    lrs=[]
    for epoch in range(args.start_epoch, args.epochs):


        lr_policy(epoch, iteration=None)

        modifier(args, epoch, model)

        cur_lr = get_lr(optimizer)
        lrs.append(cur_lr)

        # train for one epoch
        start_train = time.time()
        train_acc1, train_acc5 = train(
            data.train_loader, model, criterion, optimizer, epoch, args, writer=writer
        )
        train_time.update((time.time() - start_train) / 60)

        # evaluate on validation set
        start_validation = time.time()


        acc1, acc5 = validate(data.val_loader, model, criterion, args, writer, epoch)

        validation_time.update((time.time() - start_validation) / 60)

        print('Current best: {}'.format(best_acc1))




        # remember best acc@1 and save checkpoint
        is_best = acc1 > best_acc1
        best_acc1 = max(acc1, best_acc1)
        best_acc5 = max(acc5, best_acc5)
        best_train_acc1 = max(train_acc1, best_train_acc1)
        best_train_acc5 = max(train_acc5, best_train_acc5)

        save = ((epoch % args.save_every) == 0) and args.save_every > 0

        if is_best or save or epoch == args.epochs - 1:
            if is_best:
                print(f"==> New best, saving at {ckpt_base_dir / 'model_best.pth'}")

            save_checkpoint(
                {
                    "epoch": epoch + 1,
                    "arch": args.arch,
                    "state_dict": model.state_dict(),
                    "best_acc1": best_acc1,
                    "best_acc5": best_acc5,
                    "best_train_acc1": best_train_acc1,
                    "best_train_acc5": best_train_acc5,
                    "optimizer": optimizer.state_dict(),
                    "curr_acc1": acc1,
                    "curr_acc5": acc5,
                },
                is_best,
                filename=ckpt_base_dir / f"epoch_{epoch}.state",
                save=save,
            )

            epoch_time.update((time.time() - end_epoch) / 60)
            progress_overall.display(epoch)
            progress_overall.write_to_tensorboard(
                writer, prefix="diagnostics", global_step=epoch
            )

        if args.rerand_epoch_freq is not None:
            if epoch%args.rerand_epoch_freq==0 and epoch>0 and epoch != args.epochs - 1:
                rerandomize_model(model, args)


        writer.add_scalar("test/lr", cur_lr, epoch)
        end_epoch = time.time()
    
    '''
    args.epochs=args.epochs+250
    for e in range(epoch, args.epochs):

        lr_policy(e, iteration=None)
        modifier(args, e, model)

        cur_lr = get_lr(optimizer)
        lrs.append(cur_lr)
        '''
    


    
    
    write_result_to_csv(
        best_acc1=best_acc1,
        best_acc5=best_acc5,
        best_train_acc1=best_train_acc1,
        best_train_acc5=best_train_acc5,
        prune_rate=args.prune_rate,
        curr_acc1=acc1,
        curr_acc5=acc5,
        base_config=args.config,
        name=args.name,
    )

    config = pathlib.Path(args.config).stem
    print(f"/s/babbage/b/nobackup/nblancha/public-datasets/subnetworks/runs/{config}/{args.name}/prune_rate={args.prune_rate}")


def get_trainer(args):
    print(f"=> Using trainer from trainers.{args.trainer}")
    trainer = importlib.import_module(f"trainers.{args.trainer}")

    return trainer.train, trainer.validate, trainer.modifier, trainer.validate_pretrained


def set_gpu(args, model,):
    assert torch.cuda.is_available(), "CPU-only experiments currently unsupported"
    if args.gpu is not None:
        device=torch.device('cuda:{}'.format(args.gpu))

        model = model.to(device)

    print(device)
    cudnn.benchmark = True
    return model, device


def resume(args, model, optimizer):
    if os.path.isfile(args.resume):
        print(f"=> Loading checkpoint '{args.resume}'")

        checkpoint = torch.load(args.resume,
                                map_location=f"cuda:{args.gpu}"
                                )
        if args.start_epoch is None:
            print(f"=> Setting new start epoch at {checkpoint['epoch']}")
            args.start_epoch = checkpoint["epoch"]

        best_acc1 = checkpoint["best_acc1"]

        model.load_state_dict(checkpoint["state_dict"])

        optimizer.load_state_dict(checkpoint["optimizer"])

        print(f"=> Loaded checkpoint '{args.resume}' (epoch {checkpoint['epoch']})")

        return best_acc1
    else:
        print(f"=> No checkpoint found at '{args.resume}'")
        sys.exit()


def pretrained(args, model):
    if os.path.isfile(args.pretrained):
        print("=> loading pretrained weights from '{}'".format(args.pretrained))
        pretrained = torch.load(
            args.pretrained,
            map_location=torch.device("cuda:{}".format(args.gpu)),
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
        print("=> no pretrained weights found at '{}'".format(args.pretrained))
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
    dense_params=sum(int(p.numel() ) for n, p in model.named_parameters() if not n.endswith('scores'))
    dense_params="{:,}".format(dense_params)
    print(
        f"=> Dense model params:\n\t{dense_params}"
    )

    

    # applying sparsity to the network
    if args.conv_type != "DenseConv" and args.conv_type!='SubnetConvTiledFull':
        if args.prune_rate < 0:
            raise ValueError("Need to set a positive prune rate")

    if args.conv_type=='SubnetConvTiledFull':
        assert args.model_type=='prune' or args.model_type=='binarize',  'model type needs to be prune or binarize'
        assert args.alpha_type=='single' or args.alpha_type=='multiple', "alpha needs to be single or multiple"
    
    if args.conv_type=="SubnetConvTiledFull":

        if args.global_mask_compression_factor is not None:
            tiled_params=sum(int(p.numel()/args.global_mask_compression_factor )+args.global_mask_compression_factor*16 for n, p in model.named_parameters() if not n.endswith('scores'))
            tiled_params+=args.weight_tile_size
            tiled_params="{:,}".format(tiled_params)
            for name, param in model.named_parameters():
                if name.endswith('scores'):
                    continue
                elif 'bn' in name or 'downsample.1' in name:
                    print(f"name: {name}, weight size: {param.numel()}")
                else:
                    print(f"name: {name}, weight shape: {param.size()}, weight size: {param.numel()}, compress factor: {args.global_mask_compression_factor}")
            print(
                f"=> Tiled params: \n\t {tiled_params}"
            )
            
        if args.layer_mask_compression_factors is not None:
            
            #model.layer_mask_compression_factors
            tiled_params=0
            i=0

            for name, param in model.named_parameters():


                if name.endswith('scores'):
                    continue
                elif 'bn' in name or 'downsample.1' in name:
                    print(f"name: {name}, weight size: {param.numel()}")
                else:
                    print(f"name: {name}")
                    print(f"name: {name},i: {i}, weight shape: {param.size()}, weight size: {param.numel()}, compress factor: {model.layer_mask_compression_factors[i]}")
                    tiled_params+=int(param.numel()/model.layer_mask_compression_factors[i])
                    tiled_params+=int(model.layer_mask_compression_factors[i]*16 )
                    #print(model.layer_mask_compression_factors[i])
                    i+=1
            if args.weight_tile_size is not None: 
                tiled_params+=args.weight_tile_size
            tiled_params="{:,}".format(tiled_params)

            print(
                f"=> Tiled params: \n\t {tiled_params}"
            )


    # freezing the weights if we are only doing subnet training
    if args.conv_type=='SubnetConvTiledFull' or args.conv_type=='SubnetConvEdgePopup' or args.conv_type=='SubnetConvBiprop':
        freeze_model_weights(model)

   
    #sys.exit()
    print( f"model weight tile size: {args.weight_tile_size}")

    return model


def get_optimizer(args, model):
    for n, v in model.named_parameters():
        if v.requires_grad:
            print("<DEBUG> gradient to", n)

        if not v.requires_grad:
            print("<DEBUG> no gradient to", n)

    if args.optimizer == "sgd":
        parameters = list(model.named_parameters())
        bn_params = [v for n, v in parameters if ("bn" in n) and v.requires_grad]
        rest_params = [v for n, v in parameters if ("bn" not in n) and v.requires_grad]
        optimizer = torch.optim.SGD(
            [
                {
                    "params": bn_params,
                    "weight_decay": 0 if args.no_bn_decay else args.weight_decay,
                },
                {"params": rest_params, "weight_decay": args.weight_decay},
            ],
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )
    elif args.optimizer == "adam":
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr
        )

    return optimizer


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


def write_result_to_csv(**kwargs):
    results = pathlib.Path("runs") / "results.csv"

    if not results.exists():
        results.write_text(
            "Date Finished, "
            "Base Config, "
            "Name, "
            "Prune Rate, "
            "Current Val Top 1, "
            "Current Val Top 5, "
            "Best Val Top 1, "
            "Best Val Top 5, "
            "Best Train Top 1, "
            "Best Train Top 5\n"
        )

    now = time.strftime("%m-%d-%y_%H:%M:%S")

    with open(results, "a+") as f:
        f.write(
            (
                "{now}, "
                "{base_config}, "
                "{name}, "
                "{prune_rate}, "
                "{curr_acc1:.02f}, "
                "{curr_acc5:.02f}, "
                "{best_acc1:.02f}, "
                "{best_acc5:.02f}, "
                "{best_train_acc1:.02f}, "
                "{best_train_acc5:.02f}\n"
            ).format(now=now, **kwargs)
        )


if __name__ == "__main__":
    main()


