import time
import torch
import tqdm

from utils.eval_utils import accuracy
from utils.logging import AverageMeter, ProgressMeter

from utils.conv_type import SubnetConv,SubnetConvOrig,SubnetBinaryConvOrig,SubnetBinaryConv

__all__ = ["train", "validate", "modifier","validate_pretrained"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer):
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    losses = AverageMeter("Loss", ":.3f")
    top1 = AverageMeter("Acc@1", ":6.2f")
    top5 = AverageMeter("Acc@5", ":6.2f")

    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix=f"Epoch: [{epoch}]",
    )

    # switch to train mode
    model.train()

    batch_size = train_loader.batch_size
    num_batches = len(train_loader)
    end = time.time()
    for i, (images, target) in tqdm.tqdm(
        enumerate(train_loader), ascii=True, total=len(train_loader)
    ):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images = images.to("cuda:{}".format(args.gpu))#cuda(args.gpu, non_blocking=True)

        target = target.to("cuda:{}".format(args.gpu))#.cuda(args.gpu, non_blocking=True)

        # compute output
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            t = (num_batches * epoch + i) * batch_size
            progress.display(i)
            progress.write_to_tensorboard(writer, prefix="train", global_step=t)

    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch):
    # switch to evaluate mode
    model.eval()

    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)

    avg_meters=[ losses, top1, top5,]

    track_subnet=False
    for n, m in model.named_modules() :
        if isinstance(m, SubnetConv) or isinstance(m, SubnetBinaryConv) or isinstance(m, SubnetConvOrig) or isinstance(m, SubnetBinaryConvOrig):
            track_subnet=True
            break
    if track_subnet:
        module_meters={}
        for n, m in model.named_modules():
            if isinstance(m, SubnetConv) or isinstance(m, SubnetBinaryConv) or isinstance(m, SubnetConvOrig) or isinstance(m, SubnetBinaryConvOrig):
                module_meters[n+'_pct_prn']=AverageMeter(n, ":6.2f", write_val=False)

        avg_meters.extend(module_meters.values())

        percent_pruned= AverageMeter("PercentPruned", ":6.2f")
        avg_meters.append(percent_pruned)

    progress = ProgressMeter(
        len(val_loader), avg_meters, prefix="Test: "
    )



    if track_subnet:
        total_not_pruned=0
        total_parameters=0
        for n, m in model.named_modules():
            if isinstance(m, SubnetConv) or isinstance(m, SubnetBinaryConv) or isinstance(m, SubnetConvOrig) or isinstance(m, SubnetBinaryConvOrig):
                module_meters[n+'_pct_prn'].update(1-m.get_sparsity().item())
                total_parameters+=m.scores.size().numel()
                total_not_pruned+=(m.scores.size().numel()*m.get_sparsity())

        total_percent=(1-(total_not_pruned/total_parameters).item())
        percent_pruned.update(total_percent)

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.to("cuda:{}".format(args.gpu))#.cuda(args.gpu, non_blocking=True)

            target = target.to("cuda:{}".format(args.gpu))#.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            #batch_time.update(time.time() - end)
            end = time.time()

            if i % args.print_freq == 0:
                progress.display(i)

        progress.display(len(val_loader))

        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)
    print("Acc@1: {}, Acc@5: {}".format(top1.avg, top5.avg))
    return top1.avg, top5.avg



def validate_pretrained(val_loader, model, criterion, args):
    # switch to evaluate mode
    model.eval()


    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            # compute output
            output = model(images)

            loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))

            print(acc1)
            print(acc5)

            sys.exit()


    return top1.avg, top5.avg



def modifier(args, epoch, model):
    return
