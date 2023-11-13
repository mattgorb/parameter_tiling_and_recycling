import time
import torch
import tqdm

from utils.eval_utils import accuracy_parallel as accuracy
from utils.logging import AverageMeter, ProgressMeter


__all__ = ["train", "validate", "modifier","validate_pretrained"]


def train(train_loader, model, criterion, optimizer, epoch, args, writer,ngpus_per_node):

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
            images = images.to("cuda:{}".format(args.gpu), non_blocking=True)#cuda(args.gpu, non_blocking=True)

        target = target.to("cuda:{}".format(args.gpu), non_blocking=True)#.cuda(args.gpu, non_blocking=True)

        if args.data_type=='float16':
            images = images.to(torch.float16)
        # compute output
        output = model(images)

        loss = criterion(output, target)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        #print(acc1)
        losses.update(loss.item(), images.size(0))
        top1.update(acc1.item(), images.size(0))
        top5.update(acc5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        #torch.cuda.synchronize()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank % ngpus_per_node == 0:
            if i % args.print_freq == 0:
                t = (num_batches * epoch + i) * batch_size
                progress.display(i)
                progress.write_to_tensorboard(writer, prefix="train", global_step=t)


        if args.rerand_iter_freq is not None:
            if epoch > int(args.rerand_warmup):
                if i%args.rerand_iter_freq==0 and epoch>0:
                    rerandomize_model(model, args)
        
    return top1.avg, top5.avg


def validate(val_loader, model, criterion, args, writer, epoch,ngpus_per_node):

    # switch to evaluate mode
    model.eval()

    losses = AverageMeter("Loss", ":.3f", write_val=False)
    top1 = AverageMeter("Acc@1", ":6.2f", write_val=False)
    top5 = AverageMeter("Acc@5", ":6.2f", write_val=False)

    avg_meters=[ losses, top1, top5,]


    progress = ProgressMeter(
        len(val_loader), avg_meters, prefix="Test: "
    )

    with torch.no_grad():
        end = time.time()
        for i, (images, target) in tqdm.tqdm(
            enumerate(val_loader), ascii=True, total=len(val_loader)
        ):
            if args.gpu is not None:
                images = images.cuda(args.gpu, non_blocking=True)

            target = target.cuda(args.gpu, non_blocking=True)

            if args.data_type=='float16':
                images = images.to(torch.float16)
                
            output = model(images)

            loss = criterion(output, target)
            #torch.cuda.synchronize()
            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1.item(), images.size(0))
            top5.update(acc5.item(), images.size(0))

            # measure elapsed time
            #batch_time.update(time.time() - end)
            end = time.time()


            if args.rank % ngpus_per_node == 0:
                if i % args.print_freq == 0:
                    progress.display(i)
        #print(args.rank)
        #print(ngpus_per_node)
        if args.rank % ngpus_per_node == 0:
            progress.display(len(val_loader))
            #print("Acc@1: {}, Acc@5: {}".format(top1.avg, top5.avg))
        if writer is not None:
            progress.write_to_tensorboard(writer, prefix="test", global_step=epoch)

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
