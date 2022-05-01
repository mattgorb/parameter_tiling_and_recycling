import os

import torch
from torchvision import datasets, transforms

import torch.multiprocessing

from torch.utils.data.distributed import DistributedSampler

#torch.multiprocessing.set_sharing_strategy("file_system")

class ImageNet:
    def __init__(self, args):
        super(ImageNet, self).__init__()

        data_root = os.path.join(args.data, "imagenet")

        use_cuda = torch.cuda.is_available()

        # Data loading code
        kwargs = {"num_workers": args.workers, "pin_memory": True} if use_cuda else {}

        # Data loading code
        traindir = os.path.join(data_root, "train")
        valdir = os.path.join(data_root, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_dataset = datasets.ImageFolder(
            traindir,
            transforms.Compose(
                [
                    transforms.RandomResizedCrop(224),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )
        val_dataset=datasets.ImageFolder(
            valdir,
            transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            ),
        )

        if args.multigpu:
            self.train_sampler = DistributedSampler(train_dataset)
            self.val_sampler = DistributedSampler(val_dataset)
        else:
            self.train_sampler=None
            self.val_sampler=None

        self.train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, sampler=self.train_sampler,**kwargs
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,sampler=self.val_sampler,
            **kwargs
        )
