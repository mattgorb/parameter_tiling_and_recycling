from __future__ import print_function

from matplotlib import pyplot as plt
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import math
import random
from torch.optim.lr_scheduler import CosineAnnealingLR

import pandas as pd
import numpy as np



import torch


def create_signed_tile(tile_length):
    tile=2*torch.randint(0,2,(tile_length,))-1
    return tile

def fill_weight_signs(weight, tile):
    num_tiles=int(torch.ceil(torch.tensor(weight.numel()/tile.size(0))).item())
    tiled_tensor=tile.tile((num_tiles,))[:weight.numel()]
    tiled_weights=weight.flatten().abs()*tiled_tensor
    return torch.nn.Parameter(tiled_weights.reshape_as(weight), requires_grad=False)


def set_seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def linear_init(in_dim, out_dim, bias=False, args=None, ):
    layer = LinearTiled(in_dim, out_dim, bias=bias)
    layer.init(args)
    return layer


class LinearTiled(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.weight_align = None

    def init(self, args):
        self.args = args
        set_seed(self.args.weight_seed)
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        weights_diff_ae = torch.tensor(0)
        weights_diff_se = torch.tensor(0)
        if self.weight_align is not None:

            weights_diff_ae = torch.sum((self.weight - self.weight_align).abs())
            weights_diff_se = torch.sum(torch.square(self.weight - self.weight_align))

        return x, weights_diff_ae, weights_diff_se

class Net(nn.Module):
    def __init__(self, args, tiled=False):
        super(Net, self).__init__()
        self.args = args
        self.tiled = tiled
        if self.tiled:
            self.fc1 = linear_init(28 * 28, 1024, bias=False, args=self.args, )
            self.fc2 = linear_init(1024, 10, bias=False, args=self.args, )
        else:
            self.fc1 = nn.Linear(28 * 28, 1024, bias=False)
            self.fc2 = nn.Linear(1024, 10, bias=False)

    def forward(self, x,):
        x = self.fc1(x.view(-1, 28 * 28))
        x = F.relu(x)
        x = self.fc2(x)
        return x


def get_datasets(args):
    # not using normalization
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    dataset1 = datasets.MNIST(f'{args.base_dir}data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(f'{args.base_dir}data', train=False, transform=transform)
    train_loader = DataLoader(dataset1, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False)
    return train_loader, test_loader



class Trainer:
    def __init__(self, args, datasets, model, device, model_name):
        self.args = args
        self.model = model
        self.train_loader, self.test_loader = datasets[0], datasets[1]
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.args.lr)
        self.criterion = nn.CrossEntropyLoss(reduction='sum')
        self.device = device
        

        self.model_name = model_name


        #Results lists
        self.weight_align_ae_loss_list=[]
        self.weight_align_se_loss_list=[]

        self.test_accuracy_list=[]
        self.train_loss_list=[]
        self.test_loss_list=[]
        self.batch_epoch_list=[]
        self.epoch_list=[]


    def fit(self, log_output=False):
        self.train_loss = 1e6
        for epoch in range(1, self.args.epochs + 1):
            epoch_loss = self.train()
            self.train_loss = epoch_loss
            test_loss, test_acc = self.test()
            self.test_loss = test_loss
            self.test_acc = test_acc

            if log_output:
                print(f'Epoch: {epoch}, Train loss: {self.train_loss}, Test loss: {self.test_loss}, Test Acc: {self.test_acc}')

    def model_loss(self):
        return self.best_loss

    def train(self, ):
        self.model.train()
        train_loss = 0

        for batch_idx, (data, target) in enumerate(self.train_loader):
            data, target = data.to(self.device), target.to(self.device)
            self.optimizer.zero_grad()
            output = self.model(data)
            loss = self.criterion(output, target)

            train_loss += loss.item()
            loss.backward()
            self.optimizer.step()

        train_loss /= len(self.train_loader.dataset)
        return train_loss

    def test(self, ):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, target in self.test_loader:
                data, target = data.to(self.device), target.to(self.device)
                output, _, _ = self.model(data, )
                test_loss += self.criterion(output, target).item()
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
        test_loss /= len(self.test_loader.dataset)
        return test_loss, 100. * correct / len(self.test_loader.dataset)


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch Weight Align')
    parser.add_argument('--batch-size', type=int, default=256,
                        help='input batch size for training (default: 64)')
    parser.add_argument('--epochs', type=int, default=1,
                        help='number of epochs to train')
    parser.add_argument('--merge_iter', type=int, default=2500,
                        help='number of iterations to merge')
    parser.add_argument('--weight_align_factor', type=int, default=250, )
    parser.add_argument('--lr', type=float, default=1e-3, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7,
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--weight_seed', type=int, default=1, )
    parser.add_argument('--gpu', type=int, default=6, )
    parser.add_argument('--align_loss', type=str, default=None)
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--baseline', type=bool, default=False, help='train base model')
    parser.add_argument('--set_weight_from_weight_align', type=bool, default=True, )
    parser.add_argument('--graphs', type=bool, default=False, help='add norm graphs during training')
    parser.add_argument('--base_dir', type=str, default="/s/luffy/b/nobackup/mgorb/",
                        help='Directory for data and weights')
    args = parser.parse_args()
    set_seed(args.seed)
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    weight_dir = f'{args.base_dir}iwa_weights/'

    train_loader1, test_dataset = get_datasets(args)
    model = Net(args, weight_merge=False).to(device)
    
    trainer = Trainer(args, [train_loader1, test_dataset], model, device, 'model_baseline')
    trainer.fit(log_output=True)




if __name__ == '__main__':
    main()