from __future__ import print_function
import argparse
import os
import math
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import numpy as np

args = None



#straight through estimator with reshaping and aggregation
class GetSubnetBinaryTiled(autograd.Function):
    @staticmethod
    def forward(ctx, scores,  compression_factor, args):
        score_agg=torch.sum(scores.flatten().reshape((compression_factor, 
                            int(scores.numel()/compression_factor))), dim=0)

        out=torch.where(score_agg>0, 1, -1)
        tiled_tensor=out.tile((compression_factor,))

        #important: convert back to float in here.  
        #if you don't scores param won't get gradients. 
        return tiled_tensor.reshape_as(scores).float()
        
    
    @staticmethod
    def backward(ctx, g):
        return g , None, None, None


class LinearSubnet(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def init(self,args, compression_factor):
        self.args=args
        
        if args.global_compression_factor is not None:
            if self.weight.numel()<int(args.min_compress_size):
        
                self.compression_factor=1
            else:
                self.compression_factor=args.global_compression_factor
        else:
            self.compression_factor=compression_factor
            
        assert self.weight.numel()%int(self.compression_factor)==0

        if self.args.alpha_param=='scores':
            self.weight.requires_grad = False

        self.scores.requires_grad=True



    def alpha_scaling(self,quantnet ):
        if self.args.alpha_param=='scores':
            alpha_tens_flattened=torch.abs(self.scores).flatten()
        elif self.args.alpha_param=='weight':
            alpha_tens_flattened=torch.abs(self.weight).flatten()

        quantnet_flatten=quantnet.flatten().float()

        if self.args.alpha_type=='multiple':
            tile_size=int(self.weight.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                abs_weight = alpha_tens_flattened[i*tile_size:(i+1)*tile_size]

                alpha=torch.sum(abs_weight)/tile_size
                quantnet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
                
        else:
            num_unpruned = alpha_tens_flattened.numel()

            alpha=torch.sum(alpha_tens_flattened)/num_unpruned
            quantnet_flatten*=alpha

        return quantnet_flatten.reshape_as(quantnet)
    



    def forward(self, x):
        quantnet = GetSubnetBinaryTiled.apply(self.scores, self.compression_factor,self.args )
        quantnet_scaled=self.alpha_scaling(quantnet)

        # Pass scaled quantnet to convolution layer
        x = F.linear(
            x, quantnet_scaled , self.bias
        )

        return x











class BNN(nn.Module):
    def __init__(self, args):
        super(BNN, self).__init__()
        self.fc1 = LinearSubnet(784, args.hidden_size, bias=False)
        self.fc1.init(args,args.compression_factor)


        self.fc2 = LinearSubnet(args.hidden_size, 10, bias=False)
        self.fc2.init(args,args.compression_factor)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(-1, 784))
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, criterion, test_loader, best_acc):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784))
            test_loss += criterion(output, target)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    acc=correct/len(test_loader.dataset)
    if acc>best_acc:
        best_acc=acc
        if args.save_model:
            torch.save(model.state_dict(), "../weights/mnist_tiled_nn_{}_hidden_{}.pt".format(args.model_type, args.hidden_size))

    print("Top Test Accuracy: {}\n".format(best_acc))
    return best_acc

def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=64, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=150, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')

    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='.', help='Location to store data')

    parser.add_argument('--model_type', type=str, default='tiled',)
    parser.add_argument('--hidden_size', type=int, default=256,  help='hidden layer size')

    parser.add_argument('--compression_factor', type=int, default=4,  )
    parser.add_argument('--global_compression_factor', type=int, default=None,  )
    parser.add_argument('--min_compress_size', type=int, default=64000, )
    parser.add_argument('--alpha_param', type=str, default='weight', )
    parser.add_argument('--alpha_type', type=str, default='multiple', )

    args = parser.parse_args()
    print(args)
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:0" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(args.data, 'mnist'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.batch_size, shuffle=True,worker_init_fn=np.random.seed(0), **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(args.data, 'mnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=args.test_batch_size, shuffle=True,worker_init_fn=np.random.seed(0), **kwargs)




    model = BNN(args).to(device)



    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!

    criterion = nn.CrossEntropyLoss().to(device)
    best_acc=0
    optimizer = torch.optim.Adam(model.parameters(),)

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        best_acc=test(model, device, criterion, test_loader, best_acc)



if __name__ == '__main__':
    main()