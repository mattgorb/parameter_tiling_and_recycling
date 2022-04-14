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

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, weights, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())
        # flat_out and out access the same memory. switched 0 and 1
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        # Perform binary quantization of weights
        abs_wgt = torch.abs(weights.clone()) # Absolute value of original weights
        q_weight = abs_wgt * out # Remove pruned weights
        num_unpruned = int(k * scores.numel()) # Number of unpruned weights
        alpha = torch.sum(q_weight) / num_unpruned # Compute alpha = || q_weight ||_1 / (number of unpruned weights)

        # Save absolute value of weights for backward
        ctx.save_for_backward(abs_wgt)

        # Return pruning mask with gain term alpha for binary weights
        return alpha * out
    @staticmethod
    def backward(ctx, g):
        # Get absolute value of weights from saved ctx
        abs_wgt, = ctx.saved_tensors
        # send the gradient g times abs_wgt on the backward pass
        return g * abs_wgt, None, None


class SupermaskLinear(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.scores_copy=nn.Parameter(self.scores, requires_grad=False)

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.prune_rate = 0.5

    def rerandomize(self):
        with torch.no_grad():
            if args.rerand == 'recycle':
                sorted, indices = torch.sort(self.scores.abs().flatten())
                j = int((.10) * self.scores.numel())
                low_scores = (self.scores.abs() <  sorted[j]).nonzero(as_tuple=True)
                high_scores = (self.scores.abs() >= sorted[-j]).nonzero(as_tuple=True)
                self.weight[low_scores]=self.weight[high_scores]
                print('recycling {} out of {} weights'.format(j,self.weight.numel()))
            elif args.rerand == 'iterand':
                args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                nn.init.kaiming_normal_(weight_twin, mode="fan_in", nonlinearity="relu")
                ones = torch.ones(self.weight.size()).to(self.weight.device)
                b = torch.bernoulli(ones * .1)
                mask=GetSubnet.apply(self.clamped_scores, self.weight, self.prune_rate)
                t1 = self.weight.data * mask
                t2 = self.weight.data * (1 - mask) * (1 - b)
                t3 = weight_twin.data * (1 - mask) * b
                self.weight.data = t1 + t2 +t3
                print('rerandomizing {} out of {} weights'.format(torch.sum(b), self.weight.numel()))
    def getSparsity(self):
        return torch.mean(GetSubnet.apply(self.scores.abs(), args.sparsity))



    @property
    def clamped_scores(self):
        # For unquantized activations
        return self.scores.abs()

    def forward(self, x):
        # For debugging gradients, prints out maximum value in gradients
        # Get binary mask and gain term for subnetwork
        quantnet = GetSubnet.apply(self.clamped_scores, self.weight, self.prune_rate)
        # Binarize weights by taking sign, multiply by pruning mask and gain term (alpha)
        w = torch.sign(self.weight) * quantnet
        # Pass binary subnetwork weights to convolution layer
        return F.linear(x, w, self.bias)





class NetSparse(nn.Module):
    def __init__(self):
        super(NetSparse, self).__init__()
        self.fc1 = SupermaskLinear(784, args.hidden_size, bias=False)
        self.fc2 = SupermaskLinear(args.hidden_size, 10, bias=False)


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output

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
            torch.save(model.state_dict(), "weights/mnist_mlp_{}.pt".format(args.model_type))

    print("Top Test Accuracy: {}\n".format(best_acc))
    return best_acc

def main():
    global args
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR',
                        help='learning rate (default: 0.1)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='Momentum (default: 0.9)')
    parser.add_argument('--wd', type=float, default=0.0005, metavar='M',
                        help='Weight decay (default: 0.0005)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')

    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--data', type=str, default='/s/luffy/b/nobackup/mgorb/data/', help='Location to store data')
    parser.add_argument('--sparsity', type=float, default=0.5,
                        help='how sparse is each layer')
    parser.add_argument('--model_type', type=str, default=None,
                        help='how sparse is each layer')
    parser.add_argument('--hidden_size', type=int, default=None,
                        help='hidden layer size')
    parser.add_argument('--rerand', type=str, default=None,
                        help='rerand type')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda:4" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(args.data, 'mnist'), train=True, download=True,
                       transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.batch_size, shuffle=True,worker_init_fn=np.random.seed(0), **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join(args.data, 'mnist'), train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize((0.1307,), (0.3081,))
                       ])),
        batch_size=args.test_batch_size, shuffle=True,worker_init_fn=np.random.seed(0), **kwargs)

    if args.model_type is None:
        print('Specify model!')
        sys.exit()

    #if args.model_type=='base':
        #model = Net().to(device)
    if args.model_type=='biprop':
        model = NetSparse().to(device)

    # NOTE: only pass the parameters where p.requires_grad == True to the optimizer! Important!
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.wd,
    )
    criterion = nn.CrossEntropyLoss().to(device)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    best_acc=0

    for epoch in range(1, args.epochs + 1):
        train(model, device, train_loader, optimizer, criterion, epoch)
        best_acc=test(model, device, criterion, test_loader, best_acc)
        scheduler.step()

        if args.model_type=='biprop':
            if args.rerand=='iterand' or args.rerand=='recycle':
                if epoch%1==0:
                    model.fc1.rerandomize()
                    model.fc2.rerandomize()

if __name__ == '__main__':
    main()