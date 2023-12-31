from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

import torch.autograd as autograd
import math

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
                print(f'Setting compression size to 1 for layer of size {self.weight.numel()}')
                self.compression_factor=1
            else:
                self.compression_factor=args.global_compression_factor

        else:
            self.compression_factor=compression_factor
            
        assert self.weight.numel()%int(self.compression_factor)==0

        if self.args.alpha_param=='scores':
            self.weight.requires_grad = False

        self.scores.requires_grad=True
        print(f'layer size before compression: {self.weight.numel()}')
        print(f'layer size after compression: {self.weight.numel()//self.compression_factor}')



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


class Conv2dSubnet(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def init(self,args, compression_factor):
        self.args=args
        
        if args.global_compression_factor is not None:
            if self.weight.numel()<int(args.min_compress_size):
                print(f'Setting compression size to 1 for layer of size {self.weight.numel()}')
                self.compression_factor=1
            else:
                self.compression_factor=args.global_compression_factor

        else:
            self.compression_factor=compression_factor
            
        assert self.weight.numel()%int(self.compression_factor)==0

        if self.args.alpha_param=='scores':
            self.weight.requires_grad = False

        self.scores.requires_grad=True
        print(f'layer size before compression: {self.weight.numel()}')
        print(f'layer size after compression: {self.weight.numel()//self.compression_factor}')


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
        x = F.conv2d(
            x, quantnet_scaled , self.bias
        )

        return x
'''
class TiledNN(nn.Module):
    def __init__(self, args):
        super(TiledNN, self).__init__()
        #self.conv1 = nn.Conv2d(1, 32, 3, 1)
        #self.conv2 = nn.Conv2d(32, 64, 3, 1)
        
        self.conv1 = Conv2dSubnet(1, 32, 3, 1, bias=None)
        self.conv2 = Conv2dSubnet(32, 64, 3, 1, bias=None)

        self.conv1.init(args,args.compression_factor)
        self.conv2.init(args,args.compression_factor)

        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        #self.fc1 = nn.Linear(9216, 128)
        #self.fc2 = nn.Linear(128, 10)

        self.fc1 = LinearSubnet(9216, args.hidden_size, bias=None)
        self.fc2 = LinearSubnet(args.hidden_size, 10, bias=None)
        self.fc1.init(args,args.compression_factor)
        self.fc2.init(args,args.compression_factor)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output
    '''
class TiledNN(nn.Module):
    def __init__(self, args):
        super(TiledNN, self).__init__()
        self.fc1 = LinearSubnet(784, args.hidden_size, bias=False)
        self.fc1.init(args,args.compression_factor)


        self.fc2 = LinearSubnet(args.hidden_size, 10, bias=False)
        self.fc2.init(args,args.compression_factor)

    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output


def train(args, model, device, train_loader, optimizer, epoch):
    criterion = nn.CrossEntropyLoss().to(device)
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
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    criterion = nn.CrossEntropyLoss().to(device)
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(-1, 784))
            test_loss += criterion(output, target).item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))
    return correct / len(test_loader.dataset)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, )
    parser.add_argument('--test-batch-size', type=int, default=1000, )
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.75, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')

    parser.add_argument('--dry-run', action='store_true', default=False,)
    parser.add_argument('--seed', type=int, default=1, )
    parser.add_argument('--log-interval', type=int, default=150, )
    parser.add_argument('--save-model', action='store_true', default=False,)
    
    parser.add_argument('--hidden_size', type=int, default=128,  )
    parser.add_argument('--compression_factor', type=int, default=4,  )
    parser.add_argument('--global_compression_factor', type=int, default=4,  )
    parser.add_argument('--min_compress_size', type=int, default=64000, )
    parser.add_argument('--alpha_param', type=str, default='weight', )
    parser.add_argument('--alpha_type', type=str, default='multiple', )
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    #use_mps = not args.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(args.seed)

    if use_cuda:
        device = torch.device("cuda")

    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
       # transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)


    #model=Net().to(device)#99.2% accuracy on test set. 

    model = TiledNN(args).to(device)


    print(model)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=4, gamma=args.gamma)
    best_acc=0
    for epoch in range(1, args.epochs + 1):
        train(args, model, device, train_loader, optimizer, epoch)
        acc=test(model, device, test_loader)
        if acc>best_acc:
            print(f'new best accuracy, {acc}, saving model. ')
            best_acc=acc
            #if args.save_model:
            torch.save(model.state_dict(),  f"../weights/mnist_tiled_nn_{args.hidden_size}.pt")
        else:
            print(f'best test accuracy: {best_acc}')
        scheduler.step()




if __name__ == '__main__':
    main()