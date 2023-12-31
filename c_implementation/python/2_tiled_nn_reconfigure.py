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

    def set_params(self):
        self.set_alphas( )

        score_agg=torch.sum(self.scores.flatten().reshape((self.compression_factor, 
                            int(self.scores.numel()/self.compression_factor))), dim=0)

        self.tile=nn.Parameter(torch.where(score_agg>0, 1, -1),requires_grad=False)
        self.weight_shape=list(self.weight.size())

        del self.weight
        del self.scores

        tiled_vector=self.tile.tile(self.compression_factor)
        tiled_vector.reshape(self.weight_shape)

        print(self.tile[:50])
        print(f'tize size { self.tile.size()}')

        print(f'alphas: {self.alphas}')

        

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

        self.scores.requires_grad=False


    def set_alphas(self,):
        self.alphas=nn.ParameterList()
        if self.args.alpha_param=='scores':
            alpha_tens_flattened=torch.abs(self.scores).flatten()
        elif self.args.alpha_param=='weight':
            alpha_tens_flattened=torch.abs(self.weight).flatten()

        #quantnet_flatten=quantnet.flatten().float()

        if self.args.alpha_type=='multiple':
            tile_size=int(self.weight.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                abs_weight = alpha_tens_flattened[i*tile_size:(i+1)*tile_size]

                alpha=torch.sum(abs_weight)/tile_size
                self.alphas.append(nn.Parameter(alpha))
                #quantnet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
                #self.alphas.append(nn.Parameter(alpha))
                
        else:
            num_unpruned = alpha_tens_flattened.numel()

            alpha=torch.sum(alpha_tens_flattened)/num_unpruned
            self.alphas.append(nn.Parameter(alpha))
            #quantnet_flatten*=alpha
            #self.alphas.append(alpha)

    
    def alpha_scaling(self,quantnet):
        quantnet_flatten=quantnet.flatten().float()

        if self.args.alpha_type=='multiple':
            tile_size=int(quantnet.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                alpha=self.alphas[i]
                quantnet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
        else:
            alpha=self.alphas[0]
            quantnet_flatten*=alpha

        return quantnet_flatten.reshape_as(quantnet)


    def forward(self, x):
        tiled_vector=self.tile.tile(self.compression_factor)

        quantnet_scaled=self.alpha_scaling(tiled_vector.reshape(self.weight_shape))

        # Pass scaled quantnet to convolution layer
        x = F.linear(
            x, quantnet_scaled , self.bias
        )
        return x





class TiledNN(nn.Module):
    def __init__(self, args):
        super(TiledNN, self).__init__()
        self.fc1 = LinearSubnet(784, args.hidden_size, bias=False)
        self.fc1.init(args,args.compression_factor)


        self.fc2 = LinearSubnet(args.hidden_size, 10, bias=False)
        self.fc2.init(args,args.compression_factor)
    def init(self):
        self.fc1.set_params()
        self.fc2.set_params()


    def forward(self, x):
        x = self.fc1(x)
        x = F.relu(x)

        tiled_vector=self.fc1.tile.tile(self.fc1.compression_factor)

        quantnet_scaled=self.fc1.alpha_scaling(tiled_vector.reshape(self.fc1.weight_shape))

        x = self.fc2(x)
        return x



def test(model, device, criterion, test_loader):
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




def predict_x_images(model, device, test_loader,num_images=1):
    for i in range(num_images):
        data, target = test_loader.dataset[i]
        model.eval()

        target=torch.tensor(target)

        data, target = data.to(device), target.to(device)
        #print(data.size())
        output = model(data.view(-1, 784))
        pred = output.argmax(dim=1, keepdim=True)
        print(output)
        prob = output.max(dim=1, keepdim=True)
        print('save image')
        print(f'pred: {pred},prob: {prob} target: {target}')
        #save_image(data, f'images/img{i}.png')

        #print(data)

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
    parser.add_argument('--data', type=str, default='.', help='Location to store data')

    parser.add_argument('--model_type', type=str, default='tiled',
                        help='how sparse is each layer')
    parser.add_argument('--hidden_size', type=int, default=128,  help='hidden layer size')

    parser.add_argument('--compression_factor', type=int, default=4,  )
    parser.add_argument('--global_compression_factor', type=int, default=4,  )
    parser.add_argument('--min_compress_size', type=int, default=64000, )
    parser.add_argument('--alpha_param', type=str, default='weight', )
    parser.add_argument('--alpha_type', type=str, default='multiple', )

    args = parser.parse_args()


    
    transform=transforms.Compose([
        transforms.ToTensor(),
        #transforms.Normalize((0.1307,), (0.3081,))
        ])

    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    
    test_loader = torch.utils.data.DataLoader(dataset2, )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss().to(device)


    model = TiledNN(args).to(device)
    model.eval()
    model.load_state_dict(torch.load(  "../weights/mnist_tiled_nn_128.pt"))

    model.init()



    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())
        if 'alpha' in param_tensor:
            print(model.state_dict()[param_tensor])

    #sys.exit()
    test(model, device, criterion, test_loader,)
    predict_x_images(model, device, test_loader, num_images=1)
    torch.save(model.state_dict(),  "../weights/mnist_tiled_nn_hidden_128_reconfigured.pt")

if __name__ == '__main__':
    main()