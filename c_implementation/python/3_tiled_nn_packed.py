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
import math
import numpy as np
np.set_printoptions(suppress=True)

from torchvision.utils import save_image

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

        if self.args.alpha_type=='multiple':
            tile_size=int(self.weight.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                abs_weight = alpha_tens_flattened[i*tile_size:(i+1)*tile_size]

                alpha=torch.sum(abs_weight)/tile_size
                self.alphas.append(nn.Parameter(alpha))
                
        else:
            num_unpruned = alpha_tens_flattened.numel()

            alpha=torch.sum(alpha_tens_flattened)/num_unpruned
            self.alphas.append(nn.Parameter(alpha))

    
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


class Conv2dSubnet(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))

    def set_params(self):
        self.set_alphas()

        score_agg=torch.sum(self.scores.flatten().reshape((self.compression_factor, 
                            int(self.scores.numel()/self.compression_factor))), dim=0)

        self.tile=nn.Parameter(torch.where(score_agg>0, 1, -1),requires_grad=False)

        self.weight_shape=list(self.weight.size())

        del self.weight
        del self.scores


        

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
                # self.alphas.append(nn.Parameter(alpha))
                
        else:
            num_unpruned = alpha_tens_flattened.numel()

            alpha=torch.sum(alpha_tens_flattened)/num_unpruned
            self.alphas.append(nn.Parameter(alpha))
            #quantnet_flatten*=alpha
            #self.alphas.append(alpha)
        



    def alpha_scaling(self,quantnet):
        #self.alphas=[]


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
        x = F.conv2d(
            x, quantnet_scaled , self.bias
        )

        return x






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

        self.fc1 = LinearSubnet(9216, 128, bias=None)
        self.fc2 = LinearSubnet(128, 10, bias=None)
        self.fc1.init(args,args.compression_factor)
        self.fc2.init(args,args.compression_factor)


    def init(self):
        self.fc1.set_params()
        self.fc2.set_params()
        self.conv1.set_params()
        self.conv2.set_params()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output




class TiledNN_Packed(nn.Module):
    def __init__(self, args):
        super(TiledNN, self).__init__()
        def pack_col(num):
            return int(num/8) if num%8==0 else int(num/8+1)
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

        self.fc1 = LinearSubnet(9216, 128, bias=None)
        self.fc2 = LinearSubnet(128, 10, bias=None)
        self.fc1.init(args,args.compression_factor)
        self.fc2.init(args,args.compression_factor)

        


    def init(self):
        self.fc1.set_params()
        self.fc2.set_params()
        self.conv1.set_params()
        self.conv2.set_params()

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        #x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        #x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output



class NetSparsePacked(nn.Module):
    def __init__(self,):
        super(NetSparsePacked, self).__init__()
        def pack_col(num):
            return int(num/8) if num%8==0 else int(num/8+1)

        self.fc1 = LinearSubnet(pack_col(784), 256, bias=False)
        self.fc2 = LinearSubnet(pack_col(256), 10, bias=False)

        self.fc1.to_type(torch.uint8)
        self.fc2.to_type(torch.uint8)

    def unpack(self, packed_tensor,unpacked_size):
        matrix = packed_tensor.numpy()
        num_bits = 8  # Number of bits to extract

        shifted_matrix = matrix[:, :, np.newaxis] >> np.arange(num_bits - 1, -1, -1)
        extracted_bits_matrix = shifted_matrix & 1
        extracted_bits_matrix = extracted_bits_matrix.reshape(extracted_bits_matrix.shape[0], -1)[:, :unpacked_size]
        unpacked_binary_data = torch.tensor(extracted_bits_matrix)
        return unpacked_binary_data

    def forward(self, x):
        temp_unpacked=LinearSubnet(784, 256, bias=False)

        temp_unpacked.weight=nn.Parameter((2*self.unpack(self.fc1.weight,784)-1).to(torch.int8), requires_grad=False)
        temp_unpacked.mask=nn.Parameter((self.unpack(self.fc1.mask,784)).to(torch.int8), requires_grad=False)
        temp_unpacked.alpha = self.fc1.alpha



        x = temp_unpacked(x)
        del temp_unpacked

        x = F.relu(x)


        temp_unpacked=LinearSubnet(256, 10, bias=False)
        temp_unpacked.weight=nn.Parameter((2*self.unpack(self.fc2.weight,256)-1).to(torch.int8), requires_grad=False)
        temp_unpacked.mask=nn.Parameter((self.unpack(self.fc2.mask,256)).to(torch.int8), requires_grad=False)
        temp_unpacked.alpha = self.fc2.alpha
        x = temp_unpacked(x)
        del temp_unpacked

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




def save_x_images(model, device, test_loader,num_images=1):
    for i in range(num_images):
        data, target = test_loader.dataset[i]
        model.eval()

        target=torch.tensor(target)
        data, target = data.to(device), target.to(device)
        output = model(data.view(-1, 784))
        pred = output.argmax(dim=1, keepdim=True)
        print('model prediction: ')
        print(output.cpu().detach().numpy())
        prob = output.max(dim=1, keepdim=True)
        #print('save image')
        print(f'pred: {pred},prob: {prob} target: {target}')
        #save_image(data, f'../images/img{i}.png')

        print(data.view(-1, 784))

    #save

def pack(parameter):
    tile_ls=parameter.to(torch.uint8).cpu().numpy().tolist()
    #row=[0] * ( pack_columns* 8 - num_cols)
    print(f'tile size: {len(tile_ls)}')
    result_list = [int(''.join(map(str, tile_ls[i:i + 8])), 2) for i in range(0, len(tile_ls), 8)]

    packed_tensor = torch.tensor(result_list, dtype=torch.uint8)
    print(f'packed size: {packed_tensor.size()}')
    return packed_tensor


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
    parser.add_argument('--hidden_size', type=int, default=512,  help='hidden layer size')

    parser.add_argument('--compression_factor', type=int, default=4,  )
    parser.add_argument('--global_compression_factor', type=int, default=4,  )
    parser.add_argument('--min_compress_size', type=int, default=64000, )
    parser.add_argument('--alpha_param', type=str, default='weight', )
    parser.add_argument('--alpha_type', type=str, default='multiple', )

    args = parser.parse_args()

    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST(os.path.join('.', 'mnist'),download=True, train=False, transform=transforms.Compose([
                           transforms.ToTensor(),
                       ])),
        batch_size=64, shuffle=False,worker_init_fn=np.random.seed(0),)

    device = "cpu"
    criterion = nn.CrossEntropyLoss().to(device)

    state_dict=torch.load("../weights/mnist_tiled_nn_hidden_128_reconfigured.pt")

    packed_dict={}
    for name, val in state_dict.items():
        if 'tile' in name:
            #model.state_dict()[param_tensor].copy_(((model.state_dict()[param_tensor].to(torch.int8)+1)/2))
            
            full_tensor=(val.to(torch.int8)+1)/2

        if 'alpha' not in name:

            packed_tensor=pack(full_tensor)

        else:
            packed_tensor=val
        print(packed_tensor)
        packed_dict[name]=packed_tensor
    
    torch.save(packed_dict, "../weights/mnist_tiled_nn_128_packed.pt")

    
if __name__ == '__main__':
    main()














