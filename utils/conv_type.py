import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
import math
import sys

from args import args as parser_args
from utils.initializations import _init_weight,_init_score
import numpy as np
from utils.tile_utils import fill_weight_signs

#from torch.nn.utils.prune  import l1_unstructured

DenseConv = nn.Conv2d





class GetSubnetBinaryTiled(autograd.Function):
    @staticmethod
    def forward(ctx, scores,  compression_factor, args):
        

        score_agg=torch.sum(scores.clone().flatten().reshape((compression_factor, 
                            int(scores.clone().numel()/compression_factor))), dim=0)
        
        out=torch.where(score_agg.clone()>=0, 1, -1)
        tiled_tensor=out.tile((compression_factor,))#[:scores.numel()]
        out=tiled_tensor.reshape_as(scores) # tiled

        return out
        
    
    @staticmethod
    def backward(ctx, g):
        return g , None, None




    
class SubnetConvTiledFull(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def rerandomize(self):
        with torch.no_grad():
            if self.args.rerand_type == 'recycle':
                sorted, indices = torch.sort(self.scores.abs().flatten())
                k = int((self.args.rerand_rate) * self.scores.numel())
                low_scores=indices[:k]
                high_scores=indices[-k:]
                self.weight.flatten()[low_scores]=self.weight.flatten()[high_scores]
                print('recycling {} out of {} weights'.format(k,self.weight.numel()))
            elif self.args.rerand_type == 'iterand':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                device = self.weight.device
                ones = torch.ones(self.weight.size()).to(self.weight.device)
                b = torch.bernoulli(ones * self.args.rerand_rate)
                mask=GetQuantnet_binary.apply(self.clamped_scores, self.weight, self.prune_rate)
                t1 = self.weight.data * mask
                t2 = self.weight.data * (1 - mask) * (1 - b)
                t3 = weight_twin.data * (1 - mask) * b
                self.weight.data = t1 + t2 +t3
                self.weight=fill_weight_signs(self.weight.to(device),  self.weight_tile.to(device))
                self.weight=self.weight.to(device)

                print('rerandomizing {} out of {} weights'.format(torch.sum(b), self.weight.numel()))
            elif self.args.rerand_type == 'rerandomize_and_tile':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                device = self.weight.device
                self.weight=torch.nn.Parameter(weight_twin)
                self.weight=fill_weight_signs(self.weight.to(device),  self.weight_tile.to(device))
                self.weight=self.weight.to(device)
                
                print(f'rerandomizing and tiling {self.weight.numel()} weights ')
            elif self.args.rerand_type == 'rerandomize_only':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                device = self.weight.device
                self.weight=torch.nn.Parameter(weight_twin)
                self.weight=self.weight.to(device)
                print(f'rerandomizing {self.weight.numel()} weights ')
            else:
                print('set rerand type.')
                sys.exit(0)


    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(args,self.weight)
        self.scores=_init_score(self.args, self.scores)
        #self.prune_rate=args.prune_rate
        
        self.compression_factor=compression_factor

        assert self.weight.numel()%self.compression_factor ==0 
        self.weight.requires_grad = True


    def alpha_scaling(self,quantnet ):
        weights_flattened=torch.abs(self.weight.clone()).flatten()
        quantnet_flatten=quantnet.clone().flatten().float()

        if self.args.alpha_type=='multiple':
            tile_size=int(self.scores.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                q_weight = weights_flattened[i*tile_size:(i+1)*tile_size]

                alpha=torch.sum(q_weight)/tile_size
                #print(alpha)
                quantnet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
                #print(quantnet_flatten[i*tile_size:(i+1)*tile_size][:10])
                #print(quantnet_flatten[0*tile_size:(0+1)*tile_size][:10])
                
        else:
            num_unpruned = weights_flattened.numel()
            alpha=torch.sum(weights_flattened)/num_unpruned
            quantnet_flatten*=alpha
        #sys.exit()
        return quantnet_flatten.reshape_as(quantnet)
    

    def forward(self, x):

        quantnet = GetSubnetBinaryTiled.apply(self.scores, self.compression_factor,self.args )

        quantnet_scaled=self.alpha_scaling(quantnet)

        # Pass scaled quantnet to convolution layer
        x = F.conv2d(
            x, quantnet_scaled , self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        return x




class SubnetConv1dTiledFull(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def rerandomize(self):
        with torch.no_grad():
            if self.args.rerand_type == 'recycle':
                sorted, indices = torch.sort(self.scores.abs().flatten())
                k = int((self.args.rerand_rate) * self.scores.numel())
                low_scores=indices[:k]
                high_scores=indices[-k:]
                self.weight.flatten()[low_scores]=self.weight.flatten()[high_scores]
                print('recycling {} out of {} weights'.format(k,self.weight.numel()))
            elif self.args.rerand_type == 'iterand':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                device = self.weight.device
                ones = torch.ones(self.weight.size()).to(self.weight.device)
                b = torch.bernoulli(ones * self.args.rerand_rate)
                mask=GetQuantnet_binary.apply(self.clamped_scores, self.weight, self.prune_rate)
                t1 = self.weight.data * mask
                t2 = self.weight.data * (1 - mask) * (1 - b)
                t3 = weight_twin.data * (1 - mask) * b
                self.weight.data = t1 + t2 +t3
                self.weight=fill_weight_signs(self.weight.to(device),  self.weight_tile.to(device))
                self.weight=self.weight.to(device)

                print('rerandomizing {} out of {} weights'.format(torch.sum(b), self.weight.numel()))
            elif self.args.rerand_type == 'rerandomize_and_tile':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                device = self.weight.device
                self.weight=torch.nn.Parameter(weight_twin)
                self.weight=fill_weight_signs(self.weight.to(device),  self.weight_tile.to(device))
                self.weight=self.weight.to(device)
                
                print(f'rerandomizing and tiling {self.weight.numel()} weights ')
            elif self.args.rerand_type == 'rerandomize_only':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                device = self.weight.device
                self.weight=torch.nn.Parameter(weight_twin)
                self.weight=self.weight.to(device)
                print(f'rerandomizing {self.weight.numel()} weights ')
            else:
                print('set rerand type.')
                sys.exit(0)


    def init(self,args,compression_factor):
        self.args=args
        self.weight=_init_weight(args,self.weight)
        self.scores=_init_score(self.args, self.scores)
        self.prune_rate=args.prune_rate
        
        self.compression_factor=compression_factor

        assert self.weight.numel()%self.compression_factor ==0 
        self.weight.requires_grad = True
        
        #self.alphas=torch.nn.ParameterList()
        #for i in range(self.compression_factor):
        #    self.alphas.append(torch.nn.Parameter(torch.randn(1)*0.01, requires_grad=True))

    def alpha_scaling(self,quantnet ):
        weights_flattened=torch.abs(self.weight).flatten()
        quantnet_flatten=quantnet.clone().flatten().float()
        #if self.args.data_type=='float16':
            #quantnet_flatten=quantnet_flatten.to(torch.float16)

        if self.args.alpha_type=='multiple':
            tile_size=int(self.scores.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                q_weight = weights_flattened[i*tile_size:(i+1)*tile_size]
                alpha=torch.sum(q_weight)/tile_size
                quantnet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
        else:
            num_unpruned = weights_flattened.numel()
            alpha=torch.sum(weights_flattened)/num_unpruned
            quantnet_flatten*=alpha

        return quantnet_flatten.reshape_as(quantnet)
    

    def forward(self, x):

        quantnet = GetSubnetBinaryTiled.apply(self.scores, self.compression_factor,self.args )

        quantnet_scaled=self.alpha_scaling(quantnet)

        # Pass scaled quantnet to convolution layer
        x = F.Conv1d(
            x, quantnet_scaled , self.bias, self.stride, self.padding, self.dilation, self.groups
        )


        return x





class SubnetLinearTiledFull(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def rerandomize(self):
        with torch.no_grad():
            if self.args.rerand_type == 'recycle':
                sorted, indices = torch.sort(self.scores.abs().flatten())
                k = int((self.args.rerand_rate) * self.scores.numel())
                low_scores=indices[:k]
                high_scores=indices[-k:]
                self.weight.flatten()[low_scores]=self.weight.flatten()[high_scores]
                print('recycling {} out of {} weights'.format(k,self.weight.numel()))
            elif self.args.rerand_type == 'iterand':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                device = self.weight.device
                ones = torch.ones(self.weight.size()).to(self.weight.device)
                b = torch.bernoulli(ones * self.args.rerand_rate)
                mask=GetQuantnet_binary.apply(self.clamped_scores, self.weight, self.prune_rate)
                t1 = self.weight.data * mask
                t2 = self.weight.data * (1 - mask) * (1 - b)
                t3 = weight_twin.data * (1 - mask) * b
                self.weight.data = t1 + t2 +t3
                self.weight=fill_weight_signs(self.weight.to(device),  self.weight_tile.to(device))
                self.weight=self.weight.to(device)

                print('rerandomizing {} out of {} weights'.format(torch.sum(b), self.weight.numel()))
            elif self.args.rerand_type == 'rerandomize_and_tile':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                device = self.weight.device
                self.weight=torch.nn.Parameter(weight_twin)
                self.weight=fill_weight_signs(self.weight.to(device),  self.weight_tile.to(device))
                self.weight=self.weight.to(device)
                
                print(f'rerandomizing and tiling {self.weight.numel()} weights ')
            elif self.args.rerand_type == 'rerandomize_only':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                device = self.weight.device
                self.weight=torch.nn.Parameter(weight_twin)
                self.weight=self.weight.to(device)
                print(f'rerandomizing {self.weight.numel()} weights ')
            else:
                print('set rerand type.')
                sys.exit(0)


    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(args,self.weight)
        self.scores=_init_score(self.args, self.scores)
        self.prune_rate=args.prune_rate
        
        self.compression_factor=compression_factor

        assert self.weight.numel()%self.compression_factor ==0 
        self.weight.requires_grad = True
        
        #self.alphas=torch.nn.ParameterList()
        #for i in range(self.compression_factor):
        #    self.alphas.append(torch.nn.Parameter(torch.randn(1)*0.01, requires_grad=True))

    def alpha_scaling(self,quantnet ):
        weights_flattened=torch.abs(self.weight).flatten()
        quantnet_flatten=quantnet.clone().flatten().float()
        #if self.args.data_type=='float16':
            #quantnet_flatten=quantnet_flatten.to(torch.float16)

        if self.args.alpha_type=='multiple':
            tile_size=int(self.scores.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                q_weight = weights_flattened[i*tile_size:(i+1)*tile_size]
                alpha=torch.sum(q_weight)/tile_size
                quantnet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
        else:
            num_unpruned = weights_flattened.numel()
            alpha=torch.sum(weights_flattened)/num_unpruned
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






























class GetQuantnet_binary(autograd.Function):
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

class SubnetConvBiprop(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    @property
    def clamped_scores(self):
        # For unquantized activations
        return self.scores.abs()

    def init(self,args):
        self.args=args
        self.weight=_init_weight(self.args, self.weight)
        self.scores=_init_score(self.args, self.scores)
        self.prune_rate=args.prune_rate

    def rerandomize(self):
        with torch.no_grad():
            if self.args.rerand_type == 'recycle':
                sorted, indices = torch.sort(self.scores.abs().flatten())
                k = int((self.args.rerand_rate) * self.scores.numel())
                low_scores=indices[:k]
                if self.args.ablation:
                    print("Ablation recycling")
                    high_scores=indices[k:2*k]
                else:
                    high_scores=indices[-k:]
                self.weight.flatten()[low_scores]=self.weight.flatten()[high_scores]
                print('recycling {} out of {} weights'.format(k,self.weight.numel()))

            elif self.args.rerand_type == 'iterand':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)

                ones = torch.ones(self.weight.size()).to(self.weight.device)
                b = torch.bernoulli(ones * self.args.rerand_rate)
                mask=GetQuantnet_binary.apply(self.clamped_scores, self.weight, self.prune_rate)
                t1 = self.weight.data * mask
                t2 = self.weight.data * (1 - mask) * (1 - b)
                t3 = weight_twin.data * (1 - mask) * b
                self.weight.data = t1 + t2 +t3


                print('rerandomizing {} out of {} weights'.format(torch.sum(b), self.weight.numel()))

    def forward(self, x):

        # Get binary mask and gain term for subnetwork
        quantnet = GetQuantnet_binary.apply(self.clamped_scores, self.weight, self.prune_rate)
        # Binarize weights by taking sign, multiply by pruning mask and gain term (alpha)
        w = torch.sign(self.weight) * quantnet
        # Pass binary subnetwork weights to convolution layer
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        # Return output from convolution layer
        return x










class GetSubnetEdgePopup(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((1 - k) * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


# Not learning weights, finding subnet
class SubnetConvEdgePopup(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def init(self,args):
        self.args=args
        self.weight=_init_weight(self.args, self.weight)
        self.scores=_init_score(self.args, self.scores)
        self.prune_rate=args.prune_rate

    @property
    def clamped_scores(self):
        return self.scores.abs()

    def rerandomize(self):
        with torch.no_grad():
            if self.args.rerand_type == 'recycle':
                sorted, indices = torch.sort(self.scores.abs().flatten())
                k = int((self.args.rerand_rate) * self.scores.numel())
                low_scores=indices[:k]
                if self.args.ablation:
                    print("Ablation recycling")
                    high_scores=indices[k:2*k]
                else:
                    high_scores=indices[-k:]
                self.weight.flatten()[low_scores]=self.weight.flatten()[high_scores]
                print('recycling {} out of {} weights'.format(k,self.weight.numel()))

            elif self.args.rerand_type == 'iterand':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)

                ones = torch.ones(self.weight.size()).to(self.weight.device)
                b = torch.bernoulli(ones * self.args.rerand_rate)
                mask=GetSubnetEdgePopup.apply(self.clamped_scores,  self.prune_rate)
                t1 = self.weight.data * mask
                t2 = self.weight.data * (1 - mask) * (1 - b)
                t3 = weight_twin.data * (1 - mask) * b
                self.weight.data = t1 + t2 +t3
                print('rerandomizing {} out of {} weights'.format(torch.sum(b), self.weight.numel()))
            elif self.args.rerand_type == 'iterand_recycle':
                self.args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                weight_twin = _init_weight(self.args, weight_twin)
                ones = torch.ones(self.weight.size()).to(self.weight.device)
                b = torch.bernoulli(ones * float(self.args.rerand_rate/2))
                mask = GetSubnetEdgePopup.apply(self.clamped_scores, self.prune_rate)
                t1 = self.weight.data * mask
                t2 = self.weight.data * (1 - mask) * (1 - b)
                t3 = weight_twin.data * (1 - mask) * b
                self.weight.data = t1 + t2 + t3
                print('rerandomizing {} out of {} weights'.format(torch.sum(b), self.weight.numel()))

                sorted, indices = torch.sort(self.scores.abs().flatten())
                k = int((self.args.rerand_rate/2) * self.scores.numel())
                low_scores=indices[:k]
                high_scores=indices[-k:]
                self.weight.flatten()[low_scores]=self.weight.flatten()[high_scores]
                print('recycling {} out of {} weights'.format(k,self.weight.numel()))

    def forward(self, x):
        subnet = GetSubnetEdgePopup.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x


