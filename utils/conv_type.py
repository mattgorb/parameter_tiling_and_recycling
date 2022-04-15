import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
import math

from args import args as parser_args
from utils.initializations import _init_weight,_init_score
import numpy as np

DenseConv = nn.Conv2d


class GetSubnetSSTL(autograd.Function):
    @staticmethod
    def forward(ctx, scores,):
        # Get the subnetwork by sorting the scores and using the top k%
        return (scores>0).float()

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g

# Not learning weights, finding subnet
class SubnetConvSSTL(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))
        self.scores=nn.Parameter(self.scores)


    def init(self,args):
        self.args=args
        self.weight=_init_weight(self.args, self.weight)
        self.scores=_init_score(self.args, self.scores)
        if args.threshold is None:
            self.th=0
        else:
            self.th=self.args.threshold

    def get_sparsity(self):
        subnet = GetSubnetSSTL.apply(self.scores,)
        temp = subnet.detach().cpu()
        return temp.mean()

    def forward(self, x):
        subnet = GetSubnetSSTL.apply(self.scores, )
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
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

    def descalarization(self, idx, shape):
        res = []
        N = np.prod(shape)
        for n in shape:
            N //= n
            res.append(idx // N)
            idx %= N
        return tuple(res)

    def unravel_index(self,indices,shape,) -> torch.LongTensor:
        r"""Converts flat indices into unraveled coordinates in a target shape.
        This is a `torch` implementation of `numpy.unravel_index`.
        Args:
            indices: A tensor of (flat) indices, (*, N).
            shape: The targeted shape, (D,).
        Returns:
            The unraveled coordinates, (*, N, D).
        """

        coord = []

        for dim in reversed(shape):
            coord.append(indices % dim)
            indices = indices // dim

        coord = torch.stack(coord[::-1], dim=-1)

        return coord

    def rerandomize(self):
        with torch.no_grad():
            if self.args.rerand_type == 'recycle':
                sorted, indices = torch.sort(self.scores.abs().flatten())
                k = int((self.args.rerand_rate) * self.scores.numel())
                #high_scores=torch.tensor([self.descalarization(k, self.scores.abs().size()) for k in torch.topk(self.scores.abs().flatten(), k, largest=True).indices]).long()
                #low_scores=torch.tensor([self.descalarization(k, self.scores.abs().size()) for k in torch.topk(self.scores.abs().flatten(), k, largest=False).indices]).long()
                #print(high_scores)
                _,high_scores=torch.topk(self.scores.abs().flatten(), k,largest=True)
                high_scores=self.unravel_index(high_scores, self.scores.size())
                #print(high_scores)
                #print(high_scores[0].size())
                _,low_scores=torch.topk(self.scores.abs(), k, largest=False)
                low_scores=self.unravel_index(low_scores, self.scores.size())
                #high_scores=torch.tensor([self.descalarization(k, self.scores.size()) for k in torch.topk(self.scores.flatten(), k,  largest=True).indices])
                #low_scores = torch.tensor([self.descalarization(k, self.scores.size()) for k in torch.topk(self.scores.flatten(), k, largest=False).indices])




                self.weight[low_scores]=self.weight[high_scores]
                print('recycling {} out of {} weights'.format(k,self.weight.numel()))


                #low_scores = (self.scores.abs() <  sorted[k]).nonzero(as_tuple=True)
                #high_scores = (self.scores.abs() >= sorted[-k]).nonzero(as_tuple=True)
                #print(low_scores)
                #print(low_scores[0].size())
                sys.exit()
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

    def forward(self, x):
        subnet = GetSubnetEdgePopup.apply(self.clamped_scores, self.prune_rate)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
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
            if args.rerand_type == 'recycle':
                sorted, indices = torch.sort(self.scores.abs().flatten())
                j = int((.10) * self.scores.numel())
                low_scores = (self.scores.abs() <  sorted[j]).nonzero(as_tuple=True)
                high_scores = (self.scores.abs() >= sorted[-j]).nonzero(as_tuple=True)
                self.weight[low_scores]=self.weight[high_scores]
                print('recycling {} out of {} weights'.format(j,self.weight.numel()))
            elif args.rerand_type == 'iterand':
                args.weight_seed += 1
                weight_twin = torch.zeros_like(self.weight)
                nn.init.kaiming_normal_(weight_twin, mode="fan_in", nonlinearity="relu")
                ones = torch.ones(self.weight.size()).to(self.weight.device)
                b = torch.bernoulli(ones * .1)
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