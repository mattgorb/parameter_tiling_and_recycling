import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F

import math

from args import args as parser_args
from utils.initializations import _init_weight,_init_score

DenseConv = nn.Conv2d


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, th=0):
        out = scores.clone()
        out=(out > th).float()

        return out
    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None


# Not learning weights, finding subnet
class SubnetConv(nn.Conv2d):
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

    def rerandomize(self):
        with torch.no_grad():
            self.args.weight_seed+=1
            weight_twin=torch.zeros_like(self.weight)
            nn.init.kaiming_normal_(weight_twin, mode="fan_in", nonlinearity="relu")
            print(weight_twin)
            weight_twin=_init_weight(self.args, weight_twin)
            print(weight_twin)
            sys.exit()
            scores_lt0=(self.scores<=0).nonzero(as_tuple=False)
            if self.args.rerand_type=='iterand':
                j = int((self.args.rerand_rate) * scores_lt0.size(0))
                indices_to_replace=torch.randperm(len(scores_lt0))[:j]
                inds=scores_lt0[indices_to_replace]
                self.weight[inds[:,0], inds[:,1]]=weight_twin[inds[:,0], inds[:,1]]
            elif self.args.rerand_type=='iterand_threshold':
                scores_temp=self.scores[scores_lt0[:,0], scores_lt0[:,1]]
                sorted, indices = torch.sort(scores_temp.flatten())
                j = int((self.args.rerand_rate) * scores_temp.size(0))
                cutoff=sorted[j].item()
                inds = (self.scores < cutoff).nonzero(as_tuple=False)
                self.weight[inds[:,0], inds[:,1]]=weight_twin[inds[:,0], inds[:,1]]

    def get_sparsity(self):
        subnet = GetSubnet.apply(self.scores,self.th)
        temp = subnet.detach().cpu()
        return temp.mean()

    def forward(self, x):
        subnet = GetSubnet.apply(self.scores,self.th)
        w = self.weight * subnet
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x




class GetQuantnet_binary(autograd.Function):
    @staticmethod
    def forward(ctx, scores, weights,th=0 ):
        # Get the subnetwork by sorting the scores and using the top k%
        out = scores.clone()
        out=(out > th).float()

        # Perform binary quantization of weights
        abs_wgt = torch.abs(weights.clone()) # Absolute value of original weights
        q_weight = abs_wgt * out # Remove pruned weights
        num_unpruned = int(torch.sum(out)) # Number of unpruned weights
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


class SubnetBinaryConv(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def get_sparsity(self):
        subnet = GetSubnet.apply(self.scores)
        temp = subnet.detach().cpu()
        return temp.mean()

    def forward(self, x):
        # For debugging gradients, prints out maximum value in gradients
        # Get binary mask and gain term for subnetwork
        quantnet = GetQuantnet_binary.apply(self.scores, self.weight)
        # Binarize weights by taking sign, multiply by pruning mask and gain term (alpha)
        w = torch.sign(self.weight) * quantnet
        # Pass binary subnetwork weights to convolution layer
        x = F.conv2d(
            x, w, self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        # Return output from convolution layer
        return x










