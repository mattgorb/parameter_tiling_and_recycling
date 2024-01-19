import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm as wn
import math
import sys

#from args import args as parser_args
from utils.initializations import _init_weight,_init_score
import numpy as np
from utils.tile_utils import fill_weight_signs

#from utils.kernels import matmul, matmul_tile


#from torch.nn.utils.prune  import l1_unstructured

DenseConv = nn.Conv2d




#straight through estimator with reshaping and aggregation
#used 
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
    


    
class SubnetConvTiledFull(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))


    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(self.args,self.weight)
        self.scores=_init_score(self.args, self.scores)

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

    def alpha_scaling(self,tilednet ):
        if self.args.alpha_param=='scores':
            alpha_tens_flattened=torch.abs(self.scores).flatten()
        elif self.args.alpha_param=='weight':
            alpha_tens_flattened=torch.abs(self.weight).flatten()

        tilednet_flatten=tilednet.flatten().float()

        if self.args.alpha_type=='multiple':
            tile_size=int(self.weight.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                abs_weight = alpha_tens_flattened[i*tile_size:(i+1)*tile_size]

                alpha=torch.sum(abs_weight)/tile_size
                tilednet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
                
        else:
            num_unpruned = alpha_tens_flattened.numel()

            alpha=torch.sum(alpha_tens_flattened)/num_unpruned
            tilednet_flatten*=alpha

        return tilednet_flatten.reshape_as(tilednet)
    



    def forward(self, x):
        tilednet = GetSubnetBinaryTiled.apply(self.scores, self.compression_factor,self.args )
        tilednet_scaled=self.alpha_scaling(tilednet)

        x = F.conv2d(
            x, tilednet_scaled , self.bias, self.stride, self.padding, self.dilation, self.groups
        )
        return x




class SubnetConv1dTiledFull(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))


    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(args,self.weight)
        self.scores=_init_score(self.args, self.scores)

        
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



    def alpha_scaling(self,tilednet ):
        if self.args.alpha_param=='scores':
            alpha_tens_flattened=torch.abs(self.scores).flatten()
        elif self.args.alpha_param=='weight':
            alpha_tens_flattened=torch.abs(self.weight).flatten()

        tilednet_flatten=tilednet.flatten().float()

        if self.args.alpha_type=='multiple':
            tile_size=int(self.weight.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                abs_weight = alpha_tens_flattened[i*tile_size:(i+1)*tile_size]

                alpha=torch.sum(abs_weight)/tile_size
                tilednet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
                
        else:
            num_unpruned = alpha_tens_flattened.numel()

            alpha=torch.sum(alpha_tens_flattened)/num_unpruned
            tilednet_flatten*=alpha

        return tilednet_flatten.reshape_as(tilednet)
    



    def forward(self, x):
        tilednet = GetSubnetBinaryTiled.apply(self.scores, self.compression_factor,self.args )
        tilednet_scaled=self.alpha_scaling(tilednet)

        # Pass scaled tilednet to convolution layer
        x = F.conv1d(
            x, tilednet_scaled , self.bias, self.stride, self.padding, self.dilation, self.groups
        )


        return x





class SubnetLinearTiledFull(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))




    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(args,self.weight)
        self.scores=_init_score(self.args, self.scores)

        
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



    def alpha_scaling(self,tilednet ):
        if self.args.alpha_param=='scores':
            alpha_tens_flattened=torch.abs(self.scores).flatten()
        elif self.args.alpha_param=='weight':
            alpha_tens_flattened=torch.abs(self.weight).flatten()

        tilednet_flatten=tilednet.flatten().float()

        if self.args.alpha_type=='multiple':
            tile_size=int(self.weight.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                abs_weight = alpha_tens_flattened[i*tile_size:(i+1)*tile_size]

                alpha=torch.sum(abs_weight)/tile_size
                tilednet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
                
        else:
            num_unpruned = alpha_tens_flattened.numel()

            alpha=torch.sum(alpha_tens_flattened)/num_unpruned
            tilednet_flatten*=alpha

        return tilednet_flatten.reshape_as(tilednet)
    



    def forward(self, x):
        tilednet = GetSubnetBinaryTiled.apply(self.scores, self.compression_factor,self.args )
        tilednet_scaled=self.alpha_scaling(tilednet)

        # Pass scaled tilednet to convolution layer
        x = F.linear(
            x, tilednet_scaled , self.bias
        )

        return x





#straight through estimator with reshaping and aggregation
#used 
class GetSubnetFPTiled(autograd.Function):
    @staticmethod
    def forward(ctx, scores,  compression_factor, args):
        score_agg=torch.sum(scores.flatten().reshape((compression_factor, 
                            int(scores.numel()/compression_factor))), dim=0)


        tiled_tensor=score_agg.tile((compression_factor,))


        return tiled_tensor.reshape_as(scores).float()
        
    
    @staticmethod
    def backward(ctx, g):
        return g , None, None, None
    


class SubnetLinearTiledFullPrecision(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(args,self.weight)
        self.scores=_init_score(self.args, self.scores)

        
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

        #self.alphas=nn.Parameter(torch.ones(self.compression_factor), requires_grad=True)
        self.alphas=nn.ParameterList([nn.Parameter(torch.ones(1), requires_grad=True) for i in range(self.compression_factor)])


    def forward(self, x):

        tilednet = GetSubnetFPTiled.apply(self.scores, self.compression_factor,self.args )
        # Pass scaled tilednet to convolution layer

        #tilednet=tilednet.flatten()
        tile_size=int(tilednet.numel()/self.compression_factor)
        for i in range(self.compression_factor):
            tilednet.clone().flatten()[i*tile_size:(i+1)*tile_size]*=self.alphas[i]


        x = F.linear(
            x, tilednet , self.bias
        )

        return x



















'''
Inference kernels below
'''

class LinearTiledFullInferenceTritonKernelInference(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))



    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(args,self.weight)
        self.scores=_init_score(self.args, self.scores)
        
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
        self.tile_size=int(self.weight.flatten().numel()//self.compression_factor)

        self.alpha=torch.randn(1)
        self.tile=torch.randn(self.tile_size)*self.alpha

        self.weight_size=self.weight.size()

        if self.args.kernel=='standard':
            self.tiled_tensor=self.tile.flatten().expand((self.compression_factor,self.tile_size))
            self.tiled_tensor=self.tiled_tensor.reshape(self.weight_size).t().contiguous().cuda()

        if isinstance(self.weight_size[0]//self.compression_factor, int) and self.compression_factor>1:
            self.tile=self.tile.reshape(self.weight_size[0]//self.compression_factor, self.weight_size[1])
        else:
            assert self.compression_factor==1
            self.tile=self.tile.reshape_as(self.weight)

        del self.scores
        del self.weight

        if self.args.kernel=='standard':
            del self.tile
        else:
            self.tile=self.tile.t().contiguous().cuda()

    def clean(self):
        torch.cuda.empty_cache()
        #print(f'memory allocated (Bytes): {torch.cuda.memory_allocated()}')
    
    def check_equality(self,output,x):
        if self.args.kernel=='tiled': 
            self.tiled_tensor=self.tile.t().flatten().tile(self.compression_factor).reshape(self.weight_size).t().cuda()

        #if torch.allclose(output, F.linear(x,self.tiled_tensor.cuda().t()), atol=5e-2, rtol=0):
        if torch.allclose(output, F.linear(x,self.tiled_tensor.t()), atol=1e-1, rtol=0):
            print("✅ Triton and Torch match")
        else:
            print("❌ Triton and Torch differ")  
            #print(F.linear(x,self.tiled_tensor.reshape(self.weight_size)))
            #print(output)
            #sys.exit()

        if self.args.kernel=='tiled': 
            del self.tiled_tensor
            
        

    def forward(self, x,):
        if self.args.kernel=='tiled':
            if len(x.size())>2 and x.size()[0]==1:
                x=x.squeeze(0)
                output=matmul_tile(x,self.tile, self.weight_size, self.compression_factor)
                
                if self.args.log_perf:
                    self.check_equality(output, x)
                self.clean()
                return output.unsqueeze(0)
            else:
                output=matmul_tile(x,self.tile, self.weight_size, self.compression_factor)
                if self.args.log_perf:
                    self.check_equality(output, x)
                self.clean()
                return output
        

        if self.args.kernel=='standard':
            #this is the standard kernel
            if len(x.size())>2 and x.size()[0]==1:
                output=matmul(x.squeeze(0),self.tiled_tensor)
                if self.args.log_perf:
                    self.check_equality(output, x)
                self.clean()

                return output.unsqueeze(0)

            else:
                output=matmul(x,self.tiled_tensor)
                if self.args.log_perf:
                    self.check_equality(output, x)
                self.clean()
                return output






















class SubnetConvTiledFullInference(nn.Conv2d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))


    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(self.args,self.weight)
        self.scores=_init_score(self.args, self.scores)

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
        
        self.tile_size=int(self.weight.flatten().numel()/self.compression_factor)

        self.alpha=torch.randn(1)#.to(torch.float)
        self.tile=torch.randn(self.tile_size).sign()*self.alpha


        #self.tiled_tensor=self.tile.tile((self.compression_factor,)).float().cuda()
        self.tiled_tensor=self.tile.expand((self.compression_factor,self.tile_size)).cuda()


        assert self.weight.size(0)%self.compression_factor==0
        


    def alpha_scaling(self,tilednet ):
        if self.args.alpha_param=='scores':
            alpha_tens_flattened=torch.abs(self.scores).flatten()
        elif self.args.alpha_param=='weight':
            alpha_tens_flattened=torch.abs(self.weight).flatten()

        tilednet_flatten=tilednet.flatten().float()

        if self.args.alpha_type=='multiple':
            tile_size=int(self.weight.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                abs_weight = alpha_tens_flattened[i*tile_size:(i+1)*tile_size]

                alpha=torch.sum(abs_weight)/tile_size
                tilednet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
                
        else:
            num_unpruned = alpha_tens_flattened.numel()

            alpha=torch.sum(alpha_tens_flattened)/num_unpruned
            tilednet_flatten*=alpha

        return tilednet_flatten.reshape_as(tilednet)
    

    def forward(self, x):

        #with reshape (Okay performance)
        x = F.conv2d(x, self.tiled_tensor.reshape_as(self.weight), self.bias, self.stride, self.padding, self.dilation, self.groups)

        return x




class SubnetConv1dTiledFullInference(nn.Conv1d):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))


    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(args,self.weight)
        self.scores=_init_score(self.args, self.scores)

        
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

        self.tile_size=int(self.weight.flatten().numel()/self.compression_factor)
        self.alpha=torch.randn(1)
        self.tile=torch.randn(self.tile_size).sign()*self.alpha
        

        #self.tilednet_scaled=self.tiled_tensor.reshape_as(self.weight)*self.alpha

        #self.tiled_tensor=self.tile.tile((self.compression_factor,)).float().cuda()
        self.tiled_tensor=self.tile.expand((compression_factor,self.tile_size)).cuda()

    def alpha_scaling(self,tilednet ):
        if self.args.alpha_param=='scores':
            alpha_tens_flattened=torch.abs(self.scores).flatten()
        elif self.args.alpha_param=='weight':
            alpha_tens_flattened=torch.abs(self.weight).flatten()

        tilednet_flatten=tilednet.flatten().float()

        if self.args.alpha_type=='multiple':
            tile_size=int(self.weight.numel()/self.compression_factor)
            for i in range(self.compression_factor):
                abs_weight = alpha_tens_flattened[i*tile_size:(i+1)*tile_size]

                alpha=torch.sum(abs_weight)/tile_size
                tilednet_flatten[i*tile_size:(i+1)*tile_size]*=alpha
                
        else:
            num_unpruned = alpha_tens_flattened.numel()

            alpha=torch.sum(alpha_tens_flattened)/num_unpruned
            tilednet_flatten*=alpha

        return tilednet_flatten.reshape_as(tilednet)
    

    def forward(self, x):

        # Pass scaled tilednet to convolution layer
        x = F.conv1d(
            x, self.tiled_tensor.reshape_as(self.weight) , self.bias, self.stride, self.padding, self.dilation, self.groups
        )


        return x





class SubnetLinearTiledFullInference(nn.Linear):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))



    def init(self,args, compression_factor):
        self.args=args
        self.weight=_init_weight(args,self.weight)
        self.scores=_init_score(self.args, self.scores)

        
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


        self.tile_size=int(self.weight.flatten().numel()/self.compression_factor)

        self.alpha=torch.randn(1)
        self.tile=torch.randn(self.tile_size)*self.alpha
        #self.tiled_tensor=self.tile.tile((self.compression_factor,)).float().cuda()
        self.tiled_tensor=self.tile.expand((compression_factor,self.tile_size)).cuda()



    def forward(self, x):
        #quantnet_scaled=tiled_tensor.reshape_as(self.weight)*self.alpha
        x = F.linear(
            x,self.tiled_tensor.reshape_as(self.weight), self.bias
        )

        return x


















































#The methods below are from workshop paper on iterative weight recycling
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
        abs_weight = abs_wgt * out # Remove pruned weights
        num_unpruned = int(k * scores.numel()) # Number of unpruned weights
        alpha = torch.sum(abs_weight) / num_unpruned # Compute alpha = || q_weight ||_1 / (number of unpruned weights)

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




