import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import math
import random

def set_seed(seed):

    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def _init_weight(args,weight):
    set_seed(args.weight_seed)
    if args.weight_init == "signed_constant":


        #using signed constant from iterand code
        fan = nn.init._calculate_correct_fan(weight, 'fan_in')
        gain = nn.init.calculate_gain('relu')
        std = gain / math.sqrt(fan)
        nn.init.kaiming_normal_(weight)  # use only its sign
        weight.data = weight.data.sign() * std
        #weight.data *= scale


    elif args.weight_init == "unsigned_constant":

        fan = nn.init._calculate_correct_fan(weight, args.mode)
        if args.scale_fan:
            fan = fan * (1 - args.prune_rate)

        gain = nn.init.calculate_gain(args.nonlinearity)
        std = gain / math.sqrt(fan)
        weight.data = torch.ones_like(weight.data) * std

    elif args.weight_init == "kaiming_normal":

        if args.scale_fan:
            fan = nn.init._calculate_correct_fan(weight, args.mode)
            fan = fan * (1 - args.prune_rate)
            gain = nn.init.calculate_gain(args.nonlinearity)
            std = gain / math.sqrt(fan)
            with torch.no_grad():
                weight.data.normal_(0, std)
        else:
            nn.init.kaiming_normal_(
                weight, mode=args.mode, nonlinearity=args.nonlinearity
            )

    elif args.weight_init == "kaiming_uniform":
        nn.init.kaiming_uniform_(
            weight, mode=args.mode, nonlinearity=args.nonlinearity
        )


    elif args.weight_init == "xavier_normal":
        nn.init.xavier_normal_(weight)
    elif args.weight_init == "xavier_constant":

        fan_in, fan_out = nn.init._calculate_fan_in_and_fan_out(weight)
        std = math.sqrt(2.0 / float(fan_in + fan_out))
        weight.data = weight.data.sign() * std

    elif args.weight_init == "standard":
        nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
    else:
        raise ValueError(f"{args.init} is not an initialization option!")

    return weight

def _init_score(args, scores):
    set_seed(args.score_seed)
    if args.score_init == "kaiming_normal_score_gt0":
        nn.init.kaiming_normal_(
            scores, mode=args.mode, nonlinearity=args.nonlinearity
        )
        scores = nn.Parameter(torch.abs(scores))
    elif args.score_init == "kaiming_uniform_score_gt0":
        nn.init.kaiming_uniform_(
            scores, mode=args.mode, nonlinearity=args.nonlinearity
        )
        scores = nn.Parameter(torch.abs(scores))
    elif args.score_init == "kaiming_normal_score_lt0":
        nn.init.kaiming_normal_(
            scores, mode=args.mode, nonlinearity=args.nonlinearity
        )
        scores = nn.Parameter(-torch.abs(scores))
    elif args.score_init == "kaiming_uniform_score_lt0":
        nn.init.kaiming_uniform_(
            scores, mode=args.mode, nonlinearity=args.nonlinearity
        )
        scores = nn.Parameter(-torch.abs(scores))

    elif args.score_init == "uniform_x_lt0":
        nn.init.uniform_(
            scores
        )
        scores = nn.Parameter(scores - args.x_percent_init)
    else:
        print("Using default score initialization")

    return scores