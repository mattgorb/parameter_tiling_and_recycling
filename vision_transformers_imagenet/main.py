import torch
from vit_pytorch import ViT

model = ViT(
    image_size = 256,
    patch_size = 32,
    num_classes = 1000,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048,
    dropout = 0.1,
    emb_dropout = 0.1
)

img = torch.randn(1, 3, 256, 256)

preds = model(img) # (1, 1000)

pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)


#print(model)

import torch
from vit_pytorch import SimpleViT

v = SimpleViT(
    image_size = 32,
    patch_size = 4,
    num_classes = 10,
    dim = 1024,
    depth = 6,
    heads = 16,
    mlp_dim = 2048
)
sys.exit()

img = torch.randn(128, 3, 32, 32)

preds = v(img) # (1, 1000)
print(preds.size())
pytorch_total_params = sum(p.numel() for p in v.parameters() if p.requires_grad)
print(pytorch_total_params)

from vit_pytorch.mobile_vit import MobileViT

mbvit_xs = MobileViT(
    image_size = (32, 32),
    dims = [96, 120, 144],
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    num_classes = 10
)

img = torch.randn(1, 3, 32, 32)

pred = mbvit_xs(img) # (1, 1000)

pytorch_total_params = sum(p.numel() for p in mbvit_xs.parameters() if p.requires_grad)
print(pytorch_total_params)









mbvit_xs = MobileViT(
    image_size = (32, 32),
    dims = [96, 120, 144],
    channels = [16, 32, 48, 48, 64, 64, 80, 80, 96, 96, 384],
    num_classes = 10
)

img = torch.randn(1, 3, 256, 256)

pred = mbvit_xs(img) # (1, 1000)

pytorch_total_params = sum(p.numel() for p in mbvit_xs.parameters() if p.requires_grad)
print(pytorch_total_params)