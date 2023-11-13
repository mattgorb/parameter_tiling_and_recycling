
import torch


def create_signed_tile(tile_length):
    tile=2*torch.randint(0,2,(tile_length,))-1
    return tile

def fill_weight_signs(weight, tile):
    num_tiles=int(torch.ceil(torch.tensor(weight.numel()/tile.size(0))).item())
    tiled_tensor=tile.tile((num_tiles,))[:weight.numel()]
    tiled_weights=weight.flatten().abs()*tiled_tensor
    return torch.nn.Parameter(tiled_weights.reshape_as(weight), requires_grad=False)
