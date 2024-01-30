### Example run: 

```
nohup python -u main.py --config configs/cifar10/conv6-kn-biprop-tiled-params-randtile.yaml --prune_rate=0.5 --gpu=6 --data=~/  --epochs=4000 > results/98.txt  2>&1 & 
```

### The cool parts of the code are here: 
[Tiling Layers, Fully-Connected, Convolutional, Training, Inference](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/layer_type.py)

#### Training Kernels: 
[Straight-Through Estimator for Tiled Bit Networks](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/layer_type.py#L26)
 - 
[Rerandomize and Retile](https://github.com/mattgorb/parameter_tiling_and_recycling/blob/main/utils/layer_type.py#L120)

[Recycle](https://github.com/mattgorb/parameter_tiling_and_recycling/blob/main/utils/layer_type.py#L97)


