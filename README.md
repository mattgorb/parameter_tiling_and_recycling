### Example run: 

```
nohup python -u main.py --config configs/cifar10/conv6-kn-biprop-tiled-params-randtile.yaml --prune_rate=0.5 --gpu=6 --data=~/  --epochs=4000 > results/98.txt  2>&1 & 
```

### The cool parts of the code are here: 

[Full Bit Tiling](https://github.com/mattgorb/parameter_tiling_and_recycling/blob/main/utils/conv_type.py#L27)

[Rerandomize and Retile](https://github.com/mattgorb/parameter_tiling_and_recycling/blob/main/utils/conv_type.py#L120)

[Recycle](https://github.com/mattgorb/parameter_tiling_and_recycling/blob/main/utils/conv_type.py#L97)


