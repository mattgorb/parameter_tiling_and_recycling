
Please see `utils/layer_type.py` for the custom gradient engine containing the code for Tiled Bit Networks.  The file also has the implementations for convolutional 2d, convolutional 1d, fully-connected layers for training, as well as all modules used for inference.  

Triton kernels are in `utils/kernels.py`. 

## Training Modules: 

 **Custom Gradient Engine:**   [Straight-Through Estimator for Tiled Bit Networks](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/layer_type.py#L26)



[Convolutional 2D Layer](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/layer_type.py#L41)<br>
[Convolutional 1D Layer](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/layer_type.py#L108)<br>
[Fully-Connected Layer](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/layer_type.py#L182)

## Inference
**GPU Inference Kernels** GPU inference experiments are run in  `performance.py`.  A valid configuration file is needed. Please see the Appendix of the paper for details on what was implemented.

[Full Precision Tiled Inference Module](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/layer_type.py#L357)<br>
[Full Precision Tiled Inference Kernel](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/kernels.py#L532)<br>

[Binary Tiled Inference Module](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/layer_type.py#L573)<br>
[Binary Tiled Inference Kernel](https://github.com/mattgorb/tiled_bit_networks/blob/main/utils/kernels.py#L395)

**Microcontroller** The microcontroller implementation is in the folder `c_implementation`. We train our initial model in PyTorch using the numbered files in the `python` folder.  This trains an initial model, packs the weights, and converts them to `C` data types. Please see the `Makefile`, `models` folder, and `src` and `include` folders for more details. 



## CNN Models
Experimental settings and hyperparameter configurations can be found in the `configs` folder for ResNet and VGG models.  Experiments are executed from the `main.py` and `main_parallel.py` files. 
### Example run: 
```
nohup python -u main.py --config configs/cifar10/resnet18-kn-tiled-4.yaml --gpu=0 --data=~/  --epochs=300 > output_file.txt  2>&1 & 
```


## CNN Models
Experimental settings and hyperparameter configurations can be found in the `configs` folder for ResNet and VGG models.  Experiments are executed from the `main.py` and `main_parallel.py` files. 
#### Example run: 
```
nohup python -u main.py --config configs/cifar10/resnet18-kn-tiled-4.yaml --gpu=0 --data=~/  --epochs=300 > output_file.txt  2>&1 & 
```

## Transformer Models
Transformer models are in the `vision_transformers_cifar10` folder. We modify code in the folder to additionally run the ImageNet dataset

## PointNet Models
PointNet models are in the `pointnet` folder and run from the `train_classification.py`, `train_partseg.py`, and `train_semseg.py`. 