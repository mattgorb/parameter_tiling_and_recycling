#!/bin/bash

#SBATCH --job-name="tiling"	 # job name
#SBATCH --partition=peregrine-gpu    		 # partition to which job should be submitted
#SBATCH --qos=gpu_long			  		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --ntasks=1                		 # total number of tasks across all nodes
#SBATCH --cpus-per-task=20         		 # cpu-cores per task
#SBATCH --mem=100g                  		 # total memory per node
#SBATCH --gres=gpu:a100-sxm4-80gb:2              # Request 1 GPU (A100 40GB)
#SBATCH --time=5-0:00:00 				 #  wall time
#SBATCH --nodelist=peregrine0

module purge
module load python/bundle-3.9

#TORCH_DISTRIBUTED_DEBUG=DETAIL 
#NCCL_MAX_ASYNC_OPS=8
#NCCL_P2P_DISABLE=1 
#NCCL_DEBUG=INFO
#TORCH_DISTRIBUTED_DEBUG=DETAIL NCCL_DEBUG=INFO NCCL_MAX_ASYNC_OPS=6 
#python -u main_parallel_peregrine.py --config configs/imagenet/resnet50-biprop-tiled-params-iterand.yaml --gpu=0 --multigpu=0 --resume /s/babbage/b/nobackup/nblancha/public-datasets/subnetworks/runs/resnet50-biprop-tiled-params-iterand/biprop/prune_rate=-1/49/checkpoints/model_best.pth

python -u ../main_parallel_swint_peregrine.py --config ../configs/imagenet/swint-tiled-full-150000-4.yaml --gpu=0 --multigpu=0,1 --batch-size=256  --global_compression_factor=2 