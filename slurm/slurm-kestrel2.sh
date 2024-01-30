#!/bin/bash

#SBATCH --job-name="tiled bit networks"	 # job name
#SBATCH --partition=kestrel-gpu    		 # partition to which job should be submitted
#SBATCH --qos=gpu_long			  		 # qos type
#SBATCH --nodes=1                 		 # node count
#SBATCH --ntasks=1                		 # total number of tasks across all nodes
#SBATCH --cpus-per-task=32         		 # cpu-cores per task
#SBATCH --mem=100G                  		 # total memory per node
#SBATCH --gres=gpu:3090:3               # Request 3 GPU (3090 24GB)
#SBATCH --time=10-00:00:00 				 #  node
#SBATCH --nodelist=kestrel1

module purge
module load python/bundle-3.9

python -u ../main_parallel_kestrel.py --config ../configs/imagenet/resnet18-tiled-full-150000-2.yaml --gpu=0 --multigpu=0,1,2 --batch-size=256
