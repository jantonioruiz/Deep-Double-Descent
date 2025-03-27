#!/bin/bash

#SBATCH --job-name DDD                   
#SBATCH --gres=gpu:1
#SBATCH --mem 20g
#SBATCH --partition dios 
#SBATCH -w atenea  

export PATH="/opt/anaconda/anaconda3/bin:$PATH"
export PATH="/opt/anaconda/bin:$PATH"
eval "$(conda shell.bash hook)"
conda activate /mnt/homeGPU/jaruiz/testpy310
export TFHUB_CACHE_DIR=.

python main.py --dataset MNIST --model TwoLayerNN --output_train train.txt --output_test test.txt --units 1