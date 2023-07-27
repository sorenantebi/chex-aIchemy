#!/bin/bash

export CUDA_VISIBLE_DEVICES=1
source /vol/bitbucket/sea22/myvenv/bin/activate

export PYTHONPATH=/homes/sea22/MSC_PROJECT/main/chex-aIchemy
nvidia-smi
for value in {2..6} 
do
    alpha=$(echo "scale=10; 10^-($value) " | bc)
    echo  $alpha
done