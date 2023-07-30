#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
source /vol/bitbucket/sea22/myvenv/bin/activate

export PYTHONPATH=/vol/bitbucket/sea22/MSC_PROJECT/main/chex-aIchemy
nvidia-smi


python3 /vol/bitbucket/sea22/MSC_PROJECT/main/chex-aIchemy/train/main.py \
    --model_name pc \
    --epochs 40 \
    --label_noise True \
    --confusion race-negation \
   
     
