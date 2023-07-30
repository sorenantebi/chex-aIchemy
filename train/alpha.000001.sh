#!/bin/bash
export CUDA_VISIBLE_DEVICES=0
source /vol/bitbucket/sea22/myvenv/bin/activate

export PYTHONPATH=/vol/bitbucket/sea22/MSC_PROJECT/main/chex-aIchemy
nvidia-smi


python3 /vol/bitbucket/sea22/MSC_PROJECT/main/chex-aIchemy/train/main.py \
    --model_name pc \
    --epochs 40 \
    --confusion race-negation \
    --fading_in_steps 20000 \
    --fading_in_range 800 \
    --alpha 0.000001