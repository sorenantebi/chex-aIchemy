#!/bin/bash

export CUDA_VISIBLE_DEVICES=0
source /vol/bitbucket/sea22/myvenv/bin/activate

export PYTHONPATH=/vol/bitbucket/sea22/MSC_PROJECT/main/chex-aIchemy
nvidia-smi


python3 /vol/bitbucket/sea22/MSC_PROJECT/main/chex-aIchemy/train/store_args.py \
    --json_file /vol/bitbucket/sea22/MSC_PROJECT/main/results/NOEN/hparam_config/hparams.json 
