#!/bin/bash

source /rds/general/user/sea22/home/Job_Scripts/myenv/bin/activate
export PYTHONPATH=/rds/general/user/sea22/home/main/chex-aIchemy
python -c "import os;pythonpath = os.getenv('PYTHONPATH','');print(os.getenv('PYTHONPATH'))"
python3 /rds/general/user/sea22/home/main/chex-aIchemy/train/main.py --model_name imagenet