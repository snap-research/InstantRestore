#!/bin/bash -i
conda activate py310;
cd /nfs/hzhang9/PFR/face-replace-hzhang9;
CUDA_VISIBLE_DEVICES=0,1,2,3 accelerate launch scripts/train.py --config_path config_files/train_hzhang9.yaml