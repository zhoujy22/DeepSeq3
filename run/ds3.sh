#!/bin/bash
NUM_PROC=1
GPUS=1

cd ./src
python3 main.py \
 --exp_id stage1_ds3 \
 --data_dir ../data/trainV2 \
 --reg_loss sl1 --cls_loss bce \
 --arch mlpgnn \
 --randomwalk 10 \
 --aggr_function tfmlp\
 --prob_weight 3 --rc_weight 1 --func_weight 2 \
 --seq_weight 0 --trans_weight 4 \
 --num_rounds 2 \
 --batch_size 16\
 --num_epochs 100 \
 --gpus $GPUS \

