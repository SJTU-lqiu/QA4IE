#!/bin/bash

python3 train.py \
    --name QA4IE \
    --model SS \
    --data_dir data/span \
    --batch_size 256 \
    --eval_batch_size 1024 \
    --num_epochs 10 \
    --lr 1e-3 \
    --dropout 0.2 \
    --eval_period 1000000 \
    --sent_size_th 128
