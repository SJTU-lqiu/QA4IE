#!/bin/bash

python3 train.py \
    --name QA4IE \
    --model QA \
    --data_dir data/span \
    --batch_size 20 \
    --eval_batch_size 40 \
    --num_epochs 12 \
    --lr 1e-3 \
    --dropout 0.2 \
    --eval_period 1000000 \
    --sent_size_th 400 \
    --mode test