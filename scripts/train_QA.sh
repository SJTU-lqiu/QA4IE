#!/bin/bash

python3 train.py \
    --name debug \
    --model QA \
    --data_dir data/spanS \
    --batch_size 20 \
    --eval_batch_size 40 \
    --num_epochs 12 \
    --lr 1e-3 \
    --dropout 0.2 \
    --eval_period 1000 \
    --sent_size_th 400
