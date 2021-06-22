#!/bin/bash

python3 dump.py \
    --name QA4IE \
    --model QA \
    --batch_size 20 \
    --data_dir data/span \
    --eval_batch_size 40 \
    --sent_size_th 400 \
    --dropout 0.2 \
