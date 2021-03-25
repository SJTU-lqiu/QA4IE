#!/bin/bash

python3 dump.py \
    --name QA4IE \
    --model SS \
    --batch_size 256 \
    --eval_batch_size 1024 \
    --ss_filter naive \
    --sent_size_th 128
