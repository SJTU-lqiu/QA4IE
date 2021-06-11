from collections import defaultdict
import os
import argparse


def get_args():
    parser = argparse.ArgumentParser()
    
    # names and directories
    parser.add_argument("--name", type=str, default="qa4ie")
    parser.add_argument("--model", type=str, default="QA")
    parser.add_argument("--seed", type=int, default=442)
    parser.add_argument("--out_base_dir", type=str, default="out")
    parser.add_argument("--orig_data_dir", type=str, default='data/orig_data')
    parser.add_argument("--data_dir", type=str, default="data/span")

    # prepro
    parser.add_argument("--glove_path", type=str, default="data/glove/glove.6B.100d.txt")
    parser.add_argument("--data_type", type=str, default="span")

    # device
    parser.add_argument("--num_gpus", type=int, default=1)
    parser.add_argument("--num_cpus", type=int, default=32)

    # training & evaluation
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--eval_batch_size", type=int, default=100)
    parser.add_argument("--max_val_batches", type=int, default=100)
    parser.add_argument("--num_steps", type=int, default=0)
    parser.add_argument("--num_epochs", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--log_period", type=int, default=100)
    parser.add_argument("--eval_period", type=int, default=10000)

    # model
    parser.add_argument("--hidden_size", type=int, default=100)
    parser.add_argument("--attn_proj_size", type=int, default=100)
    parser.add_argument("--char_out_size", type=int, default=100)
    parser.add_argument("--word_emb_size", type=int, default=100)
    parser.add_argument("--char_emb_size", type=int, default=8)
    parser.add_argument("--out_channels", type=str, default="100")
    parser.add_argument("--kernel_sizes", type=str, default="5")
    parser.add_argument("--ss_feature_size", type=int, default=40)
    #TODO finetune glove
    parser.add_argument("--num_highway_layers", type=int, default=2)
    parser.add_argument("--num_modeling_layers", type=int, default=1)
    # EMA Var
    parser.add_argument("--max_decode_length", type=int, default=64)

    # data
    parser.add_argument("--word_count_th", type=int, default=100)
    parser.add_argument("--char_count_th", type=int, default=500)
    parser.add_argument("--sent_size_th", type=int, default=400)
    parser.add_argument("--ques_size_th", type=int, default=30)
    parser.add_argument("--word_size_th", type=int, default=16)
    parser.add_argument("--lower_word", action="store_true")
    parser.add_argument("--ss_filter", type=str, default='single')

    config = parser.parse_args()
    config.out_dir = os.path.join(config.out_base_dir, config.name, config.model)
    if not os.path.exists(config.out_dir):
        os.makedirs(config.out_dir)
    # config.data_dir = os.path.join(config.data_dir, config.data_type)
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)

    return config
