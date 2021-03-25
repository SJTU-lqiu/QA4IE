import torch
import os
import random
import numpy as np
import ujson as json
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import get_args
from data_utils import prepro_orig, prepro_vocab, SSDataset, get_ss_collate_fn, get_ss_filter
from models import QA4IESS, QA4IEQA, QA4IEAT
from optim_utils import *


def set_seed(config):
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def dumpSS_features(config, model, data, loader, tag='train'):
    indices = data.indices
    scores = []
    out_dict = {}
    for ai, article in enumerate(data.data):
        out_dict[ai] = {}
        for qi in range(len(article['qas'])):
            out_dict[ai][qi] = [0.] * len(article['x'])

    for batch in tqdm(loader, ncols=100):
        logits = model(*(batch[:-1]))
        # score = logits.sigmoid().cpu().numpy().tolist()
        score = logits.softmax(dim=-1)[:, 1].cpu().numpy().tolist()
        scores.extend(score)
        print(batch[-1].sum(), len(batch[-1]))
        print(max(scores), min(scores))
        exit(0)
    
    assert len(indices) == len(scores)
    for (ai, qi, si), score in zip(indices, scores):
        out_dict[ai][qi][si] = score

    torch.save(out_dict, os.path.join(config.out_dir, f'{tag}.SS.pt'))


def dumpSS(config):
    model = QA4IESS(config).to(0)
    model.load_state_dict(torch.load(os.path.join(config.out_dir, 'model.pt')))
    model.eval()

    ss_filter = get_ss_filter(config)
    ss_collate_fn = get_ss_collate_fn(config)

    # for tag in ['train', 'dev', 'test']:
    for tag in ['dev']:
        data = json.load(open(os.path.join(config.data_dir, f'{tag}.json')))
        data = SSDataset(data, ss_filter)
        loader = DataLoader(data, config.eval_batch_size, shuffle=True, collate_fn=ss_collate_fn)
        dumpSS_features(config, model, data, loader, tag)


if __name__ == "__main__":
    config = get_args()
    set_seed(config)
    prepro_orig(config)
    prepro_vocab(config)
    if config.model == 'SS':
        dumpSS(config)
