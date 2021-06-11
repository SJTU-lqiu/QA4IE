import torch
import os
import random
import numpy as np
import ujson as json
from tqdm import tqdm
from torch.utils.data import DataLoader
from config import get_args
from data_utils import prepro_orig, prepro_vocab, SSDataset, get_ss_collate_fn, get_ss_filter, QADataset, get_qa_collate_fn, get_qa_filter
from models import QA4IESS, QA4IEQA, QA4IEAT
from optim_utils import *
from squad_utils import exact_match_score, metric_max_over_ground_truths


def set_seed(config):
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def dumpSS_features(config, model, data, loader, tag='train'):
    scores = []
    out_dict = {}
    for ai, article in enumerate(data.data):
        out_dict[ai] = {}
        for qi in range(len(article['qas'])):
            out_dict[ai][qi] = [0.] * len(article['x'])

    for batch in tqdm(loader, ncols=100):
        logits = model(*(batch[:-1]))
        score = logits.sigmoid().cpu().numpy().tolist()
        scores.extend(score)
    
    assert len(data.features) == len(scores)
    for f, score in zip(data.features, scores):
        ai, qi, si = f['ridx']
        out_dict[ai][qi][si] = score

    torch.save(out_dict, os.path.join(config.out_dir, f'{tag}.SS.pt'))


@torch.no_grad()
def dumpQA_features(config, model, data, loader, split='train'):
    output = {}
    out_dict = {}
    for batch in tqdm(loader, ncols=100):
        cx, cq, x, q, x_masks, q_masks, ys, y_masks, qaids = batch
        xlens = x_masks.sum(dim=1).cpu().numpy().tolist()
        logits = model(cx, cq, x, q, x_masks, q_masks)
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()

        for qaid, pred, xlen in zip(qaids, preds, xlens):
            end = pred.index(xlen) if xlen in pred else len(pred)
            output[qaid] = pred[:end]

    assert len(data.features) == len(output)
    for f in data.features:
        sents_indices = f['sents_indices']
        ai, qi = f['ridx']
        x = [word for si in sents_indices for word in data.data[ai]['x'][si]]
        pred = [x[i] for i in output[f['id']]]
        gts = list(map(lambda x: x['text'], data.data[ai]['qas'][qi]['a']))
        pred_text = ' '.join(pred)
        em = metric_max_over_ground_truths(exact_match_score, pred_text, gts)

        out_dict[(ai, qi)] = (pred, em)
    
    torch.save(out_dict, os.path.join(config.out_dir, f'{split}.QA.pt'))


def dumpSS(config):
    model = QA4IESS(config).to(0)
    model.load_state_dict(torch.load(os.path.join(config.out_dir, 'model.pt')))
    model.eval()

    ss_filter = get_ss_filter(config)
    ss_collate_fn = get_ss_collate_fn(config)

    for tag in ['train', 'dev', 'test']:
        data = json.load(open(os.path.join(config.data_dir, f'{tag}.json')))
        data = SSDataset(data, ss_filter)
        loader = DataLoader(data, config.eval_batch_size, shuffle=False, collate_fn=ss_collate_fn)
        dumpSS_features(config, model, data, loader, tag)


def getSS_data(config, split='train'):
    path = os.path.join(config.data_dir, f'{split}.json')
    data_filter = get_ss_filter(config)
    collate_fn = get_ss_collate_fn(config)
    data = SSDataset(json.load(open(path)), data_filter)
    loader = DataLoader(data, config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    return data, loader


def getQA_data(config, split='train'):
    path = os.path.join(config.data_dir, f'{split}.json')
    data_filter = get_qa_filter(config)
    collate_fn = get_qa_collate_fn(config)
    ss_feature = torch.load(os.path.join(config.out_dir, f'../SS/{split}.SS.pt'))
    data = QADataset(json.load(open(path)), ss_feature, data_filter)
    loader = DataLoader(data, config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    return data, loader


def dump(config, model, data_fn, dump_fn):
    for split in ['ietest']:
    # for split in ['train', 'dev', 'test']:
        data, loader = data_fn(config, split)
        dump_fn(config, model, data, loader, split)


if __name__ == "__main__":
    config = get_args()
    set_seed(config)
    prepro_orig(config)
    prepro_vocab(config)
    if config.model == 'SS':
        model = QA4IESS(config).to(0)
        data_fn = getSS_data
        dump_fn = dumpSS_features
    elif config.model == 'QA':
        model = QA4IEQA(config).to(0)
        data_fn = getQA_data
        dump_fn = dumpQA_features
    else:
        raise NotImplementedError(f'dump function for {config.model} is not implemented')
    
    model.load_state_dict(torch.load(os.path.join(config.out_dir, 'model.pt')))
    model.eval()
    dump(config, model, data_fn, dump_fn)
