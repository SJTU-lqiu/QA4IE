import torch
import json
from torch.utils.data import DataLoader
import os
import squad_utils
from tqdm import tqdm
from models import QA4IESS, QA4IEQA, QA4IEAT
from config import get_args
from data_utils import get_qa_filter, get_qa_collate_fn, QADataset, prepro_orig, prepro_vocab,\
    ATDataset, get_at_filter, get_at_collate_fn


config = get_args()
prepro_orig(config)
prepro_vocab(config)
model = QA4IEQA(config).to(0)
model.load_state_dict(torch.load('out/QA4IE/QA/model.pt'))
model.eval()
data_filter = get_qa_filter(config)
collate_fn = get_qa_collate_fn(config)
ss_feature = torch.load(os.path.join('out/QA4IE/SS/ietest.SS.pt'))
with open(os.path.join(config.data_dir, f'ietest.json')) as fp:
    iedata = json.load(fp)
dataset = QADataset(iedata, ss_feature, data_filter)
loader = DataLoader(dataset, config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

id2scores = {}
if config.scorer == 'AT':
    qa_feature = torch.load(os.path.join('out/QA4IE/QA/ietest.QA.pt'))
    at_dataset = ATDataset(iedata, ss_feature, qa_feature, get_at_filter(config))
    at_loader = DataLoader(at_dataset, config.eval_batch_size, shuffle=False, collate_fn=get_at_collate_fn(config))
    at_model = QA4IEAT(config).to(0)
    at_model.load_state_dict(torch.load('out/QA4IE/AT/model.pt'))
    at_model.eval()
    with torch.no_grad():
        for batch in tqdm(at_loader, ncols=100):
            cx, cq, x, q, x_masks, q_masks, ss_scores, ys, qaids = batch
            logits = at_model(cx, cq, x, q, x_masks, q_masks, ss_scores)
            preds = logits.sigmoid()
            for qaid, pred in zip(qaids, preds):
                id2scores[qaid] = pred

with torch.no_grad():
    output = {}
    for batch in tqdm(loader, ncols=100):
        cx, cq, x, q, x_masks, q_masks, ys, y_masks, qaids = batch
        xlens = x_masks.sum(dim=1).cpu().numpy().tolist()
        logits = model(cx, cq, x, q, x_masks, q_masks)
        qa_scores = logits.softmax(dim=-1)
        preds = qa_scores.argmax(dim=-1).cpu().numpy().tolist()
        qa_scores = qa_scores.max(dim=-1).values

        for qaid, pred, xlen, qa_score in zip(qaids, preds, xlens, qa_scores):
            end = pred.index(xlen) if xlen in pred else len(pred)
            if config.scorer == 'mean':
                output[qaid] = (pred[:end], qa_score[:end+1].mean())
            elif config.scorer == 'prod':
                output[qaid] = (pred[:end], qa_score[:end+1].prod())
            elif config.scorer == 'AT':
                output[qaid] = (pred[:end], id2scores[qaid])
            else:
                raise NotImplementedError(f'scorer [{config.scorer}] not implemented')

    for f in dataset.features:
        if f['id'] not in output:
            continue
        sents_indices = f['sents_indices']
        ai, qi = f['ridx']
        x = [word for si in sents_indices for word in dataset.data[ai]['x'][si]]
        pred, score = output[f['id']]
        output[f['id']] = ' '.join([x[i] for i in pred]), score

    maxf1, prec, rec = squad_utils.evaluate_pr(dataset.data, output)
    print(f'best f1: {maxf1}\tprec: {prec}\trec: {rec}')
