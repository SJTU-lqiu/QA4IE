import torch
import json
from torch.utils.data import DataLoader
import os
import squad_utils
from tqdm import tqdm
from models import QA4IESS, QA4IEQA, QA4IEAT
from config import get_args
from data_utils import get_qa_filter, get_qa_collate_fn, QADataset, prepro_orig, prepro_vocab


config = get_args()
prepro_orig(config)
prepro_vocab(config)
model = QA4IEQA(config).to(0)
model.load_state_dict(torch.load('out/QA4IE/QA/model.pt'))
model.eval()
data_filter = get_qa_filter(config)
collate_fn = get_qa_collate_fn(config)
ss_feature = torch.load(os.path.join(config.out_dir, f'../SS/ietest.SS.pt'))
path = os.path.join(config.data_dir, f'ietest.json')
dataset = QADataset(json.load(open(path)), ss_feature, data_filter)
loader = DataLoader(dataset, config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

with torch.no_grad():
    output = {}
    for batch in tqdm(loader, ncols=100):
        cx, cq, x, q, x_masks, q_masks, ys, y_masks, qaids = batch
        xlens = x_masks.sum(dim=1).cpu().numpy().tolist()
        logits = model(cx, cq, x, q, x_masks, q_masks)
        qa_scores = logits.softmax(dim=-1)
        preds = qa_scores.argmax(dim=-1).cpu().numpy().tolist()\

        for qaid, pred, xlen, qa_score in zip(qaids, preds, xlens, qa_scores):
            end = pred.index(xlen) if xlen in pred else len(pred)
            output[qaid] = (pred[:end], qa_score[:end].prod())

    for f in dataset.features:
        if f['id'] not in output:
            continue
        sents_indices = f['sents_indices']
        ai, qi = f['ridx']
        x = [word for si in sents_indices for word in dataset.data[ai]['x'][si]]
        pred, score = output[f['id']]
        output[f['id']] = ' '.join([x[i] for i in pred]), score

    maxf1, prec, rec = squad_utils.evaluate_pr(dataset.data, output)
    print(maxf1)