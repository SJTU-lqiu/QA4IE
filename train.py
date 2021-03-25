from data_utils.data import QADataset, get_qa_filter
import torch
import os
import wandb
import math
import random
import numpy as np
import ujson as json
from tqdm import tqdm
from torch.optim import Adam, Adadelta
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader
from config import get_args
from data_utils import prepro_orig, prepro_vocab, SSDataset, get_ss_collate_fn, get_ss_filter, get_qa_collate_fn, get_qa_filter
from models import QA4IESS, QA4IEQA, QA4IEAT
from optim_utils import *


def set_seed(config):
    seed = config.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def trainSS(config):
    ss_filter = get_ss_filter(config)
    train_data = json.load(open(os.path.join(config.data_dir, 'train.json')))
    train_data = SSDataset(train_data, ss_filter)
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, collate_fn=get_ss_collate_fn(config))
    dev_data = json.load(open(os.path.join(config.data_dir, 'dev.json')))
    dev_data = SSDataset(dev_data, ss_filter)
    dev_loader = DataLoader(dev_data, config.eval_batch_size, shuffle=True, collate_fn=get_ss_collate_fn(config))

    model = QA4IESS(config).to(0)
    # optimizer = Adam(model.parameters(), lr=config.lr)
    optimizer = Adadelta(model.parameters(), 0.5)
    num_steps = int(math.ceil(len(train_data) / config.batch_size)) * config.num_epochs
    scheduler = LambdaLR(optimizer, lr_lambda=get_warmup_linear_decay(0, num_steps))

    step = 0
    log_loss = 0
    dev_score = 0
    for ei in range(config.num_epochs):
        for batch in tqdm(train_loader, ncols=100):
            loss, logits = model(*batch)
            scores = logits.softmax(dim=-1)[:, 1]
            loss.backward()
            optimizer.step()
            log_loss += float(loss)
            step += 1
            scheduler.step()
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=step)
            wandb.log({'min': float(scores.min()), 'max': float(scores.max())}, step=step)

            if step % config.log_period == 0:
                wandb.log({'loss': log_loss / config.log_period}, step=step)
                log_loss = 0

            if step % config.eval_period == 0:
                score, eval_dict = evalSS(model, dev_data, dev_loader)
                wandb.log(eval_dict, step=step)
                if score > dev_score:
                    torch.save(model.state_dict(), os.path.join(config.out_dir, 'model.pt'))

        score, eval_dict = evalSS(model, dev_data, dev_loader)
        wandb.log(eval_dict, step=step)
        if score > dev_score:
            torch.save(model.state_dict(), os.path.join(config.out_dir, 'model.pt'))


def trainQA(config):
    qa_filter = get_qa_filter(config)
    collate_fn = get_qa_collate_fn(config)
    train_data = json.load(open(os.path.join(config.data_dir, 'train.json')))
    train_data = QADataset(train_data, None, qa_filter)
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_data = json.load(open(os.path.join(config.data_dir, 'dev.json')))
    dev_data = QADataset(dev_data, None, qa_filter)
    dev_loader = DataLoader(dev_data, config.eval_batch_size, shuffle=True, collate_fn=collate_fn)

    model = QA4IEQA(config).to(0)
    optimizer = Adam(model.parameters(), lr=config.lr)
    num_steps = int(math.ceil(len(train_data) / config.batch_size)) * config.num_epochs
    scheduler = LambdaLR(optimizer, lr_lambda=get_warmup_linear_decay(0, num_steps))
    step = 0
    log_loss = 0
    dev_score = -1
    for ei in range(config.num_epochs):
        for batch in tqdm(train_loader, ncols=100):
            cx, cq, x, q, x_masks, q_masks, ys, y_masks, qaids = batch
            loss, logits = model(cx, cq, x, q, x_masks, q_masks, ys, y_masks)
            scores = logits.softmax(dim=-1)
            loss.backward()
            optimizer.step()
            log_loss += float(loss)
            step += 1
            scheduler.step()
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=step)
            wandb.log({'min': float(scores.min()), 'max': float(scores.max())}, step=step)

            if step % config.log_period == 0:
                wandb.log({'loss': log_loss / config.log_period}, step=step)
                log_loss = 0

            if step % config.eval_period == 0:
                score, eval_dict = evalQA(model, dev_loader, config.max_val_batches)
                wandb.log(eval_dict, step=step)
                if score > dev_score:
                    torch.save(model.state_dict(), os.path.join(config.out_dir, 'model.pt'))


@torch.no_grad()
def evalSS(model, data, loader):
    model.eval()
    scores = []
    ss_dict = {}
    for ai, article in enumerate(data.data):
        ss_dict[ai] = {}
        for qi in range(len(article['qas'])):
            ss_dict[ai][qi] = [0.] * len(article['x'])

    for batch in tqdm(loader, ncols=100):
        logits = model(*(batch[:-1]))
        score = logits.softmax(dim=1)[:, 1].cpu().numpy().tolist()
        # score = logits.sigmoid().cpu().numpy().tolist()
        scores.extend(score)
    
    assert len(data.features) == len(scores)
    for f, score in zip(data.features, scores):
        ai, qi, si = f['ridx']
        ss_dict[ai][qi][si] = score

    cover, total = 0, 0
    for ai, article in enumerate(data.data):
        sents_len = [len(sent) for sent in article['x']]
        for qi, qa in enumerate(article['qas']):
            answers = qa['a']

            scores = ss_dict[ai][qi]
            sents_indices = np.argsort(scores)[::-1]
            selected_indices, sum_len = [], 0
            for si in sents_indices:
                if sum_len + sents_len[si] <= 400:
                    selected_indices.append(si)
                    sum_len += sents_len[si]
                else:
                    break
            selected_indices = sorted(selected_indices)

            for a in answers:
                complete = True
                for si, ti in a['y']:
                    if si not in selected_indices:
                        complete = False
                        break
                if complete:
                    cover += 1
                    break
            
            total += 1

    cover_score = float(cover) / total
    model.train()
    return cover_score, {'cover': cover_score}


@torch.no_grad()
def evalQA(model, loader, max_batches):
    model.eval()
    dev_loss = 0
    true, total = 0, 0.
    for batch, bi in tqdm(zip(loader, range(max_batches)), ncols=100):
        cx, cq, x, q, x_masks, q_masks, ys, y_masks, qaids = batch
        loss, logits = model(cx, cq, x, q, x_masks, q_masks, ys, y_masks)
        # preds = logits.softmax(dim=-1)[:, 1] > 0.5
        dev_loss += loss.item()
        # true += float((preds == batch[-1].bool()).sum())
        # total += len(preds)
    model.train()
    dev_loss = dev_loss / min(len(loader), max_batches)
    return dev_loss, {'dev_loss': dev_loss, 'acc': 0.}

if __name__ == "__main__":
    config = get_args()
    set_seed(config)
    prepro_orig(config)
    prepro_vocab(config)
    if config.model == 'SS':
        wandb.init(project="QA4IE-J", name=f'{config.name}-{config.model}')
        # wandb.config.update(config)
        trainSS(config)
    elif config.model == 'QA':
        wandb.init(project="QA4IE-J", name=f'{config.name}-{config.model}')
        # wandb.config.update(config)
        trainQA(config)
