from random import shuffle
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
from data_utils import prepro_orig, prepro_vocab, SSDataset, get_ss_collate_fn, get_ss_filter, \
    QADataset, get_qa_collate_fn, get_qa_filter, ATDataset, get_at_collate_fn, get_at_filter
from models import QA4IESS, QA4IEQA, QA4IEAT
from optim_utils import *
import squad_utils


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
    dev_loader = DataLoader(dev_data, config.eval_batch_size, shuffle=False, collate_fn=get_ss_collate_fn(config))

    model = QA4IESS(config).to(0)
    optimizer = Adam(model.parameters(), lr=config.lr)
    num_steps = int(math.ceil(len(train_data) / config.batch_size)) * config.num_epochs
    scheduler = LambdaLR(optimizer, lr_lambda=get_warmup_linear_decay(0, num_steps))

    step = 0
    log_loss = 0
    dev_score = 0
    for ei in range(config.num_epochs):
        for batch in tqdm(train_loader, ncols=100):
            optimizer.zero_grad()
            loss, logits = model(*batch)
            loss.backward()
            optimizer.step()
            log_loss += float(loss)
            step += 1
            scheduler.step()
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=step)

            if step % config.log_period == 0:
                wandb.log({'loss': log_loss / config.log_period}, step=step)
                log_loss = 0

            if step % config.eval_period == 0:
                score, eval_dict = evalSS(model, dev_data, dev_loader)
                wandb.log(eval_dict, step=step)
                if score > dev_score:
                    torch.save(model.state_dict(), os.path.join(config.out_dir, 'model.pt'))
                    dev_score = score

        score, eval_dict = evalSS(model, dev_data, dev_loader)
        wandb.log(eval_dict, step=step)
        if score > dev_score:
            torch.save(model.state_dict(), os.path.join(config.out_dir, 'model.pt'))
            dev_score = score


def trainQA(config):
    qa_filter = get_qa_filter(config)
    collate_fn = get_qa_collate_fn(config)
    train_data = json.load(open(os.path.join(config.data_dir, 'train.json')))
    train_ss = torch.load(os.path.join(config.out_dir, '../SS/train.SS.pt'))
    train_data = QADataset(train_data, train_ss, qa_filter)
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_ss = torch.load(os.path.join(config.out_dir, '../SS/dev.SS.pt'))
    dev_data = json.load(open(os.path.join(config.data_dir, 'dev.json')))
    dev_data = QADataset(dev_data, dev_ss, qa_filter)
    dev_loader = DataLoader(dev_data, config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    model = QA4IEQA(config).to(0)
    optimizer = Adam(model.parameters(), lr=config.lr)
    num_steps = int(math.ceil(len(train_data) / config.batch_size)) * config.num_epochs
    scheduler = LambdaLR(optimizer, lr_lambda=get_warmup_linear_decay(0, num_steps))
    step = 0
    log_loss = 0
    dev_score = -1
    for ei in range(config.num_epochs):
        for batch in tqdm(train_loader, ncols=100):
            optimizer.zero_grad()
            cx, cq, x, q, x_masks, q_masks, ys, y_masks, qaids = batch
            loss, logits = model(cx, cq, x, q, x_masks, q_masks, ys, y_masks)
            loss.backward()
            optimizer.step()
            log_loss += float(loss)
            step += 1
            scheduler.step()
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=step)

            if step % config.log_period == 0:
                wandb.log({'loss': log_loss / config.log_period}, step=step)
                log_loss = 0

            if step % config.eval_period == 0:
                score, eval_dict = evalQA(model, dev_data, dev_loader)
                wandb.log(eval_dict, step=step)
                if score > dev_score:
                    torch.save(model.state_dict(), os.path.join(config.out_dir, 'model.pt'))
                    dev_score = score
        
        score, eval_dict = evalQA(model, dev_data, dev_loader)
        wandb.log(eval_dict, step=step)
        if score > dev_score:
            torch.save(model.state_dict(), os.path.join(config.out_dir, 'model.pt'))
            dev_score = score


def trainAT(config):
    at_filter = get_at_filter(config)
    collate_fn = get_at_collate_fn(config)
    train_data = json.load(open(os.path.join(config.data_dir, 'ietrain.json')))
    train_ss = torch.load(os.path.join(config.out_dir, '../SS/ietrain.SS.pt'))
    train_qa = torch.load(os.path.join(config.out_dir, '../QA/ietrain.QA.pt'))
    train_data = ATDataset(train_data, train_ss, train_qa, at_filter)
    train_loader = DataLoader(train_data, config.batch_size, shuffle=True, collate_fn=collate_fn)
    dev_ss = torch.load(os.path.join(config.out_dir, '../SS/iedev.SS.pt'))
    dev_qa = torch.load(os.path.join(config.out_dir, '../QA/iedev.QA.pt'))
    dev_data = json.load(open(os.path.join(config.data_dir, 'iedev.json')))
    dev_data = ATDataset(dev_data, dev_ss, dev_qa, at_filter)
    dev_loader = DataLoader(dev_data, config.eval_batch_size, shuffle=False, collate_fn=collate_fn)

    model = QA4IEAT(config).to(0)
    optimizer = Adam(model.parameters(), lr=config.lr)
    num_steps = int(math.ceil(len(train_data) / config.batch_size)) * config.num_epochs
    scheduler = LambdaLR(optimizer, lr_lambda=get_warmup_linear_decay(0, num_steps))
    step = 0
    log_loss = 0
    dev_score = -1
    print(evalAT(model, dev_data, dev_loader))
    for ei in range(config.num_epochs):
        for batch in tqdm(train_loader, ncols=100):
            optimizer.zero_grad()
            cx, cq, x, q, x_masks, q_masks, scores, ys, qaids = batch
            loss, logits = model(cx, cq, x, q, x_masks, q_masks, scores, ys)
            loss.backward()
            optimizer.step()
            log_loss += float(loss)
            step += 1
            scheduler.step()
            wandb.log({'lr': scheduler.get_last_lr()[0]}, step=step)

            if step % config.log_period == 0:
                wandb.log({'loss': log_loss / config.log_period}, step=step)
                log_loss = 0

            if step % config.eval_period == 0:
                score, eval_dict = evalAT(model, dev_data, dev_loader)
                print(eval_dict)
                wandb.log(eval_dict, step=step)
                if score > dev_score:
                    torch.save(model.state_dict(), os.path.join(config.out_dir, 'model.pt'))
                    dev_score = score
        
        score, eval_dict = evalAT(model, dev_data, dev_loader)
        wandb.log(eval_dict, step=step)
        if score > dev_score:
            torch.save(model.state_dict(), os.path.join(config.out_dir, 'model.pt'))
            dev_score = score


@torch.no_grad()
def evalSS(model, data, loader):
    model.eval()
    scores = []
    labels = []
    ss_dict = {}

    for ai, article in enumerate(data.data):
        ss_dict[ai] = {}
        for qi in range(len(article['qas'])):
            ss_dict[ai][qi] = [0.] * len(article['x'])

    for batch in tqdm(loader, ncols=100):
        logits = model(*(batch[:-1]))
        score = logits.sigmoid()
        scores.append(score)
        labels.append(batch[-1])

    scores = torch.cat(scores)
    labels = torch.cat(labels)
    sorted_scores, sorted_indices = scores.sort(descending=True)
    sorted_labels = labels[sorted_indices]
    true = sorted_labels.cumsum(dim=0)
    positive = torch.arange(len(true)).to(true) + 1
    prec = true / positive
    rec = true / labels.sum()
    f1s = 2 * prec * rec / (prec + rec)
    f1s[f1s.isnan()] = 0.
    maxf1, maxi = f1s.max(dim=0)
    prec, rec = prec[maxi], rec[maxi]
    theta = sorted_scores[maxi]

    assert len(data.features) == len(scores)

    for f, score in zip(data.features, scores):
        ai, qi, si = f['ridx']
        ss_dict[ai][qi][si] = score.item()

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
    return cover_score, {'cover': cover_score, 'prec': prec, 'rec': rec, 'f1': maxf1, 'theta': theta}


@torch.no_grad()
def evalQA(model, dataset, loader):
    model.eval()
    
    output = {}
    for batch in tqdm(loader, ncols=100):
        cx, cq, x, q, x_masks, q_masks, ys, y_masks, qaids = batch
        xlens = x_masks.sum(dim=1).cpu().numpy().tolist()
        logits = model(cx, cq, x, q, x_masks, q_masks)
        preds = logits.argmax(dim=-1).cpu().numpy().tolist()

        for qaid, pred, xlen in zip(qaids, preds, xlens):
            end = pred.index(xlen) if xlen in pred else len(pred)
            output[qaid] = pred[:end]

    for f in dataset.features:
        if f['id'] not in output:
            continue
        sents_indices = f['sents_indices']
        ai, qi = f['ridx']
        x = [word for si in sents_indices for word in dataset.data[ai]['x'][si]]
        pred = output[f['id']]
        output[f['id']] = ' '.join([x[i] for i in pred])

    em, f1 = squad_utils.evaluate(dataset.data, output)
    model.train()
    return em, {'em': em, 'f1': f1}


@torch.no_grad()
def evalAT(model, dataset, loader):
    model.eval()
    scores = []
    labels = []
    for batch in tqdm(loader, ncols=100):
        cx, cq, x, q, x_masks, q_masks, ss_scores, ys, qaids = batch
        logits = model(cx, cq, x, q, x_masks, q_masks, ss_scores)
        preds = logits.sigmoid()
        scores.append(preds)
        labels.append(ys)

    scores = torch.cat(scores)
    labels = torch.cat(labels)
    sorted_scores, sorted_indices = scores.sort(descending=True)
    sorted_labels = labels[sorted_indices]
    true = sorted_labels.cumsum(dim=0)
    positive = torch.arange(len(true)).to(true) + 1
    prec = true / positive
    rec = true / labels.sum()
    f1s = 2 * prec * rec / (prec + rec)
    f1s[f1s.isnan()] = 0.
    maxf1, maxi = f1s.max(dim=0)
    prec, rec = prec[maxi], rec[maxi]
    theta = sorted_scores[maxi]

    model.train()
    return maxf1, {'prec': prec * 100., 'rec': rec * 100., 'f1': maxf1 * 100., 'theta': theta}


if __name__ == "__main__":
    config = get_args()
    set_seed(config)
    prepro_orig(config)
    prepro_vocab(config)
    if config.mode == 'train':
        if config.model == 'SS':
            wandb.init(project="QA4IE-J", name=f'{config.name}-{config.model}')
            # wandb.config.update(config)
            trainSS(config)
        elif config.model == 'QA':
            wandb.init(project="QA4IE-J", name=f'{config.name}-{config.model}')
            # wandb.config.update(config)
            trainQA(config)
        elif config.model == 'AT':
            wandb.init(project="QA4IE-J", name=f'{config.name}-{config.model}')
            # wandb.config.update(config)
            trainAT(config)
    elif config.mode == 'test':
        if config.model == 'SS':
            ss_filter = get_ss_filter(config)
            data = json.load(open(os.path.join(config.data_dir, 'test.json')))
            data = SSDataset(data, ss_filter)
            loader = DataLoader(data, config.eval_batch_size, shuffle=False, collate_fn=get_ss_collate_fn(config))
            model = QA4IESS(config).to(0)
            model.load_state_dict(torch.load(os.path.join(config.out_dir, 'model.pt')))
            print(evalSS(model, data, loader))
        elif config.model == 'QA':
            qa_filter = get_qa_filter(config)
            collate_fn = get_qa_collate_fn(config)
            ss = torch.load(os.path.join(config.out_dir, '../SS/test.SS.pt'))
            data = json.load(open(os.path.join(config.data_dir, 'test.json')))
            data = QADataset(data, ss, qa_filter)
            loader = DataLoader(data, config.eval_batch_size, shuffle=False, collate_fn=collate_fn)
            model = QA4IEQA(config).to(0)
            model.load_state_dict(torch.load(os.path.join(config.out_dir, 'model.pt')))
            print(evalQA(model, data, loader))
        elif config.model == 'AT':
            at_filter = get_at_filter(config)
            collate_fn = get_at_collate_fn(config)
            ss = torch.load(os.path.join(config.out_dir, '../SS/ietest.SS.pt'))
            qa = torch.load(os.path.join(config.out_dir, '../QA/ietest.QA.pt'))
            data = json.load(open(os.path.join(config.data_dir, 'ietest.json')))
            data = ATDataset(data, ss, qa, at_filter)
            loader = DataLoader(data, config.eval_batch_size, shuffle=False, collate_fn=collate_fn)
            model = QA4IEAT(config).to(0)
            model.load_state_dict(torch.load(os.path.join(config.out_dir, 'model.pt')))
            print(evalAT(model, data, loader))
