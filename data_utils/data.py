import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader


class QADataset(Dataset):
    def __init__(self, data, ss_feature=None, filter=None):
        super(QADataset, self).__init__()
        print('building QA dataset')
        self.data = data
        self.xs, self.features = filter(data, ss_feature)
        print(f"{len(self.features)}/{sum(len(a['qas']) for a in self.data)} have at least one answer")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        ai, qi = self.features[idx]['ridx']
        sents_indices = self.features[idx]['sents_indices']
        x, cx = list(zip(*[self.xs[ai][si] for si in sents_indices]))
        x, cx = torch.cat(x, dim=0), torch.cat(cx, dim=0)
        q, cq = self.features[idx]['q']
        y = random.choice(self.features[idx]['y'])
        qaid = self.features[idx]['id']

        return cx, x, cq, q, y, qaid


def get_qa_collate_fn(config):
    a_len = config.max_decode_length
    def qa_collate_fn(batch):
        
        cx = pad_sequence([f[0] for f in batch]).to(0).transpose(0, 1)
        x = pad_sequence([f[1] for f in batch]).to(0).transpose(0, 1)

        cq = pad_sequence([f[2] for f in batch]).to(0).transpose(0, 1)
        q = pad_sequence([f[3] for f in batch]).to(0).transpose(0, 1)

        x_mask, q_mask = x > 0, q > 0

        qaids = [f[-1] for f in batch]

        ys = torch.zeros([len(batch), a_len], dtype=torch.long).to(0)
        y_masks = torch.zeros([len(batch), a_len], dtype=torch.bool).to(0)
        
        for i, f in enumerate(batch):
            y = f[4][:a_len - 1]
            y_masks[i][:len(y) + 1] = True
            ys[i] = torch.tensor(y + [len(f[1])] * (a_len - len(y)))

        return cx, cq, x, q, x_mask, q_mask, ys, y_masks, qaids

    return qa_collate_fn


def get_qa_filter(config):
    word2idx = config.word2idx
    char2idx = config.char2idx
    x_len = config.sent_size_th
    q_len = config.ques_size_th
    w_len = config.word_size_th

    def qa_filter(data, ss_feature=None):
        xs = []
        features = []
        for ai, article in enumerate(tqdm(data, ncols=100, desc="filtering data")):
            context = []
            sents_len = [len(sent) for sent in article['x']]
            for sent in article['x']:
                x = torch.LongTensor([word2idx.get(word, 1) for word in sent[:x_len]])
                cx = torch.LongTensor([[char2idx.get(char, 1) for char in word[:w_len]] + [0] * (w_len - len(word)) for word in sent[:x_len]])
                context.append((x, cx))
            xs.append(context)

            for qi, qa in enumerate(article['qas']):
                qt, answers, qaid = qa['q'], qa['a'], qa['id']
                q = torch.LongTensor([word2idx.get(word, 1) for word in qt[:q_len]])
                cq = torch.LongTensor([[char2idx.get(char, 1) for char in word[:w_len]] + [0] * (w_len - len(word)) for word in qt[:q_len]])

                scores = ss_feature[ai][qi] if ss_feature is not None else [1. for _ in context]
                sents_indices = np.argsort(scores)[::-1]
                selected_indices, sents_start, sum_len = [], {}, 0
                for si in sents_indices:
                    if sum_len + sents_len[si] <= x_len:
                        selected_indices.append(si)
                        sum_len += sents_len[si]
                    else:
                        break
                selected_indices = sorted(selected_indices)

                selected_x = [word if word in word2idx else '[UNK]' for si in selected_indices for word in article['x'][si]]
                
                sent_start = 0
                for si in selected_indices:
                    sents_start[si] = sent_start
                    sent_start += sents_len[si]
                
                selected_as = []
                for a in answers:
                    # print(a['text'])
                    new_y, complete = [], True
                    for si, ti in a['y']:
                        if si not in selected_indices:
                            complete = False
                            break
                        else:
                            new_y.append(sents_start[si] + ti)
                    if complete:
                        selected_as.append(new_y)
                        # print(' '.join([selected_x[yi] for yi in new_y]))

                if not selected_as:
                    continue
                
                features.append({
                    'sents_indices': selected_indices,
                    'id': qaid,
                    'q': (q, cq),
                    'y': selected_as,
                    'ridx': (ai, qi)
                })
                
        return xs, features

    return qa_filter


class ATDataset(Dataset):
    def __init__(self, data, ss_feature, qa_feature, filter):
        super(ATDataset, self).__init__()
        print('building AT dataset')
        self.data = data
        self.xs, self.features = filter(data, ss_feature, qa_feature)

    
    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        ai, qi = self.features[idx]['ridx']
        sents_indices = self.features[idx]['sents_indices']
        x, cx = list(zip(*[self.xs[ai][si] for si in sents_indices]))
        x, cx = torch.cat(x, dim=0), torch.cat(cx, dim=0)
        q, cq = self.features[idx]['q']
        scores = self.features[idx]['scores']
        qaid = self.features[idx]['id']
        y = self.features[idx]['y']

        return cx, x, cq, q, scores, y, qaid


def get_at_filter(config):
    word2idx = config.word2idx
    char2idx = config.char2idx
    x_len = config.sent_size_th
    q_len = config.ques_size_th
    w_len = config.word_size_th

    def at_filter(data, ss_feature, qa_feature):
        xs = []
        features = []
        for ai, article in enumerate(tqdm(data, ncols=100, desc='filtering data')):
            context = []
            sents_len = [len(sent) for sent in article['x']]
            for sent in article['x']:
                x = torch.LongTensor([word2idx.get(word, 1) for word in sent[:x_len]])
                cx = torch.LongTensor([[char2idx.get(char, 1) for char in word[:w_len]] + [0] * (w_len - len(word)) for word in sent[:x_len]])
                context.append((x, cx))
            xs.append(context)
            for qi, qa in enumerate(article['qas']):
                if (ai, qi) not in qa_feature:
                    continue
                qt, qaid = qa['q'], qa['id']
                at, label = qa_feature[(ai, qi)]
                qa_text = qt[:q_len] + ['[SEP]'] + at[:q_len]
                q = torch.LongTensor([word2idx.get(word, 1) for word in qa_text])
                cq = torch.LongTensor([[char2idx.get(char, 1) for char in word[:w_len]] + [0] * (w_len - len(word)) for word in qa_text])

                scores = ss_feature[ai][qi]
                sents_indices = np.argsort(scores)[::-1]
                selected_indices, sum_len = [], 0
                for si in sents_indices:
                    if sum_len + sents_len[si] <= x_len:
                        selected_indices.append(si)
                        sum_len += sents_len[si]
                    else:
                        break
                selected_indices = sorted(selected_indices)

                features.append({
                    'sents_indices': selected_indices,
                    'id': qaid,
                    'q': (q, cq),
                    'ridx': (ai, qi),
                    'scores': sorted(scores, reverse=True)[:config.ss_feature_size],
                    'y': label
                })
        return xs, features

    return at_filter


def get_at_collate_fn(config):
    score_size = config.ss_feature_size

    def at_collate_fn(batch):
        cx = pad_sequence([f[0] for f in batch]).to(0).transpose(0, 1)
        x = pad_sequence([f[1] for f in batch]).to(0).transpose(0, 1)

        cq = pad_sequence([f[2] for f in batch]).to(0).transpose(0, 1)
        q = pad_sequence([f[3] for f in batch]).to(0).transpose(0, 1)

        x_mask, q_mask = x > 0, q > 0

        scores = torch.tensor([f[4] + [0.] * (score_size - len(f[4])) for f in batch]).to(0)

        qaids = [f[-1] for f in batch]
        ys = torch.tensor([f[5] for f in batch], dtype=torch.bool).to(0)

        return cx, cq, x, q, x_mask, q_mask, scores, ys, qaids

    return at_collate_fn


class SSDataset(Dataset):
    def __init__(self, data, filter=None):
        super(SSDataset, self).__init__()
        print("building SS dataset...")
        self.data = data
        self.xs, self.features = filter(data)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        ai, qi, si = self.features[idx]['ridx']
        q, cq = self.features[idx]['q']
        y = self.features[idx]['y']
        x, cx = self.xs[ai][si]

        return cx, x, cq, q, y


def get_ss_filter(config):
    x_len = config.sent_size_th
    q_len = config.ques_size_th
    w_len = config.word_size_th
    word2idx = config.word2idx
    char2idx = config.char2idx

    def single_filter(data):
        xs = []
        features = []
        for ai, article in enumerate(tqdm(data, ncols=100, desc="filtering data")):
            context = []
            for sent in article['x']:
                x = torch.LongTensor([word2idx.get(word, 1) for word in sent[:x_len]])
                cx = torch.LongTensor([[char2idx.get(char, 1) for char in word[:w_len]] + [0] * (w_len - len(word)) for word in sent[:x_len]])
                context.append((x, cx))
            xs.append(context)
            for qi, qa in enumerate(article['qas']):
                qt, answers, qaid = qa['q'], qa['a'], qa['id']
                if len(answers) > 1:
                    continue

                q = torch.LongTensor([word2idx.get(word, 1) for word in qt[:q_len]])
                cq = torch.LongTensor([[char2idx.get(char, 1) for char in word[:w_len]] + [0] * (w_len - len(word)) for word in qt[:q_len]])
                
                answer_indices = set([pair[0] for pair in answers[0]['y']])
                for si in range(len(article['x'])):
                    features.append({
                        'q': (q, cq),
                        'y': si in answer_indices,
                        'id': qaid,
                        'ridx': (ai, qi, si)
                    })
        return xs, features

    def naive_filter(data):
        xs = []
        features = []
        for ai, article in enumerate(tqdm(data, ncols=100, desc="filtering data (naive)")):
            context = []
            for sent in article['x']:
                x = torch.LongTensor([word2idx.get(word, 1) for word in sent[:x_len]])
                cx = torch.LongTensor([[char2idx.get(char, 1) for char in word[:w_len]] + [0] * (w_len - len(word)) for word in sent[:x_len]])
                context.append((x, cx))
            xs.append(context)
            for qi, qa in enumerate(article['qas']):
                qt, answers, qaid = qa['q'], qa['a'], qa['id']
                q = torch.LongTensor([word2idx.get(word, 1) for word in qt[:q_len]])
                cq = torch.LongTensor([[char2idx.get(char, 1) for char in word[:w_len]] + [0] * (w_len - len(word)) for word in qt[:q_len]])

                answer_indices = set([pair[0] for pair in answers[0]['y']])
                for si in range(len(article['x'])):
                    features.append({
                        'q': (q, cq),
                        'y': si in answer_indices,
                        'id': qaid,
                        'ridx': (ai, qi, si)
                    })
        return xs, features

    if config.ss_filter == 'single':
        return single_filter
    return naive_filter


def get_ss_collate_fn(config):

    def ss_collate_fn(batch):
        
        cx = pad_sequence([f[0] for f in batch]).to(0).transpose(0, 1)
        x = pad_sequence([f[1] for f in batch]).to(0).transpose(0, 1)
        x_mask = torch.zeros_like(x, dtype=torch.bool, device=torch.device(0))

        cq = pad_sequence([f[2] for f in batch]).to(0).transpose(0,1)
        q = pad_sequence([f[3] for f in batch]).to(0).transpose(0, 1)
        q_mask = torch.zeros_like(q, dtype=torch.bool, device=torch.device(0))
        
        for i, f in enumerate(batch):
            x_mask[i][:len(f[1])] = True
            q_mask[i][:len(f[3])] = True

        y = torch.Tensor([f[4] for f in batch]).float().to(0)

        return cx, cq, x, q, x_mask, q_mask, y

    return ss_collate_fn


if __name__ == "__main__":
    max_sent_len, max_q_len, max_word_len = 0, 0, 0
    import ujson as json
    data = json.load(open('data/span/dev.json'))
    from config import get_args
    config = get_args()
    from data_utils.prepro import prepro_vocab
    prepro_vocab(config)
    ss_feature = torch.load('out/QA4IE/SS/dev.SS.pt')
    qa_feature = torch.load('out/QA4IE/QA/dev.QA.pt')
    # ss_feature = None
    # config.ss_filter = 'naive'
    # dataset = SSDataset(data, get_ss_filter(config))
    qa_dataset = QADataset(data, ss_feature, get_qa_filter(config))
    qa_loader = DataLoader(qa_dataset, 2, collate_fn=get_qa_collate_fn(config))
    for batch in qa_loader:
        print(batch)
        break

    at_dataset = ATDataset(data, ss_feature, qa_feature, get_at_filter(config))
    at_loader = DataLoader(at_dataset, 2, collate_fn=get_at_collate_fn(config))
    for batch in at_loader:
        print(batch)
        exit(0)

    exit(0)

    from data_utils.prepro import prepro_vocab
    prepro_vocab(config)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=get_ss_collate_fn(config))
    total_pos = 0
    from tqdm import tqdm
    for i in tqdm(range(len(dataset))):
        sent, q, pos = dataset[i]
        max_sent_len = max(max_sent_len, len(sent))
        max_q_len = max(max_q_len, len(q))
        max_word_len = max(max_word_len, max(map(len, sent)), max(map(len, q)))
        total_pos += pos
    for batch in tqdm(loader):
        ...
    print(len(dataset), total_pos, max_sent_len, max_q_len, max_word_len)
