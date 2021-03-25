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

        print(len(self.features), sum(len(a['qas']) for a in self.data))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        ai, qi = self.features[idx]['ridx']
        sents_indices = self.features[idx]['sents_indices']
        x, cx = list(zip(*[self.xs[ai][si] for si in sents_indices]))
        x, cx = torch.cat(x + (torch.LongTensor([2]),), dim=0), torch.cat(cx + (torch.LongTensor([[2]*cx[0].size(1)]),), dim=0)
        q, cq = self.features[idx]['q']
        y = random.choice(self.features[idx]['y'])
        qaid = self.features[idx]['id']

        return cx, x, cq, q, y, qaid


def get_qa_collate_fn(config):
    a_len = config.max_decode_length
    def qa_collate_fn(batch):
        
        cx = pad_sequence([f[0] for f in batch]).to(0).transpose(0, 1)
        x = pad_sequence([f[1] for f in batch]).to(0).transpose(0, 1)
        x_mask = torch.zeros_like(x, dtype=torch.bool, device=torch.device(0))

        cq = pad_sequence([f[2] for f in batch]).to(0).transpose(0,1)
        q = pad_sequence([f[3] for f in batch]).to(0).transpose(0, 1)
        q_mask = torch.zeros_like(q, dtype=torch.bool, device=torch.device(0))

        qaids = [f[-1] for f in batch]

        ys = torch.zeros([len(batch), a_len], dtype=torch.long).to(0)
        y_masks = torch.zeros([len(batch), a_len], dtype=torch.bool).to(0)
        
        for i, f in enumerate(batch):
            x_mask[i][:len(f[1])] = True
            q_mask[i][:len(f[3])] = True

            # print(f[4])
            y = f[4][:a_len - 1]
            y_masks[i][:len(y) + 1] = True
            # print(y)
            # print(y + [len(f[1])] * (a_len - len(y)))
            ys[i] = torch.tensor(y + [len(f[1]) - 1] * (a_len - len(y)))

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
                
                sent_start = 0
                for si in selected_indices:
                    sents_start[si] = sent_start
                    sent_start += sents_len[si]
                
                selected_as = []
                for a in answers:
                    new_y, complete = [], True
                    for si, ti in a['y']:
                        if si not in selected_indices:
                            complete = False
                            break
                        else:
                            new_y.append(sents_start[si] + ti)
                    if complete:
                        selected_as.append(new_y)

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

    # def qa_filter(ai, article, ss_feature):
    #     ret = []
    #     sents_len = [len(sent) for sent in article['x']]
    #     for qi, qa in enumerate(article['qas']):
    #         q, answers, scores = qa['q'], qa['a'], ss_feature[qi]
    #         sents_indices = np.argsort(scores)[::-1]
    #         selected_indices, sents_start, sum_len = [], {}, 0
    #         for si in sents_indices:
    #             if sum_len + sents_len[si] <= len_limit:
    #                 selected_indices.append(si)
    #                 sum_len += sents_len[si]
    #             else:
    #                 break
    #         selected_indices = sorted(selected_indices)
    #         # print(len(selected_indices), max(scores), min(scores), len(article['x']))
    #         sent_start = 0
    #         for si in selected_indices:
    #             sents_start[si] = sent_start
    #             sent_start += sents_len[si]

    #         selected_as = []
    #         for a in answers:
    #             new_y, complete = [], True
    #             for si, ti in a['y']:
    #                 if si not in selected_indices:
    #                     complete = False
    #                     break
    #                 else:
    #                     new_y.append(sents_start[si] + ti)
    #             if complete:
    #                 selected_as.append(new_y)
            
    #         if not selected_as:
    #             continue
            
    #         ret.append([ai, qi, selected_indices, selected_as])
    #     return ret

    # return qa_filter


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

        # if y > 0:
        #     print(self.data[ai]['x'][si], self.data[ai]['qas'][qi])

        return cx, x, cq, q, y


def get_ss_filter(config):
    x_len = config.sent_size_th
    q_len = config.ques_size_th
    w_len = config.word_size_th
    word2idx = config.word2idx
    char2idx = config.char2idx

    # def single_filter(article, ai):
    #     indices = []
    #     labels = []
    #     for qi, qa in enumerate(article['qas']):
    #         q, answers = qa['q'], qa['a']
    #         answer_indices = set([pair[0] for a in answers for pair in a['y']])
    #         if len(answer_indices) > 1:
    #             continue
    #         for si in range(len(article['x'])):
    #             indices.append([ai, qi, si])
    #             labels.append(int(si in answer_indices))
    #             # if si in answer_indices:
    #                 # print(q, article['x'][si], [config.word2idx.get(token, 1) for token in article['x'][si]], answers[0]['text'])

    #     #TODO debug test
    #     print(sum(labels), len(labels))
    #     # random.shuffle(labels)
    #     return indices, labels

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

    def naive_filter(article, ai):
        indices = []
        labels = []
        for qi, qa in enumerate(article['qas']):
            q, answers = qa['q'], qa['a']
            answer_indices = set([pair[0] for a in answers for pair in a['y']])
            for si in range(len(article['x'])):
                indices.append([ai, qi, si])
                labels.append(int(si in answer_indices))

        return indices, labels

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

        y = torch.LongTensor([f[4] for f in batch]).to(0)

        return cx, cq, x, q, x_mask, q_mask, y

    return ss_collate_fn

    # def ss_collate_fn(batch):
    #     N = len(batch)
    #     x = torch.zeros([N, config.sent_size_th], dtype=torch.long).to(0)
    #     cx = torch.zeros([N, config.sent_size_th, config.word_size_th], dtype=torch.long).to(0)
    #     x_mask = torch.zeros([N, config.sent_size_th], dtype=torch.bool).to(0)
    #     q = torch.zeros([N, config.ques_size_th], dtype=torch.long).to(0)
    #     cq = torch.zeros([N, config.ques_size_th, config.word_size_th], dtype=torch.long).to(0)
    #     q_mask = torch.zeros([N, config.ques_size_th], dtype=torch.bool).to(0)
    #     labels = torch.zeros([N], dtype=torch.long).to(0)

    #     for i in range(N):
    #         senti, qi, label = batch[i]
    #         x_len = min(config.sent_size_th, len(senti))
    #         q_len = min(config.ques_size_th, len(qi))
    #         x[i][:x_len] = torch.Tensor([config.word2idx.get(senti[j], 1) for j in range(x_len)])
    #         x_mask[i][:x_len] = True
    #         for j in range(x_len):
    #             w_len = min(config.word_size_th, len(senti[j]))
    #             cx[i][j][:w_len] = torch.Tensor([config.char2idx.get(senti[j][k], 1) for k in range(w_len)])
    #         q[i][:q_len] = torch.Tensor([config.word2idx.get(qi[j], 1) for j in range(q_len)])
    #         q_mask[i][:q_len] = True
    #         for j in range(q_len):
    #             w_len = min(config.word_size_th, len(qi[j]))
    #             cq[i][j][:w_len] = torch.Tensor([config.char2idx.get(qi[j][k], 1) for k in range(w_len)])
    #         labels[i] = label

    #     return cx, cq, x, q, x_mask, q_mask, labels

    # return ss_collate_fn


if __name__ == "__main__":
    max_sent_len, max_q_len, max_word_len = 0, 0, 0
    import ujson as json
    data = json.load(open('data/spanS/train.json'))
    from config import get_args
    config = get_args()
    from data_utils.prepro import prepro_vocab
    prepro_vocab(config)
    # ss_feature = torch.load('out/QA4IE/SS/dev.SS.pt')
    ss_feature = None
    # config.ss_filter = 'naive'
    # dataset = SSDataset(data, get_ss_filter(config))
    qa_dataset = QADataset(data, ss_feature, get_qa_filter(config))
    qa_dataset[1]

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
