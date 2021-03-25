from collections import Counter
import os
import numpy as np
import ujson as json
from tqdm import tqdm
from multiprocessing import Pool


class SpacyTokenizer(object):
    def __init__(self, model_name="en_core_web_sm", lower=True):
        import spacy
        self.model = spacy.load(model_name)
        self.lower = lower
    
    def tokenize(self, doc, return_sents=True):
        doc = self.model(doc)

        sents, token_map = [], {}
        for sidx, sent in enumerate(doc.sents):
            tokens = []
            for tidx, token in enumerate(sent):
                token_text = token.text.lower() if self.lower else token.text
                tokens.append(token_text)
                for idx in range(token.idx, token.idx + len(token_text)):
                    token_map[idx] = (sidx, tidx)
            sents.append(tokens)

        if return_sents:
            return sents, token_map
        
        return [token for sent in sents for token in sent]


def load_glove(file_in):
    glove_emb = {}
    for line in tqdm(open(file_in)):
        token, *vec = line.strip().split()
        vec = [float(item) for item in vec]
        assert len(vec) == 100
        glove_emb[token.lower()] = np.array(vec)
    return glove_emb


def prepro_orig_item(article):
    tokenizer = SpacyTokenizer()
    assert len(article['paragraphs']) == 1
    paragraph = article['paragraphs'][0]
    context = paragraph['context']
    x, x_map = tokenizer.tokenize(context)

    qas = []
    for qa in paragraph['qas']:
        q = tokenizer.tokenize(qa['question'], return_sents=False)

        answers = []
        for answer in qa['answers']:
            text, start = answer['text'], answer['answer_start']
            answer_indices = sorted(set([x_map[i] for i in range(start, start+len(text)) if i in x_map]))
            answers.append({'y': answer_indices, 'text': answer['text']})

        qas.append({
            'id': qa['id'], 'q': q, 'a': answers
        })

    return {'context': context, 'x': x, 'qas': qas}


def prepro_orig(config):
    for split in ['train', 'dev', 'test']:
        data_out_path = os.path.join(config.data_dir, f'{split}.json')
        if os.path.exists(data_out_path):
            return

        data_out = []
        # for file_size in ['0-400', '400-700', '700-']:
        for file_size in ['0-400']:
            in_path = os.path.join(
                config.orig_data_dir, config.data_type, file_size, 
                f"{split}.{config.data_type}.json"
            )
            with open(in_path) as fin:
                data = json.load(fin)['data']
            with Pool(config.num_cpus) as pool:
                for item in tqdm(pool.imap(prepro_orig_item, data), total=len(data), ncols=80):
                    data_out.append(item)
                    
        with open(data_out_path, 'w') as fout:
            json.dump(data_out, fout)


def prepro_SS(config):
    ...


def prepro_QA(config):
    ...


def prepro_AT(config):
    ...


def prepro_vocab(config):
    if os.path.exists(os.path.join(config.data_dir, 'word_emb.npy')):
        config.char2idx = json.load(open(os.path.join(config.data_dir, 'char2idx.json')))
        config.word2idx = json.load(open(os.path.join(config.data_dir, 'word2idx.json')))
        config.word_vocab_size = len(config.word2idx)
        config.char_vocab_size = len(config.char2idx)
        config.word_emb = np.load(os.path.join(config.data_dir, 'word_emb.npy'))
        return

    SPECIALS = ['[PAD]', '[UNK]', '[SEP]']
    glove_emb = load_glove(config.glove_path)
    word_counter, char_counter = Counter(), Counter()
    word2idx, char2idx = {w: i for i, w in enumerate(SPECIALS)}, {w: i for i, w in enumerate(SPECIALS)}
    train_path = os.path.join(config.data_dir, 'train.json')
    with open(train_path) as fin:
        for item in tqdm(json.load(fin), ncols=80):
            for sent in item['x']:
                word_counter.update(sent)
                for token in sent:
                    char_counter.update(token)
            
            for qa in item['qas']:
                q = qa['q']
                word_counter.update(q)
                for token in q:
                    char_counter.update(token)

    print(len(word_counter), len(char_counter))

    num_words, num_chars = len(SPECIALS), len(SPECIALS)
    for word in word_counter:
        if word_counter[word] > config.word_count_th:
            word2idx[word] = num_words
            num_words += 1
    for char in char_counter:
        if char_counter[char] > config.char_count_th:
            char2idx[char] = num_chars
            num_chars += 1

    in_glove = 0
    for word in word2idx:
        if word in glove_emb:
            in_glove += 1
    print(num_words, num_chars, in_glove)
    
    idx2word = {idx: w for w, idx in word2idx.items()}
    emb_mat = np.array([
        glove_emb.get(
            idx2word[idx],
            np.random.multivariate_normal(
                np.zeros(config.word_emb_size),
                np.eye(config.word_emb_size)
            )
        ) for idx in range(num_words)], dtype=np.float32)

    config.word_vocab_size = num_words
    config.char_vocab_size = num_chars
    config.word2idx = word2idx
    config.char2idx = char2idx
    config.word_emb = emb_mat

    with open(os.path.join(config.data_dir, 'word2idx.json'), 'w') as fp:
        json.dump(word2idx, fp)
    with open(os.path.join(config.data_dir, 'char2idx.json'), 'w') as fp:
        json.dump(char2idx, fp)
    np.save(os.path.join(config.data_dir, 'word_emb.npy'), emb_mat)


def build_vocab(config):
    ...

if __name__ == "__main__":
    from config import get_args
    config = get_args()
    config.orig_data_dir = 'data/orig_data'
    config.data_dir = 'data/spanS'
    config.word_count_th = 10
    config.char_count_th = 50
    if not os.path.exists(config.data_dir):
        os.makedirs(config.data_dir)
    # prepro_orig(config)
    prepro_vocab(config)
