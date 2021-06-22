import json
import os
from collections import Counter, defaultdict
from tqdm import tqdm
import random


data_dir = 'data/span'
th = 0.001
r_counter, r2_counter = Counter(), Counter()
rr_dict = defaultdict(list)

with open(os.path.join(data_dir, 'train.json')) as fp:
    train_data = json.load(fp)
    print(len(train_data))

for article in tqdm(train_data, ncols=100):
    qs = [tuple(qa['q']) for qa in article['qas']]
    r_counter.update(qs)
    for qi in qs:
        for qj in qs:
            r2_counter[(qi, qj)] += 1

for qi, qj in tqdm(r2_counter, ncols=100):
    p = r2_counter[(qi, qj)] / float(r_counter[qi])
    if p >= th:
        rr_dict[qi].append(qj)

with open('data/span/train.json') as fp:
    ietrain = random.sample(json.load(fp), 5000)

with open('data/span/dev.json') as fp:
    iedev = random.sample(json.load(fp), 1000)

with open('data/span/test.json') as fp:
    ietest = random.sample(json.load(fp), 5000)

for tag, iedata in zip(['train', 'dev', 'test'], [ietrain, iedev, ietest]):
    pos_num, neg_num = 0, 0
    for ai, article in tqdm(enumerate(iedata), ncols=100):
        pos_qs = [tuple(qa['q']) for qa in article['qas']]
        neg_qs = list(set([qj for qi in pos_qs for qj in rr_dict[qi] if qj not in pos_qs]))
        pos_num += len(pos_qs)
        neg_num += len(neg_qs)
        
        qas = article['qas']
        added_qas = []
        for qi, q in enumerate(neg_qs):
            added_qas.append({
                'id': f'ieneg-span-{ai}-{qi}',
                'q': list(q),
                # placeholder answer
                'a': [{
                    'y': [[0, 0]],
                    'text': article['x'][0][0]
                }]
            })
        new_qas = qas + added_qas
        article['qas'] = new_qas

    with open(os.path.join(data_dir, f'ie{tag}.json'), 'w') as fp:
        json.dump(iedata, fp)

    print(f'{pos_num}/{pos_num + neg_num} positive triples in ie {tag} set')    
