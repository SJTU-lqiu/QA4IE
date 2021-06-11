"""python inline script for squad style evaluation"""
from re import T
import string
from collections import Counter
import re
import torch


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""
    def remove_articles(text):
        return re.sub(r'\b(a|an|the)\b', ' ', text)

    def white_space_fix(text):
        return ' '.join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return ''.join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction, ground_truth):
    prediction_tokens = normalize_answer(prediction).split()
    ground_truth_tokens = normalize_answer(ground_truth).split()
    common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction_tokens)
    recall = 1.0 * num_same / len(ground_truth_tokens)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions):
    total, f1, em = 0, 0, 0
    for article in dataset:
        for qa in article['qas']:
            total += 1
            qaid, answers = qa['id'], qa['a']
            if qaid not in predictions:
                continue
            ground_truths = list(map(lambda x: x['text'], answers))
            prediction = predictions[qa['id']]
            em_score = metric_max_over_ground_truths(
                exact_match_score, prediction, ground_truths
            )
            em += em_score
            f1 += metric_max_over_ground_truths(
                f1_score, prediction, ground_truths
            )

    em = 100. * em / total
    f1 = 100. * f1 / total

    return em, f1


"""evaluate with negative examples"""
def evaluate_pr(dataset, predictions):
    tps = []
    scores = []
    total = 0
    for article in dataset:
        for qa in article['qas']:
            total += 1
            qaid, answers = qa['id'], qa['a']
            if qaid not in predictions:
                continue
            if qaid.startswith('ietest'):
                tps.append(0)
                scores.append(predictions[qaid][1])
            else:
                pred_text, score = predictions[qaid]
                ground_truths = list(map(lambda x: x['text'], answers))
                em_score = metric_max_over_ground_truths(
                    exact_match_score, pred_text, ground_truths
                )
                tps.append(em_score)
                scores.append(score)
    
    sorted_scores, sorted_indices = torch.tensor(scores).sort(descending=True)
    sorted_tps = torch.tensor(tps)[sorted_indices]
    tps = sorted_tps.cumsum(dim=0)
    positive = torch.arange(len(tps)).to(tps) + 1
    prec = tps / positive
    rec = tps / total
    f1s = 2 * prec * rec / (prec + rec)
    f1s[f1s.isnan()] = 0.
    maxf1, maxi = f1s.max(dim=0)

    return maxf1, prec, rec


