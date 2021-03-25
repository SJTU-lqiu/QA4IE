""" Official evaluation script for v1.1 of the SQuAD dataset. """
from __future__ import print_function
from collections import Counter
import string
import re
import argparse
import json
import sys


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

def exact_match_score(prediction, ground_truth):
    return (normalize_answer(prediction) == normalize_answer(ground_truth))


def metric_max_over_ground_truths(metric_fn, prediction, ground_truths):
    scores_for_ground_truths = []
    for ground_truth in ground_truths:
        score = metric_fn(prediction, ground_truth)
        scores_for_ground_truths.append(score)
    return max(scores_for_ground_truths)


def evaluate(dataset, predictions, th):
    fe = open('./em_result', 'w')
    pred_num = gt_num = correct_num = total = 0
    for article in dataset:
        for paragraph in article['paragraphs']:
            for qa in paragraph['qas']:
                total += 1
                if qa['id'] not in predictions:
                    message = 'Unanswered question ' + qa['id'] + \
                              ' will receive score 0.'
                    print(message, file=sys.stderr)
                    continue
                ground_truths = list(map(lambda x: x['text'], qa['answers']))
                prediction = predictions[qa['id']]
                if ground_truths != []:
                    gt_num += 1
                if (predictions['scores'][qa['id']] >= float(th)) & (prediction != ''):
                    pred_num += 1
                if (ground_truths != []) & (predictions['scores'][qa['id']] >= float(th)) & (prediction != ''):
                    em_res = metric_max_over_ground_truths(
                        exact_match_score, prediction, ground_truths)
                    hasP = 0
                    for str in ground_truths:
                        for i, ch in enumerate(str):
                            if ch in string.punctuation:
                                hasP = 1
                    print('%s:\t%d\t%d' % (qa['id'], em_res, hasP), file = fe)
                    correct_num += em_res
                #print('%s:\t%d\t%d\t%d\t%d' % (qa['id'], pred_num, gt_num, correct_num, total), file=fe)
    fe.close()

    exact_match = 100.0 * correct_num / total
    pre = 100.0 * correct_num / pred_num
    rec = 100.0 * correct_num / gt_num
    f1 = 2 * pre * rec / (pre + rec)

    f = open('./fpr_result', 'a+')
    print('%.2f\t%.2f\t%.2f\t%.2f\t%.2f' % (float(th), exact_match, pre, rec, f1), file = f)
    f.close()
    
    return 1


if __name__ == '__main__':
    expected_version = '1.0'
    parser = argparse.ArgumentParser(
        description='Evaluation for SQuAD ' + expected_version)
    parser.add_argument('dataset_file', help='Dataset file')
    parser.add_argument('prediction_file', help='Prediction File')
    parser.add_argument('threshold', help='threshold of prediction score')
    args = parser.parse_args()
    th = args.threshold
    with open(args.dataset_file) as dataset_file:
        dataset_json = json.load(dataset_file)
        if (dataset_json['version'] != expected_version):
            print('Evaluation expects v-' + expected_version +
                  ', but got dataset with v-' + dataset_json['version'],
                  file=sys.stderr)
        dataset = dataset_json['data']
    with open(args.prediction_file) as prediction_file:
        predictions = json.load(prediction_file)
    evaluate(dataset, predictions, th)