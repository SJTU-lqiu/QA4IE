import numpy as np
import tensorflow as tf
import copy
import logging

from basic.read_data import DataSet
from my.nltk_utils import span_f1
from my.tensorflow import padded_reshape
from my.utils import argmax
from squad.utils import get_phrase, get_best_span, my_get_best_span, my_get_phrase
from tensorflow.python.client import timeline


class Evaluation(object):
    def __init__(self, data_type, global_step, idxs, yp, yp2, y, correct, loss, f1s, id2answer_dict, tensor_dict=None):
        self.data_type = data_type
        self.global_step = global_step
        self.idxs = idxs
        self.yp = yp
        self.num_examples = len(yp)
        self.tensor_dict = None
        self.dict = {'data_type': data_type,
                     'global_step': global_step,
                     'yp': yp,
                     'idxs': idxs,
                     'num_examples': self.num_examples}
        if tensor_dict is not None:
            self.tensor_dict = {key: val.tolist() for key, val in tensor_dict.items()}
            for key, val in self.tensor_dict.items():
                self.dict[key] = val
        self.summaries = None

        self.y = y
        self.dict['y'] = y

        self.loss = loss
        self.correct = correct
        self.acc = sum(correct) / len(correct)
        self.dict['loss'] = loss
        self.dict['correct'] = correct
        self.dict['acc'] = self.acc
        loss_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/loss'.format(data_type), simple_value=self.loss)])
        acc_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/acc'.format(data_type), simple_value=self.acc)])
        self.summaries = [loss_summary, acc_summary]

        self.yp2 = yp2
        self.f1s = f1s
        self.f1 = float(np.mean(f1s))
        self.dict['yp2'] = yp2
        self.dict['f1s'] = f1s
        self.dict['f1'] = self.f1
        self.id2answer_dict = id2answer_dict
        f1_summary = tf.Summary(value=[tf.Summary.Value(tag='{}/f1'.format(data_type), simple_value=self.f1)])
        self.summaries.append(f1_summary)

    def __repr__(self):
        return "{} step {}: loss={:.4f}".format(self.data_type, self.global_step, self.loss)

    def __add__(self, other):
        if other == 0:
            return self
        assert self.data_type == other.data_type
        assert self.global_step == other.global_step
        new_idxs = self.idxs + other.idxs
        new_yp = self.yp + other.yp
        new_yp2 = self.yp2 + other.yp2
        new_y = self.y + other.y
        new_correct = self.correct + other.correct
        new_f1s = self.f1s + other.f1s
        new_loss = (self.loss * self.num_examples + other.loss * other.num_examples) / len(new_correct)
        new_id2answer_dict = dict(list(self.id2answer_dict.items()) + list(other.id2answer_dict.items()))
        new_id2score_dict = dict(
            list(self.id2answer_dict['scores'].items()) + list(other.id2answer_dict['scores'].items()))
        new_id2answer_dict['scores'] = new_id2score_dict
        return Evaluation(self.data_type, self.global_step, new_idxs, new_yp, new_yp2, new_y, new_correct, new_loss,
                          new_f1s, new_id2answer_dict)

    def __radd__(self, other):
        return self.__add__(other)


class Evaluator(object):
    def __init__(self, config, models, tensor_dict=None):
        self.config = config
        self.model = models[0]
        self.models = models
        self.global_step = models[0].global_step
        self.yp = models[0].yp
        self.yp2 = models[0].yp2
        self.yp_list = models[0].decoder_inference
        self.yp_mat = models[0].decoder_train_softmax
        self.loss = models[0].loss
        self.tensor_dict = {} if tensor_dict is None else tensor_dict
        self.sent_size_th = config.sent_size_th

        self.y = models[0].y

        #word2index = config.word2idx.copy()
        #word2index.update(config.new_word2idx)

        #self.index2word = {v: k for k, v in word2index.items()}

        with tf.name_scope("eval_concat"):
            N, M, JX = config.batch_size, config.max_num_sents, config.max_sent_size
            self.yp = tf.concat([padded_reshape(model.yp, [N, M, JX]) for model in models], 0)
            self.yp2 = tf.concat([padded_reshape(model.yp2, [N, M, JX]) for model in models], 0)
            self.loss = tf.add_n([model.loss for model in models]) / len(models)

    def _split_batch(self, batches):
        idxs_list, data_sets = zip(*batches)
        idxs = sum(idxs_list, ())
        data_set = sum(data_sets, data_sets[0].get_empty())
        return idxs, data_set

    def _get_feed_dict(self, batches):
        feed_dict = {}
        for model, (_, data_set) in zip(self.models, batches):
            feed_dict.update(model.get_feed_dict(data_set, False))
        return feed_dict

    def get_evaluation_from_batches(self, sess, batches):
        e = sum(self.get_evaluation(sess, batch) for batch in batches)
        return e

    def get_evaluation(self, sess, batch):
        idxs, data_set = self._split_batch(batch)
        assert isinstance(data_set, DataSet)
        feed_dict = self._get_feed_dict(batch)
        if self.config.profiling:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            global_step, yp, yp2, loss, vals, yp_list, yp_mat = sess.run(
                [self.global_step, self.yp, self.yp2, self.loss, list(self.tensor_dict.values()), self.yp_list,
                 self.yp_mat], feed_dict=feed_dict, options=run_options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('evaluation_timeline.json', 'w') as f:
                f.write(ctf)
                print('profile info save into file: ', 'evaluation_timeline.json')
        else:
            global_step, yp, yp2, loss, vals, yp_list, yp_mat = sess.run(
                [self.global_step, self.yp, self.yp2, self.loss, list(self.tensor_dict.values()), self.yp_list, self.yp_mat], feed_dict=feed_dict)

        y = data_set.data['y']

        yp, yp2 = yp[:data_set.num_examples], yp2[:data_set.num_examples]
        yp_list = yp_list[:data_set.num_examples]

        self.convert_y_word_by_word(y)

        if self.config.conditional_probability:
            yp_list = self.get_ans_pointers(yp_mat)

        ans_text_list = self.get_ans_text_list(yp_list, data_set.data['x'])

        gt_text_list = [self.get_gt_ans_text (xi, yi)
                        for xi, yi in zip(data_set.data['x'], y)]

        def _get(words, yplist, ypmat, context):
            rets, probs = [], []
            words = words[0]
            ff = open('./eval_debug', 'w')
            print(yplist, file=ff)
            print(words, file=ff)
            print(context, file=ff)
            ch_idx = 0
            word_pos = []
            for word in words:
                ch_idx = context.find(word, ch_idx)
                st = ch_idx
                assert ch_idx >= 0
                ch_idx += len(word)
                word_pos.append([st, ch_idx])
            ret = ''
            cnt = 0
            if self.config.score_mode == 'mul_score':
                prob = 1.0
            else:
                prob = 0.0
            eos_symbol = len(words)
            for idx, yp in enumerate(yplist):
                if yp >= eos_symbol or yp >= self.sent_size_th:
                    break
                cnt += 1
                if idx == 0:
                    ret = context[word_pos[yp][0]:word_pos[yp][1]]
                else:
                    if yp == yplist[idx - 1] + 1:
                        fyp = yplist[idx - 1]
                        if word_pos[fyp][1] < word_pos[yp][0]:
                            ret += context[word_pos[fyp][1]:word_pos[yp][0]]
                    else:
                        ret += ' '
                    ret += context[word_pos[yp][0]:word_pos[yp][1]]
                assert yp == np.argmax(ypmat[idx])
                if self.config.score_mode == 'mul_score':
                    prob = prob * ypmat[idx][yp]
                else:
                    prob = prob + ypmat[idx][yp]
                print(ret+' '+str(prob), file=ff)
                rets.append(ret)
                probs.append(prob)
            assert ret != None
            if self.config.score_mode == 'avg_score':
                if cnt == 0:
                    prob = 0.0
                else:
                    prob = prob / cnt
            # prob = max(probs)
            # ret = rets[np.argmax(probs)]
            print(ret + ' ' + str(prob), file=ff)
            ff.close()
            return ret, prob


        # def _get2(context, xi, span):
        #     if len(xi) <= span[0][0]:
        #         return ""
        #     if len(xi[span[0][0]]) <= span[1][1]:
        #         return ""
        #
        #     return " ".join(my_get_phrase(xi, span))

        # for i in  ans_text_list:
        #     print (i)
        id2answer_dict = {}
        id2score_dict = {}
        for id_, yp_, yp_m, x_, context in zip(data_set.data['ids'], yp_list, yp_mat, data_set.data['x'], data_set.data['p']):
            ans, score = _get(x_, yp_, yp_m, context)
            id2answer_dict.update({id_: ans})
            id2score_dict.update({id_: score})
        id2answer_dict['scores'] = id2score_dict
        assert len(data_set.data['ids']) == len(yp_list)
        assert len(data_set.data['x']) == len(yp_list)
        assert len(data_set.data['p']) == len(yp_list)
        correct = [self.__class__.is_exact_match(yi, ypi, xi, gi) for yi, ypi, xi, gi in zip(gt_text_list, ans_text_list, data_set.data['x'], gt_text_list)]
        f1s = [self.__class__.f1_score(yi, ypi, xi) for yi, ypi, xi in zip(gt_text_list, ans_text_list, data_set.data['x'])]
        # f1s = [self.__class__.span_f1(yi, span) for yi, span in zip(gt_text_list, spans)]
        tensor_dict = dict(zip(self.tensor_dict.keys(), vals))
        e = Evaluation(data_set.data_type, int(global_step), idxs, yp.tolist(), yp2.tolist(), y,
                       correct, float(loss), f1s, id2answer_dict, tensor_dict=tensor_dict)
        return e


    @staticmethod
    def is_exact_match(yi, ypi, xi, gi):
        #if len(ypi) > 10:
            #print('###########ypi', ypi)
            #print('x:', xi[0])
            #print('###########g', gi)

        for i, word_list in enumerate(yi):
            if word_list == ypi:
                return True
        return False

    @staticmethod
    def f1_score(yi, ypi, xi):
        max_f1 = 0
        for yij in yi:
            f1 = span_f1(yij, ypi)
            max_f1 = max(f1, max_f1)
        return max_f1

    @staticmethod
    def span_f1(yi, span):
        max_f1 = 0
        for start, stop in yi:
            if start[0] == span[0][0]:
                true_span = start[1], stop[1]
                pred_span = span[0][1], span[1][1]
                f1 = span_f1(true_span, pred_span)
                max_f1 = max(f1, max_f1)
        return max_f1

    def convert_y_word_by_word(self, y):
        for i, yi in enumerate(y):
            for ij, yij in enumerate(yi):
                if self.config.data_type == 'span':
                    ans_sent_index = yij[0][0]
                    start, stop = yij
                    y[i][ij] = []
                    for index in range(start[1], stop[1] + 1):
                        y[i][ij].append([ans_sent_index, index])
                elif self.config.data_type == 'seq':
                    ans_sent_index = yij[0][0]
                    y[i][ij] = []
                    for loc in yij:
                        y[i][ij].append([ans_sent_index, loc[1]])


    def get_ans_text_list(self, yp_list, x):
        ans_text_list = []
        for ai,xi in zip(yp_list, x):
            ans_text = []
            eos_symbol = len(xi[0])
            for ti in ai:
                if ti == eos_symbol:
                    break
                if ti <= len(xi[0]):
                    ans_text.append(xi[0][ti])
            ans_text_list.append(ans_text)
        #print("debug_info: ", ans_text_list)
        return ans_text_list

    def get_gt_ans_text(self, xi, yi):
        gt_text = []
        sent_num = yi[0][0][0]
        for yij in yi:
            del yij[-1]
            gt_text_i = []
            for word_pos in yij:
                if word_pos[0] == sent_num and word_pos[1] < len(xi[sent_num]):
                    gt_text_i.append(xi[word_pos[0]][word_pos[1]])
            gt_text.append(gt_text_i)
        return gt_text

    def get_ans_pointers(self, yp_mat):
        ans_index_mat = []
        for i, yi in enumerate(yp_mat):
            tmp = []
            tmp_idx = []
            end_index = len(yi[0]) - 1
            for j, yij in enumerate(yi):
                max_i = np.argmax(yij)
                if max_i == end_index or j == len(yi) - 1:
                    tmp_idx.append(end_index)
                    tmp.append([0, yij[end_index]])
                    break
                else:
                    tmp_idx.append(max_i)
                    tmp.append([yij[max_i], yij[end_index]])
            #print('tmp', tmp)
            one_ans = []
            pre_vec = []
            mul = 1
            for item in tmp:
                mul *= item[0]
                pre_vec.append(mul)

            for i in range(0, len(tmp) - 1):
                end_prob = tmp[i][1]
                cont_prob = 0
                for j in range(i, len(tmp) - 1):
                    cont_prob += pre_vec[j] * tmp[j + 1][1]
                #print(end_prob, cont_prob)
                if end_prob >= cont_prob:
                    one_ans.append(end_index)
                    #print('end', end_index)
                    break
                else:
                    one_ans.append(tmp_idx[i])

                for j in range(i, len(pre_vec)):
                    pre_vec[j] /= tmp[i][0]

            if len(one_ans) == 0 or one_ans[-1] != end_index:
                one_ans.append(end_index)
                #print(one_ans)
            ans_index_mat.append(one_ans[:])

        return ans_index_mat  # [N, JA + 1]
