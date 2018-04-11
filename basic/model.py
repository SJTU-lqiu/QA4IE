import random

import itertools
import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple, GRUCell, LSTMBlockFusedCell

from basic.read_data import DataSet
from my.tensorflow import get_initializer, flatten
from my.tensorflow.nn import softsel, get_logits, highway_network, multi_conv1d, softmax, linear
from my.tensorflow.rnn import bidirectional_dynamic_rnn
from my.tensorflow.rnn_cell import SwitchableDropoutWrapper, AttentionCell
from basic.ptrnet import ptr_decoder

def get_multi_gpu_models(config):
    models = []
    with tf.variable_scope(""):
        for gpu_idx in range(config.num_gpus):
            with tf.name_scope("model_{}".format(gpu_idx)) as scope, tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                model = Model(config, scope, rep=gpu_idx == 0)
                tf.get_variable_scope().reuse_variables()
                models.append(model)
    return models


class Model(object):
    def __init__(self, config, scope, rep=True):
        self.scope = scope
        self.config = config
        self.global_step = tf.get_variable('global_step', shape=[], dtype='int32',
                                           initializer=tf.constant_initializer(0), trainable=False)

        # Define forward inputs here
        N, M, JX, JQ, VW, VC, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.max_word_size
        self.x = tf.placeholder('int32', [N, None, None], name='x')
        self.emx = tf.placeholder('int32', [N, None, None], name='emx') # exact match feature of context
        self.cx = tf.placeholder('int32', [N, None, None, W], name='cx')
        self.x_mask = tf.placeholder('bool', [N, None, None], name='x_mask')
        self.q = tf.placeholder('int32', [N, None], name='q')
        self.emq = tf.placeholder('int32', [N, None], name='emq') # exact match feature of query
        self.cq = tf.placeholder('int32', [N, None, W], name='cq')
        self.q_mask = tf.placeholder('bool', [N, None], name='q_mask')
        self.y = tf.placeholder('bool', [N, None, None], name='y')
        self.y2 = tf.placeholder('bool', [N, None, None], name='y2')
        self.answer_string = tf.placeholder('int32', [N, None, W], name='answer_string')
        self.answer_string_length = tf.placeholder('int32', [N, None, W], name='answer_string')
        self.is_train = tf.placeholder('bool', [], name='is_train')
        self.new_emb_mat = tf.placeholder('float', [None, config.word_emb_size], name='new_emb_mat')

        # Define misc
        self.tensor_dict = {}

        # Forward outputs / loss inputs
        self.logits = None
        self.yp = None

        # Loss outputs
        self.loss = None

        self._build_forward()
        self._build_loss()
        self.var_ema = None
        if rep:
            self._build_var_ema()
        if config.mode == 'train':
            self._build_ema()

        self.summary = tf.summary.merge_all()
        self.summary = tf.summary.merge(tf.get_collection("summaries", scope=self.scope))

    def _build_forward(self):
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, \
            config.max_word_size
        print("VW:", VW, "N:", N, "M:", M, "JX:", JX, "JQ:", JQ)
        JA = config.max_answer_length
        JX = tf.shape(self.x)[2]
        JQ = tf.shape(self.q)[1]
        M = tf.shape(self.x)[1]
        print("VW:", VW, "N:", N, "M:", M, "JX:", JX, "JQ:", JQ)
        dc, dw, dco = config.char_emb_size, config.word_emb_size, config.char_out_size

        with tf.variable_scope("emb"):
            # Char-CNN Embedding
            if config.use_char_emb:
                with tf.variable_scope("emb_var"), tf.device("/cpu:0"):
                    char_emb_mat = tf.get_variable("char_emb_mat", shape=[VC, dc], dtype='float')

                with tf.variable_scope("char"):
                    Acx = tf.nn.embedding_lookup(char_emb_mat, self.cx)  # [N, M, JX, W, dc]
                    Acq = tf.nn.embedding_lookup(char_emb_mat, self.cq)  # [N, JQ, W, dc]
                    Acx = tf.reshape(Acx, [-1, JX, W, dc])
                    Acq = tf.reshape(Acq, [-1, JQ, W, dc])

                    filter_sizes = list(map(int, config.out_channel_dims.split(',')))
                    heights = list(map(int, config.filter_heights.split(',')))
                    assert sum(filter_sizes) == dco, (filter_sizes, dco)
                    with tf.variable_scope("conv"):
                        xx = multi_conv1d(Acx, filter_sizes, heights, "VALID",  self.is_train, config.keep_prob, scope="xx")
                        if config.share_cnn_weights:
                            tf.get_variable_scope().reuse_variables()
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="xx")
                        else:
                            qq = multi_conv1d(Acq, filter_sizes, heights, "VALID", self.is_train, config.keep_prob, scope="qq")
                        xx = tf.reshape(xx, [-1, M, JX, dco])
                        qq = tf.reshape(qq, [-1, JQ, dco])

            # Word Embedding
            if config.use_word_emb:
                with tf.variable_scope("emb_var") as scope, tf.device("/cpu:0"):
                    if config.mode == 'train':
                        word_emb_mat = tf.get_variable("word_emb_mat", dtype='float', shape=[VW, dw], initializer=get_initializer(config.emb_mat))
                    else:
                        word_emb_mat = tf.get_variable("word_emb_mat", shape=[VW, dw], dtype='float')
                    tf.get_variable_scope().reuse_variables()
                    self.word_emb_scope = scope
                    if config.use_glove_for_unk:
                        word_emb_mat = tf.concat([word_emb_mat, self.new_emb_mat], 0)

                with tf.name_scope("word"):
                    Ax = tf.nn.embedding_lookup(word_emb_mat, self.x)  # [N, M, JX, d]
                    Aq = tf.nn.embedding_lookup(word_emb_mat, self.q)  # [N, JQ, d]
                    self.tensor_dict['x'] = Ax
                    self.tensor_dict['q'] = Aq
                # Concat Char-CNN Embedding and Word Embedding
                if config.use_char_emb:
                    xx = tf.concat([xx, Ax], 3)  # [N, M, JX, di]
                    qq = tf.concat([qq, Aq], 2)  # [N, JQ, di]
                else:
                    xx = Ax
                    qq = Aq

            # exact match
            if config.use_exact_match:
                emx = tf.expand_dims(tf.cast(self.emx, tf.float32), -1)
                xx = tf.concat([xx, emx], 3)  # [N, M, JX, di+1]
                emq = tf.expand_dims(tf.cast(self.emq, tf.float32), -1)
                qq = tf.concat([qq, emq], 2)  # [N, JQ, di+1]


        # 2 layer highway network on Concat Embedding
        if config.highway:
            with tf.variable_scope("highway"):
                xx = highway_network(xx, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)
                tf.get_variable_scope().reuse_variables()
                qq = highway_network(qq, config.highway_num_layers, True, wd=config.wd, is_train=self.is_train)

        self.tensor_dict['xx'] = xx
        self.tensor_dict['qq'] = qq

        # Bidirection-LSTM (3rd layer on paper)
        cell = GRUCell(d) if config.GRU else BasicLSTMCell(d, state_is_tuple=True)
        d_cell = SwitchableDropoutWrapper(cell, self.is_train, input_keep_prob=config.input_keep_prob)
        x_len = tf.reduce_sum(tf.cast(self.x_mask, 'int32'), 2)  # [N, M]
        q_len = tf.reduce_sum(tf.cast(self.q_mask, 'int32'), 1)  # [N]
        flat_x_len = flatten(x_len, 0)  # [N * M]

        with tf.variable_scope("prepro"):
            if config.use_fused_lstm:
                with tf.variable_scope("u1"):
                    fw_inputs = tf.transpose(qq, [1, 0, 2]) #[time_len, batch_size, input_size]
                    bw_inputs = tf.reverse_sequence(fw_inputs, q_len, batch_dim=1, seq_dim=0)
                    fw_inputs = tf.nn.dropout(fw_inputs, config.input_keep_prob)
                    bw_inputs = tf.nn.dropout(bw_inputs, config.input_keep_prob)
                    prep_fw_cell = LSTMBlockFusedCell(d, cell_clip=0)
                    prep_bw_cell = LSTMBlockFusedCell(d, cell_clip=0)
                    fw_outputs, fw_final = prep_fw_cell(fw_inputs, dtype=tf.float32, sequence_length=q_len, scope="fw")
                    bw_outputs, bw_final = prep_bw_cell(bw_inputs, dtype=tf.float32, sequence_length=q_len, scope="bw")
                    bw_outputs = tf.reverse_sequence(bw_outputs, q_len, batch_dim=1, seq_dim = 0)
                    current_inputs = tf.concat((fw_outputs, bw_outputs), 2)
                    output = tf.transpose(current_inputs, [1, 0, 2])
                    u = output
                flat_xx = flatten(xx, 2)  # [N * M, JX, d]
                if config.share_lstm_weights:
                    tf.get_variable_scope().reuse_variables()
                    with tf.variable_scope("u1"):
                        fw_inputs = tf.transpose(flat_xx, [1, 0, 2]) #[time_len, batch_size, input_size]
                        bw_inputs = tf.reverse_sequence(fw_inputs, flat_x_len, batch_dim=1, seq_dim=0)
                        # fw_inputs = tf.nn.dropout(fw_inputs, config.input_keep_prob)
                        # bw_inputs = tf.nn.dropout(bw_inputs, config.input_keep_prob)
                        fw_outputs, fw_final = prep_fw_cell(fw_inputs, dtype=tf.float32, sequence_length=flat_x_len, scope="fw")
                        bw_outputs, bw_final = prep_bw_cell(bw_inputs, dtype=tf.float32, sequence_length=flat_x_len, scope="bw")
                        bw_outputs = tf.reverse_sequence(bw_outputs, flat_x_len, batch_dim=1, seq_dim=0)
                        current_inputs = tf.concat((fw_outputs, bw_outputs), 2)
                        output = tf.transpose(current_inputs, [1, 0, 2])
                else:
                    with tf.variable_scope("h1"):
                        fw_inputs = tf.transpose(flat_xx, [1, 0, 2]) #[time_len, batch_size, input_size]
                        bw_inputs = tf.reverse_sequence(fw_inputs, flat_x_len, batch_dim=1, seq_dim=0)
                        # fw_inputs = tf.nn.dropout(fw_inputs, config.input_keep_prob)
                        # bw_inputs = tf.nn.dropout(bw_inputs, config.input_keep_prob)
                        prep_fw_cell = LSTMBlockFusedCell(d, cell_clip=0)
                        prep_bw_cell = LSTMBlockFusedCell(d, cell_clip=0)
                        fw_outputs, fw_final = prep_fw_cell(fw_inputs, dtype=tf.float32, sequence_length=flat_x_len, scope="fw")
                        bw_outputs, bw_final = prep_bw_cell(bw_inputs, dtype=tf.float32, sequence_length=flat_x_len, scope="bw")
                        bw_outputs = tf.reverse_sequence(bw_outputs, flat_x_len, batch_dim=1, seq_dim=0)
                        current_inputs = tf.concat((fw_outputs, bw_outputs), 2)
                        output = tf.transpose(current_inputs, [1, 0, 2])
                h = tf.expand_dims(output, 1) # [N, M, JX, 2d]
            else:
                (fw_u, bw_u), _ = bidirectional_dynamic_rnn(d_cell, d_cell, qq, q_len, dtype='float', scope='u1')  # [N, J, d], [N, d]
                u = tf.concat([fw_u, bw_u], 2)
                if config.share_lstm_weights:
                    tf.get_variable_scope().reuse_variables()
                    (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='u1')  # [N, M, JX, 2d]
                    h = tf.concat([fw_h, bw_h], 3)  # [N, M, JX, 2d]
                else:
                    (fw_h, bw_h), _ = bidirectional_dynamic_rnn(cell, cell, xx, x_len, dtype='float', scope='h1')  # [N, M, JX, 2d]
                    h = tf.concat([fw_h, bw_h], 3)  # [N, M, JX, 2d]
            self.tensor_dict['u'] = u
            self.tensor_dict['h'] = h

        # Attention Flow Layer (4th layer on paper)
        with tf.variable_scope("main"):
            if config.dynamic_att:
                p0 = h
                u = tf.reshape(tf.tile(tf.expand_dims(u, 1), [1, M, 1, 1]), [N * M, JQ, 2 * d])
                q_mask = tf.reshape(tf.tile(tf.expand_dims(self.q_mask, 1), [1, M, 1]), [N * M, JQ])
                first_cell = AttentionCell(cell, u, size=d, mask=q_mask, mapper='sim',
                                           input_keep_prob=self.config.input_keep_prob, is_train=self.is_train)
            else:
                p0 = attention_layer(config, self.is_train, h, u, h_mask=self.x_mask, u_mask=self.q_mask, scope="p0", tensor_dict=self.tensor_dict)
                first_cell = d_cell
            tp0 = p0

        # Modeling layer (5th layer on paper)
        with tf.variable_scope('modeling_layer'):
            if config.use_fused_lstm:
                g1, encoder_state_final = build_fused_bidirectional_rnn(inputs=p0,
                                                                        num_units=config.hidden_size,
                                                                        num_layers=config.num_modeling_layers,
                                                                        inputs_length=flat_x_len,
                                                                        input_keep_prob=config.input_keep_prob,
                                                                        scope='modeling_layer_g')

            else:
                for layer_idx in range(config.num_modeling_layers-1):
                    (fw_g0, bw_g0), _ = bidirectional_dynamic_rnn(first_cell, first_cell, p0, x_len,
                                                                  dtype='float', scope="g_{}".format(layer_idx))  # [N, M, JX, 2d]
                    p0 = tf.concat([fw_g0, bw_g0], 3)
                (fw_g1, bw_g1), (fw_s_f, bw_s_f) = bidirectional_dynamic_rnn(first_cell, first_cell, p0, x_len,
                                                                             dtype='float', scope='g1')  # [N, M, JX, 2d]
                g1 = tf.concat([fw_g1, bw_g1], 3)  # [N, M, JX, 2d]

        # Self match layer
        if config.use_self_match:
            s0 = tf.reshape(g1, [N * M, JX, 2 * d])                     # [N * M, JX, 2d]
            x_mask = tf.reshape(self.x_mask, [N * M, JX])               # [N * M, JX]
            if config.use_static_self_match:
                with tf.variable_scope("StaticSelfMatch"):              # implemented follow r-net section 3.3
                    W_x_Vj = tf.contrib.layers.fully_connected(         # [N * M, JX, d]
                        s0, int(d / 2), scope='row_first',
                        activation_fn=None, biases_initializer=None
                    )
                    W_x_Vt = tf.contrib.layers.fully_connected(         # [N * M, JX, d]
                        s0, int(d / 2), scope='col_first',
                        activation_fn=None, biases_initializer=None
                    )
                    sum_rc = tf.add(                                    # [N * M, JX, JX, d]
                        tf.expand_dims(W_x_Vj, 1),
                        tf.expand_dims(W_x_Vt, 2)
                    )
                    v = tf.get_variable('second', shape=[1, 1, 1, int(d / 2)], dtype=tf.float32)
                    Sj = tf.reduce_sum(tf.multiply(v, tf.tanh(sum_rc)), -1)     # [N * M, JX, JX]
                    Ai = softmax(Sj, mask = tf.expand_dims(x_mask, 1))          # [N * M, JX, JX]
                    Ai = tf.expand_dims(Ai, -1)                                 # [N * M, JX, JX, 1]
                    Vi = tf.expand_dims(s0, 1)                                  # [N * M, 1, JX, 2d]
                    Ct = tf.reduce_sum(                                         # [N * M, JX, 2d]
                        tf.multiply(Ai, Vi),
                        axis = 2
                    )
                    inputs_Vt_Ct = tf.concat([s0, Ct], 2)                       # [N * M, JX, 4d]
                    if config.use_fused_lstm:
                        fw_inputs = tf.transpose(inputs_Vt_Ct, [1, 0, 2])  # [time_len, batch_size, input_size]
                        bw_inputs = tf.reverse_sequence(fw_inputs, flat_x_len, batch_dim=1, seq_dim=0)
                        fw_inputs = tf.nn.dropout(fw_inputs, config.input_keep_prob)
                        bw_inputs = tf.nn.dropout(bw_inputs, config.input_keep_prob)
                        prep_fw_cell = LSTMBlockFusedCell(d, cell_clip=0)
                        prep_bw_cell = LSTMBlockFusedCell(d, cell_clip=0)
                        fw_outputs, fw_s_f = prep_fw_cell(fw_inputs, dtype=tf.float32, sequence_length=flat_x_len,
                                                            scope="fw")
                        bw_outputs, bw_s_f = prep_bw_cell(bw_inputs, dtype=tf.float32, sequence_length=flat_x_len,
                                                            scope="bw")
                        fw_s_f = LSTMStateTuple(c=fw_s_f[0], h=fw_s_f[1])
                        bw_s_f = LSTMStateTuple(c=bw_s_f[0], h=bw_s_f[1])
                        bw_outputs = tf.reverse_sequence(bw_outputs, flat_x_len, batch_dim=1, seq_dim=0)
                        current_inputs = tf.concat((fw_outputs, bw_outputs), 2)
                        s1 = tf.transpose(current_inputs, [1, 0, 2])
                    else:
                        (fw_s, bw_s), (fw_s_f, bw_s_f) = bidirectional_dynamic_rnn(first_cell, first_cell, inputs_Vt_Ct,
                                                                                   flat_x_len, dtype='float',
                                                                                   scope='s')  # [N, M, JX, 2d]
                        s1 = tf.concat([fw_s, bw_s], 2)  # [N * M, JX, 2d], M == 1
            else:
                with tf.variable_scope("DynamicSelfMatch"):
                    first_cell = AttentionCell(cell, s0, size=d, mask=x_mask, is_train=self.is_train)
                    (fw_s, bw_s), (fw_s_f, bw_s_f) = bidirectional_dynamic_rnn(first_cell, first_cell, s0, x_len,
                                                                               dtype='float', scope='s')  # [N, M, JX, 2d]
                    s1 = tf.concat([fw_s, bw_s], 2)  # [N * M, JX, 2d], M == 1
            g1 = tf.expand_dims(s1, 1) # [N, M, JX, 2d]

        # prepare for PtrNet
        encoder_output = g1  # [N, M, JX, 2d]
        encoder_output = tf.expand_dims(tf.cast(self.x_mask, tf.float32), -1) * encoder_output  # [N, M, JX, 2d]

        if config.use_self_match or not config.use_fused_lstm:
            if config.GRU:
                encoder_state_final = tf.concat((fw_s_f, bw_s_f), 1, name='encoder_concat')
            else:
                if isinstance(fw_s_f, LSTMStateTuple):
                    encoder_state_c = tf.concat(
                        (fw_s_f.c, bw_s_f.c), 1, name='encoder_concat_c')
                    encoder_state_h = tf.concat(
                        (fw_s_f.h, bw_s_f.h), 1, name='encoder_concat_h')
                    encoder_state_final = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)
                elif isinstance(fw_s_f, tf.Tensor):
                    encoder_state_final = tf.concat((fw_s_f, bw_s_f), 1, name='encoder_concat')
                else:
                    encoder_state_final = None
                    tf.logging.error("encoder_state_final not set")

        print("encoder_state_final:", encoder_state_final)

        with tf.variable_scope("output"):
            # eos_symbol = config.eos_symbol
            # next_symbol = config.next_symbol

            tf.assert_equal(M, 1)  # currently dynamic M is not supported, thus we assume M==1
            answer_string = tf.placeholder(
                shape=(N, 1, JA + 1),
                dtype=tf.int32,
                name='answer_string'
            )  # [N, M, JA + 1]
            answer_string_mask = tf.placeholder(
                shape=(N, 1, JA + 1),
                dtype=tf.bool,
                name='answer_string_mask'
            )  # [N, M, JA + 1]
            answer_string_length = tf.placeholder(
                shape=(N, 1),
                dtype=tf.int32,
                name='answer_string_length',
            ) # [N, M]
            self.tensor_dict['answer_string'] = answer_string
            self.tensor_dict['answer_string_mask'] = answer_string_mask
            self.tensor_dict['answer_string_length'] = answer_string_length
            self.answer_string = answer_string
            self.answer_string_mask = answer_string_mask
            self.answer_string_length = answer_string_length

            answer_string_flattened = tf.reshape(answer_string, [N * M, JA + 1])
            self.answer_string_flattened = answer_string_flattened  # [N * M, JA+1]
            print("answer_string_flattened:", answer_string_flattened)

            answer_string_length_flattened = tf.reshape(answer_string_length, [N * M])
            self.answer_string_length_flattened = answer_string_length_flattened  # [N * M]
            print("answer_string_length_flattened:", answer_string_length_flattened)

            decoder_cell = GRUCell(2 * d) if config.GRU else BasicLSTMCell(2 * d, state_is_tuple=True)

            with tf.variable_scope("Decoder"):
                decoder_train_logits = ptr_decoder(decoder_cell,
                                                   tf.reshape(tp0, [N * M, JX, 2 * d]),  # [N * M, JX, 2d]
                                                   tf.reshape(encoder_output, [N * M, JX, 2 * d]),  # [N * M, JX, 2d]
                                                   flat_x_len,
                                                   encoder_final_state=encoder_state_final,
                                                   max_encoder_length=config.sent_size_th,
                                                   decoder_output_length=answer_string_length_flattened,  # [N * M]
                                                   batch_size=N,  # N * M (M=1)
                                                   attention_proj_dim=self.config.decoder_proj_dim,
                                                   scope='ptr_decoder')  # [batch_size, dec_len*, enc_seq_len + 1]

                self.decoder_train_logits = decoder_train_logits
                print("decoder_train_logits:", decoder_train_logits)
                self.decoder_train_softmax = tf.nn.softmax(self.decoder_train_logits)
                self.decoder_inference = tf.argmax(decoder_train_logits, axis=2,
                                                   name='decoder_inference')  # [N, JA + 1]

            self.yp = tf.ones([N, M, JX], dtype=tf.int32) * -1
            self.yp2 = tf.ones([N, M, JX], dtype=tf.int32) * -1

            # logits = get_logits([g1, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
            #                     mask=self.x_mask, is_train=self.is_train, func=config.answer_func, scope='logits1')
            # a1i = softsel(tf.reshape(g1, [N, M * JX, 2 * d]), tf.reshape(logits, [N, M * JX]))
            # a1i = tf.tile(tf.expand_dims(tf.expand_dims(a1i, 1), 1), [1, M, JX, 1])
            #
            # (fw_g2, bw_g2), _ = bidirectional_dynamic_rnn(d_cell, d_cell, tf.concat([p0, g1, a1i, g1 * a1i], 3),
            #                                               x_len, dtype='float', scope='g2')  # [N, M, JX, 2d]
            # g2 = tf.concat([fw_g2, bw_g2], 3)
            # logits2 = get_logits([g2, p0], d, True, wd=config.wd, input_keep_prob=config.input_keep_prob,
            #                      mask=self.x_mask,
            #                      is_train=self.is_train, func=config.answer_func, scope='logits2')
            #
            # flat_logits = tf.reshape(logits, [-1, M * JX])
            # flat_yp = tf.nn.softmax(flat_logits)  # [-1, M*JX]
            # yp = tf.reshape(flat_yp, [-1, M, JX])
            # flat_logits2 = tf.reshape(logits2, [-1, M * JX])
            # flat_yp2 = tf.nn.softmax(flat_logits2)
            # yp2 = tf.reshape(flat_yp2, [-1, M, JX])
            #
            # self.tensor_dict['g1'] = g1
            # self.tensor_dict['g2'] = g2
            #
            # self.logits = flat_logits
            # self.logits2 = flat_logits2
            # self.yp = yp
            # self.yp2 = yp2

    def _build_loss(self):

        N = self.config.batch_size
        JX = tf.shape(self.x)[2]
        M = tf.shape(self.x)[1]
        JQ = tf.shape(self.q)[1]

        logits = self.decoder_train_logits[
                 :,
                 :tf.reduce_max(self.answer_string_length_flattened),
                 :]  # [N * M, JX, JX + 1] -> [N * M, enc_seq_len + 1, JX + 1]
        targets = self.answer_string_flattened[:, :tf.shape(logits)[1]]  # [N * M, JA+1] -> [N * M, JX + 1]

        print("logits:", logits, "targets:", targets)

        logits = tf.Print(logits, [tf.shape(logits), tf.argmax(logits, 2)], 'logits: ', summarize=100)
        targets = tf.Print(targets, [targets], 'targets: ', summarize=100)

        self.logits = logits
        self.targets = targets

        weights_mask = tf.reshape(self.answer_string_mask[:,:,:tf.shape(logits)[1]], [N * 1, tf.shape(logits)[1]])  # [N * M] -> [N * M, JX + 1]
        decoder_loss = tf.losses.sparse_softmax_cross_entropy(logits=logits,
                                                              labels=targets,
                                                              weights=weights_mask,
                                                              loss_collection=None)
        tf.add_to_collection(tf.GraphKeys.LOSSES, decoder_loss)
        self.decoder_loss = decoder_loss

        # loss_mask = tf.reduce_max(tf.cast(self.q_mask, 'float'), 1)
        # losses = tf.nn.softmax_cross_entropy_with_logits(
        #     logits=self.logits, labels=tf.cast(tf.reshape(self.y, [-1, M * JX]), 'float'))
        # ce_loss = tf.reduce_mean(loss_mask * losses)
        # tf.add_to_collection(tf.GraphKeys.LOSSES, ce_loss)
        # ce_loss2 = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        #     logits=self.logits2, labels=tf.cast(tf.reshape(self.y2, [-1, M * JX]), 'float')))
        # tf.add_to_collection(tf.GraphKeys.LOSSES, ce_loss2)

        self.loss = tf.add_n(tf.get_collection(tf.GraphKeys.LOSSES, scope=self.scope), name='loss')
        tf.summary.scalar(self.loss.op.name, self.loss)
        tf.add_to_collection('ema/scalar', self.loss)

    def _build_ema(self):
        self.ema = tf.train.ExponentialMovingAverage(self.config.decay)
        ema = self.ema
        tensors = tf.get_collection("ema/scalar", scope=self.scope) + tf.get_collection("ema/vector", scope=self.scope)
        ema_op = ema.apply(tensors)
        for var in tf.get_collection("ema/scalar", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.scalar(ema_var.op.name, ema_var)
        for var in tf.get_collection("ema/vector", scope=self.scope):
            ema_var = ema.average(var)
            tf.summary.histogram(ema_var.op.name, ema_var)

        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def _build_var_ema(self):
        self.var_ema = tf.train.ExponentialMovingAverage(self.config.var_decay)
        ema = self.var_ema
        ema_op = ema.apply(tf.trainable_variables())
        with tf.control_dependencies([ema_op]):
            self.loss = tf.identity(self.loss)

    def get_loss(self):
        return self.loss

    def get_global_step(self):
        return self.global_step

    def get_feed_dict(self, batch, is_train, supervised=True):
        assert isinstance(batch, DataSet)
        config = self.config
        N, M, JX, JQ, VW, VC, d, W = \
            config.batch_size, config.max_num_sents, config.max_sent_size, \
            config.max_ques_size, config.word_vocab_size, config.char_vocab_size, config.hidden_size, config.max_word_size
        feed_dict = {}

        if config.len_opt:
            """
            Note that this optimization results in variable GPU RAM usage (i.e. can cause OOM in the middle of training.)
            First test without len_opt and make sure no OOM, and use len_opt
            """
            if sum(len(sent) for para in batch.data['x'] for sent in para) == 0:
                new_JX = 1
            else:
                new_JX = max(len(sent) for para in batch.data['x'] for sent in para)
            JX = min(JX, new_JX)

            if sum(len(ques) for ques in batch.data['q']) == 0:
                new_JQ = 1
            else:
                new_JQ = max(len(ques) for ques in batch.data['q'])
            JQ = min(JQ, new_JQ)

        if config.cpu_opt:
            if sum(len(para) for para in batch.data['x']) == 0:
                new_M = 1
            else:
                new_M = max(len(para) for para in batch.data['x'])
            M = min(M, new_M)
        tf.logging.info("M: %s, JX: %s, JQ: %s", str(M), str(JX), str(JQ))

        x = np.zeros([N, M, JX], dtype='int32')
        cx = np.zeros([N, M, JX, W], dtype='int32')
        x_mask = np.zeros([N, M, JX], dtype='bool')
        q = np.zeros([N, JQ], dtype='int32')
        cq = np.zeros([N, JQ, W], dtype='int32')
        q_mask = np.zeros([N, JQ], dtype='bool')

        feed_dict[self.x] = x
        feed_dict[self.x_mask] = x_mask
        feed_dict[self.cx] = cx
        feed_dict[self.q] = q
        feed_dict[self.cq] = cq
        feed_dict[self.q_mask] = q_mask
        feed_dict[self.is_train] = is_train
        if config.use_glove_for_unk:
            feed_dict[self.new_emb_mat] = batch.shared['new_emb_mat']
        if config.use_exact_match:
            emx = np.zeros([N, M, JX], dtype='int32')
            emq = np.zeros([N, JQ], dtype='int32')
            feed_dict[self.emx] = emx
            feed_dict[self.emq] = emq

        X = batch.data['x']
        CX = batch.data['cx']

        def _get_word(word):
            d = batch.shared['word2idx']
            for each in (word, word.lower(), word.capitalize(), word.upper()):
                if each in d:
                    return d[each]
            if config.use_glove_for_unk:
                d2 = batch.shared['new_word2idx']
                for each in (word, word.lower(), word.capitalize(), word.upper()):
                    if each in d2:
                        return d2[each] + len(d)
            return 1

        if supervised:
            JA = config.max_answer_length
            answer_string = np.zeros([N, M, JA + 1], dtype='int32')
            answer_string_mask = np.zeros([N, M, JA + 1], dtype='bool')
            answer_string_length = np.zeros([N, M], dtype='int32')

            for i, (xi, cxi, yi) in enumerate(zip(X, CX, batch.data['y'])):
                if config.data_type == 'span':
                    start_idx, stop_idx = random.choice(yi)
                    j, k = start_idx
                    j2, k2 = stop_idx
                    k = min(k, config.sent_size_th)
                    k2 = min(k2, config.sent_size_th)
                    assert j == j2, (j, j2)
                    assert k2 - k <= JA, (k2 - k, JA)

                    for t, k_current in enumerate(range(k, k2)):
                        answer_string[i, j, t] = k_current
                    answer_string[i, j, k2 - k] = min(len(xi[j]), config.sent_size_th)
                    answer_string_length[i, j] = k2 - k + 1
                    answer_string_mask[i, j, :k2 - k + 1] = True

                elif config.data_type == 'seq':
                    idx_list = random.choice(yi)
                    j = idx_list[0][0]
                    assert len(idx_list) <= JA, (len(idx_list), JA)

                    for t, idx in enumerate(idx_list):
                        j1, k1 = idx
                        k1 = min(k1, config.sent_size_th)
                        assert j == j1, (j, j1)
                        answer_string[i, j, t] = k1
                    answer_string[i, j, len(idx_list)] = min(len(xi[j]), config.sent_size_th)
                    answer_string_length[i, j] = len(idx_list) + 1
                    answer_string_mask[i, j, :len(idx_list) + 1] = True


            feed_dict[self.answer_string] = answer_string
            feed_dict[self.answer_string_mask] = answer_string_mask
            feed_dict[self.answer_string_length] = answer_string_length

            # y = np.zeros([N, M, JX], dtype='bool')
            # y2 = np.zeros([N, M, JX], dtype='bool')
            # feed_dict[self.y] = y
            # feed_dict[self.y2] = y2
            #
            # for i, (xi, cxi, yi) in enumerate(zip(X, CX, batch.data['y'])):
            #     start_idx, stop_idx = random.choice(yi)
            #     j, k = start_idx
            #     j2, k2 = stop_idx
            #     if config.single:
            #         X[i] = [xi[j]]
            #         CX[i] = [cxi[j]]
            #         j, j2 = 0, 0
            #     if config.squash:
            #         offset = sum(map(len, xi[:j]))
            #         j, k = 0, k + offset
            #         offset = sum(map(len, xi[:j2]))
            #         j2, k2 = 0, k2 + offset
            #     y[i, j, k] = True
            #     y2[i, j2, k2-1] = True
        if not is_train:
            answer_string = np.zeros([N, M, config.max_answer_length + 1], dtype='int32')
            answer_string_mask = np.ones([N, M, config.max_answer_length + 1], dtype='bool')
            answer_string_length = np.ones([N, M], dtype='int32') * config.max_answer_length

            feed_dict[self.answer_string] = answer_string
            feed_dict[self.answer_string_mask] = answer_string_mask
            feed_dict[self.answer_string_length] = answer_string_length

        def _get_char(char):
            d = batch.shared['char2idx']
            if char in d:
                return d[char]
            return 1

        for i, xi in enumerate(X):
            if self.config.squash:
                xi = [list(itertools.chain(*xi))]
            for j, xij in enumerate(xi):
                if j == config.max_num_sents:
                    break
                for k, xijk in enumerate(xij):
                    if k == config.max_sent_size:
                        break
                    each = _get_word(xijk)
                    assert isinstance(each, int), each
                    x[i, j, k] = each
                    x_mask[i, j, k] = True

        for i, cxi in enumerate(CX):
            if self.config.squash:
                cxi = [list(itertools.chain(*cxi))]
            for j, cxij in enumerate(cxi):
                if j == config.max_num_sents:
                    break
                for k, cxijk in enumerate(cxij):
                    if k == config.max_sent_size:
                        break
                    for l, cxijkl in enumerate(cxijk):
                        if l == config.max_word_size:
                            break
                        cx[i, j, k, l] = _get_char(cxijkl)

        for i, qi in enumerate(batch.data['q']):
            for j, qij in enumerate(qi):
                q[i, j] = _get_word(qij)
                q_mask[i, j] = True
                if config.use_exact_match:
                    emq[i, j] = 1 if q[i, j] in x[i, 0] else 0 # exact match feature of query

        for i, cqi in enumerate(batch.data['cq']):
            for j, cqij in enumerate(cqi):
                for k, cqijk in enumerate(cqij):
                    cq[i, j, k] = _get_char(cqijk)
                    if k + 1 == config.max_word_size:
                        break

        if config.use_exact_match:
            for i, xi in enumerate(X):
                if self.config.squash:
                    xi = [list(itertools.chain(*xi))]
                for j, xij in enumerate(xi):
                    if j == config.max_num_sents:
                        break
                    for k, xijk in enumerate(xij):
                        if k == config.max_sent_size:
                            break
                        emx[i, j, k] = 1 if x[i, j, k] in q[i] else 0

        return feed_dict


def bi_attention(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "bi_attention"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        h_aug = tf.tile(tf.expand_dims(h, 3), [1, 1, 1, JQ, 1])
        u_aug = tf.tile(tf.expand_dims(tf.expand_dims(u, 1), 1), [1, M, JX, 1, 1])
        if h_mask is None:
            hu_mask = None
        else:
            h_mask_aug = tf.tile(tf.expand_dims(h_mask, 3), [1, 1, 1, JQ])
            u_mask_aug = tf.tile(tf.expand_dims(tf.expand_dims(u_mask, 1), 1), [1, M, JX, 1])
            hu_mask = h_mask_aug & u_mask_aug

        u_logits = get_logits([h_aug, u_aug], None, True, wd=config.wd, mask=hu_mask,
                              is_train=is_train, func=config.logit_func, scope='u_logits')  # [N, M, JX, JQ]
        u_a = softsel(u_aug, u_logits)  # [N, M, JX, d]
        h_a = softsel(h, tf.reduce_max(u_logits, 3))  # [N, M, d]
        h_a = tf.tile(tf.expand_dims(h_a, 2), [1, 1, JX, 1])

        if tensor_dict is not None:
            a_u = tf.nn.softmax(u_logits)  # [N, M, JX, JQ]
            a_h = tf.nn.softmax(tf.reduce_max(u_logits, 3))
            tensor_dict['a_u'] = a_u
            tensor_dict['a_h'] = a_h
            variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=tf.get_variable_scope().name)
            for var in variables:
                tensor_dict[var.name] = var

        return u_a, h_a


def attention_layer(config, is_train, h, u, h_mask=None, u_mask=None, scope=None, tensor_dict=None):
    with tf.variable_scope(scope or "attention_layer"):
        JX = tf.shape(h)[2]
        M = tf.shape(h)[1]
        JQ = tf.shape(u)[1]
        if config.q2c_att or config.c2q_att:
            u_a, h_a = bi_attention(config, is_train, h, u, h_mask=h_mask, u_mask=u_mask, tensor_dict=tensor_dict)
        if not config.c2q_att:
            u_a = tf.tile(tf.expand_dims(tf.expand_dims(tf.reduce_mean(u, 1), 1), 1), [1, M, JX, 1])
        if config.q2c_att:
            p0 = tf.concat([h, u_a, h * u_a, h * h_a], 3)
        else:
            p0 = tf.concat([h, u_a, h * u_a], 3)
        return p0

def build_fused_bidirectional_rnn(inputs, num_units, num_layers, inputs_length, input_keep_prob=1.0, scope=None, dtype=tf.float32):
    ''' The input of sequences should be time major
        And the Dropout is independent per time. '''
    assert num_layers > 0
    with tf.variable_scope(scope or "bidirectional_rnn"):
        inputs = flatten(inputs, 2) # [N * M, JX, d]
        current_inputs = tf.transpose(inputs, [1, 0, 2]) #[time_len, batch_size, input_size]
        for layer_id in range(num_layers):
            reverse_inputs = tf.reverse_sequence(current_inputs, inputs_length, batch_dim=1, seq_dim=0)
            fw_inputs = tf.nn.dropout(current_inputs, input_keep_prob)
            bw_inputs = tf.nn.dropout(reverse_inputs, input_keep_prob)
            fw_cell = LSTMBlockFusedCell(num_units, cell_clip=0)
            bw_cell = LSTMBlockFusedCell(num_units, cell_clip=0)
            fw_outputs, fw_final = fw_cell(fw_inputs, dtype=dtype, sequence_length=inputs_length, scope="fw_"+str(layer_id))
            bw_outputs, bw_final = bw_cell(bw_inputs, dtype=dtype, sequence_length=inputs_length, scope="bw_"+str(layer_id))
            bw_outputs = tf.reverse_sequence(bw_outputs, inputs_length, batch_dim=1, seq_dim=0)
            current_inputs = tf.concat((fw_outputs, bw_outputs), 2)
        output = tf.transpose(current_inputs, [1, 0, 2])
        output = tf.expand_dims(output, 1) # [N, M, JX, 2d]

        final_state_c = tf.concat((fw_final[0], bw_final[0]), 1, name=scope + '_final_c')
        final_state_h = tf.concat((fw_final[1], bw_final[1]), 1, name=scope + '_final_h')
        final_state = LSTMStateTuple(c=final_state_c, h=final_state_h)  # ([N, 2 * d], [N, 2 * d])
        return output, final_state