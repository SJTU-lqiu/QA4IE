import tensorflow as tf
from tensorflow.contrib.rnn import BasicLSTMCell, LSTMStateTuple
from tensorflow.contrib.cudnn_rnn.python.ops.cudnn_rnn_ops import CudnnLSTM
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import array_ops
import numpy as np

from datetime import datetime


def ptr_decoder(cell,
                encoder_inputs,
                encoder_outputs,
                encoder_inputs_length,
                encoder_final_state,
                max_encoder_length,
                decoder_output_length=None,
                batch_size=None,
                attention_proj_dim=None,
                time_major=False,
                scope=None):
    """

    :param cell:
    :param encoder_inputs: [batch_size, enc_seq_len, enc_input_size]
    :param encoder_outputs: [batch_size, enc_seq_len, enc_hid_size]
    :param encoder_final_state: [batch_size, enc_hid_size]
    :param max_encoder_length: const int32, max_encoder_length >= enc_seq_len
    :param decoder_output_length: [batch_size] dtype=int32, None for inference
    :param batch_size: const int32
    :param attention_proj_dim: const int32
    :param time_major:
    :param scope:
    :return: decoder_outputs: [batch_size, dec_seq_len*, enc_seq_len + 1]
    """

    with vs.variable_scope(scope or "ptr_decoder", [cell, encoder_inputs, encoder_outputs,
                                                    encoder_final_state, max_encoder_length, decoder_output_length,
                                                    time_major]):
        assert len(encoder_outputs.get_shape()) == 3

        if time_major:
            encoder_inputs = tf.transpose(encoder_inputs, [1, 0, 2])
            encoder_outputs = tf.transpose(encoder_outputs, [1, 0, 2])  # [batch_size, enc_seq_len, enc_hid_size]

        enc_seq_len = tf.shape(encoder_outputs)[1]
        batch_size = batch_size or encoder_outputs.get_shape()[0].value
        tf.assert_equal(batch_size, tf.shape(encoder_outputs)[0])
        tf.assert_equal(batch_size, tf.shape(encoder_inputs)[0])
        enc_hid_size = encoder_outputs.get_shape()[2].value
        # enc_input_size = encoder_inputs.get_shape()[2].value
        dec_hid_size = cell.output_size
        attention_proj_dim = attention_proj_dim or dec_hid_size

        eos_vector = vs.get_variable("eos_vector", [1, enc_hid_size], tf.float32)

        # eos_slice = tf.tile(eos_vector, [batch_size, 1, 1])
        # encoder_outputs = tf.concat([encoder_outputs, eos_slice], 1)

        for i in range(batch_size):
            slice_with_eos = tf.concat( # [enc_seq_len + 1, enc_hid_size]
                [
                    encoder_outputs[i, :encoder_inputs_length[i]],
                    eos_vector,
                    encoder_outputs[i, encoder_inputs_length[i]:]
                ],
                0
            )
            slice_with_eos = tf.expand_dims(slice_with_eos, 0) # [1, enc_seq_len + 1, enc_hid_size]
            if i == 0:
                temp = slice_with_eos
            else:
                temp = tf.concat([temp, slice_with_eos], 0) # [batch_size, enc_seq_len + 1, enc_hid_size]
        encoder_outputs = temp

        # encoder_inputs = tf.concat([encoder_inputs, tf.ones([batch_size, 1, enc_input_size], dtype=tf.float32) * 0], 1)

        if decoder_output_length is None:
            decoder_output_length = tf.ones([batch_size], dtype=tf.int32) * (enc_seq_len + 1)

        def loop_fn_initial():
            elements_finished = (decoder_output_length <= 0)
            initial_input = tf.zeros([batch_size, enc_hid_size], dtype=tf.float32)
            initial_cell_state = encoder_final_state
            initial_cell_output = tf.zeros(max_encoder_length + 1)
            print("initial_cell_output", initial_cell_output)
            initial_loop_state = None

            return (elements_finished,
                    initial_input,
                    initial_cell_state,
                    initial_cell_output,
                    initial_loop_state)

        def loop_fn_transition(time, cell_output, cell_state, previous_loop_state):  # cell_output <- [batch_size, dec_hid_size]
            def get_attention():
                with tf.variable_scope('CalcAttention') as scope:
                    w_x_d = tf.contrib.layers.fully_connected(
                        cell_output, attention_proj_dim,
                        scope='project_decoder',
                        activation_fn=None,
                        biases_initializer=tf.random_normal_initializer())  # Wd*d [batch_size, attention_proj_dim]
                    w_x_e = tf.contrib.layers.fully_connected(
                        encoder_outputs, attention_proj_dim,
                        scope='project_encoder',
                        activation_fn=None)  # We*e [batch_size, enc_seq_len + 1, attention_proj_dim]
                    print("w_x_e", w_x_e)
                    print("enc_seq_len", enc_seq_len)
                    w_x_d_ext = tf.expand_dims(w_x_d, 1, name='expand_decoder')  # [batch_size, 1, attention_proj_dim]
                    w_x_d_ext = tf.tile(w_x_d_ext, [1, enc_seq_len + 1, 1], name='tile_decoder')  # [batch_size, enc_seq_len + 1, attention_proj_dim]
                    attention_tanh = tf.tanh(w_x_e + w_x_d_ext, name='attention_tanh')  # [batch_size, enc_seq_len + 1, attention_proj_dim]
                    print("attention_tanh", attention_tanh)
                    a = tf.contrib.layers.fully_connected(
                        attention_tanh, 1,
                        activation_fn=None,
                        scope='calc_attention_a',
                        biases_initializer=tf.random_normal_initializer())  # [batch_size, enc_seq_len + 1, 1]
                    print("a", a)
                    a = tf.reshape(a,
                                   [batch_size, enc_seq_len + 1],
                                   name='reshape_attention_a')  # [batch_size, enc_seq_len + 1]
                return a

            elements_finished = (time >= decoder_output_length)  # [batch_size]
            finished = tf.reduce_all(elements_finished)
            next_cell_state = cell_state
            next_loop_state = None

            attention = get_attention()  # [batch_size, enc_seq_len + 1]
            mask = -1e30 * (1.0 - tf.concat( # [batch_size, enc_seq_len + 1]
                [tf.ones([batch_size, 1]),
                array_ops.sequence_mask(
                    encoder_inputs_length,
                    enc_seq_len,
                    dtype=tf.float32,
                    name="attention_mask")],
                1
            ))
            attention += mask
            softmax = tf.nn.softmax(attention, name='attention_softmax')
            print("softmax", softmax)
            softmax_ext = tf.expand_dims(softmax, 1)  # [batch_size, 1, enc_seq_len + 1]
            attended_input = tf.squeeze(tf.matmul(softmax_ext, encoder_outputs), axis=1)
            print("attended_input", attended_input)
            next_cell_input = tf.cond(finished,
                                     lambda: tf.zeros([batch_size, enc_hid_size]),
                                     lambda: attended_input)
            emit_output = tf.cond(enc_seq_len > max_encoder_length,
                                  lambda: attention[:,:max_encoder_length + 1],
                                  lambda: tf.pad(attention, [[0, 0], [0, max_encoder_length - enc_seq_len]])) # [batch_size, max_encoder_length + 1]
            print("elements_finished", elements_finished)
            print("next_cell_input", next_cell_input)
            print("next_cell_state", next_cell_state)
            print("emit_output", emit_output)
            return elements_finished, next_cell_input, next_cell_state, emit_output, next_loop_state

        def loop_fn(time, cell_output, cell_state, previous_loop_state):
            with tf.variable_scope("ptr_net_loop_fn"):
                if cell_state is None:  # time == 0
                    assert cell_output is None and cell_state is None
                    return loop_fn_initial()
                else:
                    tf.get_variable_scope().reuse_variables()
                    return loop_fn_transition(time, cell_output, cell_state, previous_loop_state)

        decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(cell, loop_fn)
        decoder_outputs = decoder_outputs_ta.stack()  # [max(decoder_output_length), batch_size, max_encoder_length]

        if not time_major:
            decoder_outputs = tf.transpose(decoder_outputs, [1, 0, 2])  # [batch_size, max(decoder_output_length), max_encoder_length]
        decoder_outputs = decoder_outputs[:, :, :enc_seq_len + 1]  # keep eos symbol [batch_size, max(decoder_output_length), enc_seq_len + 1]
    return decoder_outputs

def build_basic_encoder(encoder_cell, num_layers, encoder_inputs, encoder_inputs_length, scope=None):
    assert num_layers > 0
    with tf.variable_scope(scope or 'basic_encoder'):
        current_inputs = encoder_inputs
        for layer_id in range(num_layers):
            (encoder_outputs, encoder_state) = (
                tf.nn.dynamic_rnn(cell=encoder_cell,
                                  inputs=current_inputs,
                                  sequence_length=encoder_inputs_length,
                                  time_major=False,
                                  dtype=tf.float32,
                                  scope='encoder_l' + str(layer_id))
            )
            current_inputs = encoder_outputs

    return encoder_outputs, encoder_state

def build_bidirectional_encoder(encoder_cell, num_layers, encoder_inputs, encoder_inputs_length, scope=None):
    assert num_layers > 0
    with tf.variable_scope(scope or 'basic_encoder'):
        current_inputs = encoder_inputs
        for layer_id in range(num_layers):
            ((encoder_fw_outputs, encoder_bw_outputs),
             (encoder_fw_state, encoder_bw_state)) = (
                tf.nn.bidirectional_dynamic_rnn(cell_fw=encoder_cell,
                                                cell_bw=encoder_cell,
                                                inputs=current_inputs,
                                                sequence_length=encoder_inputs_length,
                                                time_major=False,
                                                dtype=tf.float32,
                                                scope='encoder_l' + str(layer_id))
            )
            encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs),
                                    2)  # [batch_size, enc_seq_len, 2 * enc_hid_size]
            current_inputs = encoder_outputs

        encoder_state_c = tf.concat(
            (encoder_fw_state.c, encoder_bw_state.c), 1, name='bidirectional_concat_c')
        encoder_state_h = tf.concat(
            (encoder_fw_state.h, encoder_bw_state.h), 1, name='bidirectional_concat_h')
        encoder_final_state = LSTMStateTuple(c=encoder_state_c, h=encoder_state_h)  # [batch_size, enc_hid_size*2] for c,h

    return encoder_outputs, encoder_final_state

def calc_cudnn_num_params(num_layers, num_units, input_size):
    input_size = int(input_size)
    num_wh = num_units * num_units * 4 * num_layers
    num_wx = num_units * num_units * 4 * (num_layers - 1) + num_units * input_size * 4
    num_b = num_units * 4 * num_layers
    return num_wh + num_wx + num_b * 2

def build_cudnn_encoder(encoder_cell, num_layers, encoder_inputs, encoder_inputs_length, time_major=False, scope=None):
    assert num_layers > 0
    batch_size = encoder_inputs.get_shape()[0]
    num_units = encoder_cell.output_size
    input_size = encoder_inputs.get_shape()[-1]
    with tf.variable_scope(scope):
        model = CudnnLSTM(num_layers, num_units, input_size)
        params_size_t = calc_cudnn_num_params(num_layers, num_units, input_size)
        #params_size_t = model.params_size()
        if not time_major:
            encoder_inputs = tf.transpose(encoder_inputs, [1, 0, 2])
        input_h = tf.zeros([num_layers, batch_size, num_units])
        input_c = tf.zeros([num_layers, batch_size, num_units])
        params = tf.Variable(
            tf.random_normal([params_size_t]))
        output, output_h, output_c = model(
            is_training=True,
            input_data=encoder_inputs,
            input_h=input_h,
            input_c=input_c,
            params=params)
        print("output", output)
        print("output_h", output_h)

        if not time_major:
            output = tf.transpose(output, [1, 0, 2])

    return output, LSTMStateTuple(c=output_c[0], h=output_h[0])

def test():
    from tensorflow.python import debug as tf_debug

    use_cudnn_rnn = False
    bidirectional = False
    num_encoder_layers = 2

    encoder_input_size = 2
    max_seq_length = 8
    encoder_hidden_size = 40
    decoder_hidden_size = encoder_hidden_size * (2 if bidirectional else 1)
    encoder_cell = BasicLSTMCell(encoder_hidden_size)
    decoder_cell = BasicLSTMCell(decoder_hidden_size)
    batch_size = 32

    epochs = 1000000

    encoder_inputs = tf.placeholder(
        dtype=tf.float32,
        shape=[batch_size, max_seq_length, encoder_input_size],
        name='encoder_inputs',
    )
    encoder_inputs_length = tf.placeholder(
        dtype=tf.int32,
        shape=[batch_size],
        name='encoder_inputs_length',
    )

    # required for training, not required for testing
    decoder_targets = tf.placeholder(
        dtype=tf.int32,
        shape=[batch_size, None],
        name='decoder_targets'
    )
    decoder_targets_length = tf.placeholder(
        dtype=tf.int32,
        shape=[batch_size],
        name='decoder_targets_length',
    )

    with tf.variable_scope("BidirectionalEncoder") as scope:
        if use_cudnn_rnn:
            encoder_outputs, encoder_final_state = build_cudnn_encoder(
                encoder_cell, num_encoder_layers, encoder_inputs, encoder_inputs_length, scope=scope
            )
        else:
            if bidirectional:
                encoder_outputs, encoder_final_state = build_bidirectional_encoder(
                    encoder_cell, num_encoder_layers, encoder_inputs, encoder_inputs_length, scope=scope
                    )
            else:
                encoder_outputs, encoder_final_state = build_basic_encoder(
                    encoder_cell, num_encoder_layers, encoder_inputs, encoder_inputs_length, scope=scope
                    )

    with tf.name_scope('DecoderTrainFeeds'):
        loss_weights = tf.constant([
            [[1] * 4] * 3 + [[0] * 4] * (4 - 3),
            [[1] * 4] * 4 + [[0] * 4] * (4 - 4)
        ],
        dtype=tf.float32, name="loss_weights")

    with tf.name_scope('Decoder'):
        decoder_train_outputs = ptr_decoder(decoder_cell,
                                            encoder_inputs,
                                            encoder_outputs,
                                            decoder_output_length=decoder_targets_length,
                                            encoder_final_state=encoder_final_state,
                                            max_encoder_length=max_seq_length)  # [batch_size, dec_len*, enc_seq_len + 1]
        print("decoder_train_outputs", decoder_train_outputs)

    with tf.name_scope('loss'):
        loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decoder_train_outputs, labels=decoder_targets)
        loss = tf.reduce_sum(loss) / batch_size
        print('loss', loss)
        reg_loss = sum([tf.nn.l2_loss(x) for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
        loss += reg_loss * 0.0001
    train_op = tf.train.AdadeltaOptimizer(learning_rate=0.5).minimize(loss)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    def generate_random_batch(batch_size):
        enc_inputs = np.random.rand(batch_size, max_seq_length, encoder_input_size)
        enc_inputs_len = np.random.randint(1, max_seq_length + 1, size=[batch_size])

        dec_targets_len = enc_inputs_len + 1
        dec_targets = []
        for l, inp in zip(enc_inputs_len, enc_inputs):
            sorted_idx = np.argsort(np.sum(inp[:l], 1))
            sorted_idx = np.hstack([sorted_idx, l])
            sorted_idx = np.pad(sorted_idx, (0, np.max(dec_targets_len) - len(sorted_idx)), 'constant', constant_values=0)
            dec_targets.append(sorted_idx)
        dec_targets = np.array(dec_targets)

        return enc_inputs, enc_inputs_len, dec_targets, dec_targets_len

    def calc_accuracy(dec_targets, dec_targets_len, dec_outputs):
        token_correct = 0
        tokens_in_targets = 0
        tokens_in_outputs = 0
        exact_match = 0

        for target, target_len, output in zip(dec_targets, dec_targets_len, dec_outputs):
            target = target[:target_len]
            output_end = np.where(output == target_len - 1)[0]
            output_end_idx = output_end[0] if len(output_end) else 0
            output = output[:output_end_idx + 1]
            tokens_in_targets += len(target)
            tokens_in_outputs += len(output)

            if np.array_equal(target, output):
                exact_match += 1
            for i in range(min(len(target), len(output))):
                if target[i] == output[i]:
                    token_correct += 1

        em_rate = exact_match / (len(dec_targets) + 1e-8)
        p = token_correct / (tokens_in_outputs + 1e-8)
        r = token_correct / (tokens_in_targets + 1e-8)
        f1 = 2 / (1 / (p + 1e-8) + 1 / (r + 1e-8) + 1e-8)
        return em_rate, f1

    print("# trainable variables:", sum([x.get_shape().num_elements()
                                         for x in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)]))
    last_time = datetime.now()
    data = []
    for i in range(1000):
        data.append(generate_random_batch(batch_size))
    print("%d training data generated", len(data))

    for global_step in range(epochs):

        (encoder_inputs_batch,
         encoder_inputs_length_batch,
         decoder_targets_batch,
         decoder_targets_length_batch) = data[global_step % len(data)]

        feed_dict = {encoder_inputs: encoder_inputs_batch,
                     encoder_inputs_length: encoder_inputs_length_batch,
                     decoder_targets: decoder_targets_batch,
                     decoder_targets_length: decoder_targets_length_batch
        }

        if global_step % 100 == 0:
            infer, loss_value = sess.run([tf.argmax(decoder_train_outputs, 2), loss], feed_dict)
            print("global_step:", global_step, "loss:", loss_value, "time: ", datetime.now() - last_time)
            last_time = datetime.now()
            print("gt:", decoder_targets_batch[0], decoder_targets_length_batch[0])
            print("argmax(decoder_train_outputs):", infer[0])
            em, f1 = calc_accuracy(decoder_targets_batch, decoder_targets_length_batch, infer)
            print("em:", em, "f1:", f1)

        sess.run([train_op], feed_dict)

if __name__ == '__main__':
    test()