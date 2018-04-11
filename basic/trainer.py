import tensorflow as tf

from basic.model import Model
from my.tensorflow import average_gradients
from tensorflow.python.client import timeline


class MultiGPUTrainer(object):
    def __init__(self, config, models):
        model = models[0]
        assert isinstance(model, Model)
        self.config = config
        self.model = model
        self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.models = models
        losses = []
        grads_list = []
        with tf.variable_scope("grad"):
            for gpu_idx, model in enumerate(models):
                with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                    loss = model.get_loss()
                    grads = self.opt.compute_gradients(loss)
                    losses.append(loss)
                    grads_list.append(grads)
                    tf.get_variable_scope().reuse_variables()

        self.loss = tf.add_n(losses)/len(losses)
        self.grads = average_gradients(grads_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def step(self, sess, batches, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            _, ds = batch
            feed_dict.update(model.get_feed_dict(ds, True))

        if self.config.debug_verbose:
            print("shape of logits and targets:", sess.run([tf.shape(self.models[0].logits),
                                                            tf.shape(self.models[0].targets)
                                                            ], feed_dict=feed_dict))
            print("decoder_inference: ", sess.run([tf.shape(self.models[0].decoder_inference),
                                                            self.models[0].decoder_inference
                                                            ], feed_dict=feed_dict))
            print("decoder_train_logits (logits): ", sess.run([tf.shape(self.models[0].logits),
                                                               self.models[0].logits
                                                               ], feed_dict=feed_dict))
            print("targets: ", sess.run([tf.shape(self.models[0].targets),
                                         self.models[0].targets
                                         ], feed_dict=feed_dict))
            print("answer_string_length_flattened: ", sess.run([tf.shape(self.models[0].answer_string_length_flattened),
                                                                self.models[0].answer_string_length_flattened
                                                                ], feed_dict=feed_dict))
            print("decoder_loss: ", sess.run(self.models[0].decoder_loss, feed_dict=feed_dict))

        if self.config.profiling:
            run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            run_metadata = tf.RunMetadata()
            loss, summary, train_op = \
            sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict,
                     options=run_options, run_metadata=run_metadata)
            tl = timeline.Timeline(run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('evaluation_timeline.json', 'w') as f:
                f.write(ctf)
                print('profile info save into file: ', 'evaluation_timeline.json')
        elif get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op
