import time
import tensorflow as tf
import logging
from v5 import config as cfg
from v5.model import read_rec
import numpy as np
# from tensorflow.contrib.rnn import LayerNormBasicLSTMCell, BasicRNNCell, GridLSTMCell, GRUCell, BasicLSTMCell
from v5.model.bnlstm import BNLSTMCell
logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                    datefmt='%b %d %Y %H:%M:%S')


class Model:
    def __init__(self, session):
        self.session = session
        self.right = 0
        self.samples = 0
        self.right_list = np.zeros([5], dtype=int)
        self.samples_list = np.zeros([5], dtype=int)
        self.w = {'fc_weight_1':
                      tf.Variable(tf.truncated_normal([cfg.time_step * cfg.state_size, cfg.class_num], stddev=0.01, dtype=tf.float32), name='fc_weight_1'),
                  }
        self.b = {'fc_bias_1':
                      tf.Variable(tf.constant(0.01, dtype=tf.float32, shape=[cfg.class_num]), name='fc_bias_1'),
                  }

    def build_graph(self, batch_data, batch_label, valid=False):
        # self.batch_data, self.batch_label = read_rec.read_and_decode(cfg.rec_file)
        param = {}
        with tf.variable_scope('ce') as scope:
            if valid:
                # tf.get_variable_scope().reuse = True
                scope.reuse_variables()
            multi_cell = tf.contrib.rnn.MultiRNNCell(
                [BNLSTMCell(cfg.state_size, training=True) for _ in range(cfg.hidden_layers)])
            state_init = multi_cell.zero_state(cfg.batch_size, dtype=tf.float32)
            val, states = tf.nn.dynamic_rnn(multi_cell, batch_data, initial_state=state_init, dtype=tf.float32)

            """reshape the RNN output"""
            # val = tf.transpose(val, [1, 0, 2])
            # self.val = tf.gather(val, val.get_shape()[0] - 1)
            dim = cfg.time_step * cfg.state_size
            val = tf.reshape(val, [-1, dim])

            fc_weight = tf.get_variable(name='fc_weight_1', shape=[cfg.time_step * cfg.state_size, cfg.class_num],
                                          initializer=tf.truncated_normal_initializer(stddev=0.01, dtype=tf.float32))
            fc_bias = tf.get_variable(name='fc_bias', shape=[cfg.class_num], initializer=tf.constant_initializer())
            logits = tf.nn.xw_plus_b(val, fc_weight, fc_bias)
            param['logits'] = logits
            if valid:
                return logits
        cross_entropy = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=batch_label, name="cross_entropy"))
        param['ce'] = cross_entropy
        global_step = tf.get_variable('global_step', shape=[], dtype=tf.int64,
                                      initializer=tf.zeros_initializer(),
                                      trainable=False)
        poly_decay_lr = tf.train.polynomial_decay(learning_rate=cfg.learning_rate,
                                                  global_step=global_step,
                                                  decay_steps=cfg.decay_steps,
                                                  end_learning_rate=0.0002,
                                                  power=cfg.power)
        param['lr'] = poly_decay_lr
        weight = [v for _, v in self.w.items()]
        norm = tf.add_n([tf.nn.l2_loss(i) for i in weight])
        minimize = tf.train.MomentumOptimizer(
            learning_rate=poly_decay_lr, momentum=cfg.momentum).\
            minimize(cross_entropy + cfg.weght_decay * norm, global_step=global_step)
        param['minimize'] = minimize
        self.saver = tf.train.Saver()
        return param

    def save_model(self):
        save_time = time.strftime("%Y-%m-%d-%H-%M", time.localtime())
        self.saver.save(self.session, "../log/train/%s.ckpt" % save_time)
        logging.info("save file time: %s", save_time)

    def run(self):
        train_data, train_label = read_rec.read_and_decode(cfg.rec_file)
        val_data, val_label = read_rec.read_and_decode(cfg.rec_file_val)
        param = self.build_graph(train_data, train_label)
        val_logits = self.build_graph(val_data, val_label, valid=True)

        if cfg.ckpt_file is None:
            self.session.run(tf.global_variables_initializer())
            logging.info("init all variables by random")
        else:
            logging.info("init all variables by previous file")
            self.saver.restore(self.session, cfg.ckpt_file)

        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.session, coord=coord)
        for i in range(cfg.iter_num):
            _, logits, labels = self.session.run([param['minimize'], param['logits'], train_label])
            self.acc_dist(logits, labels, i)
            if (i + 1) % 40 == 0:
                ce, lr = self.session.run([param['ce'], param['lr']])
                logging.info("%d th iter, cross_entropy == %s, learning rate == %s", i, ce, lr)
                logging.info('accuracy == %s',  self.right_list / self.samples_list)
                logging.info('samples distribute == %s', self.samples_list)
                self.right_list = np.zeros([5], dtype=int)
                self.samples_list = np.zeros([5], dtype=int)
            if (i + 1) % 2000 == 0:
                self.valid(val_logits, val_label)
                # self.save_model()
                
        coord.request_stop()
        coord.join(threads=threads)

    def valid(self, logits, labels):
        samples_list = np.zeros([5], dtype=int)
        right_list = np.zeros([5], dtype=int)
        for i in range(int(117350 / cfg.batch_size)):  # total 117350 samples
            logits_result, labels_result = self.session.run([logits, labels])
            threshold = np.arange(0.5, 1, 0.1)
            max = np.max(logits_result, axis=1, keepdims=True)
            prob = np.exp(logits_result - max) / np.sum(np.exp(logits_result - max), axis=1, keepdims=True)
            b_labels = labels_result.astype(bool)[:, np.newaxis]
            b_idx = np.concatenate((~b_labels, b_labels), axis=1)
            target = b_idx.astype(int)
            for j, t in enumerate(threshold):
                bool_index = prob > t
                samples_list[j] += np.count_nonzero(bool_index)
                right_list[j] += np.count_nonzero(target[bool_index])
        logging.info('valid right == %s', right_list)
        logging.info('valid accuracy == %s', right_list / samples_list)

    def acc(self, logits, label, gs):
        max_idx = np.argmax(logits, axis=1)
        equal = np.sum(np.equal(max_idx, label).astype(int))
        self.right += equal
        self.samples += cfg.batch_size

    def acc_dist(self, logits, labels, gs):
        max_idx = np.argmax(logits, axis=1)
        if (gs + 1) % 40 == 0:
            print(np.count_nonzero(max_idx))
            print(np.count_nonzero(labels))
            print('--------------')
        threshold = np.arange(0.5, 1, 0.1)
        max = np.max(logits, axis=1, keepdims=True)
        prob = np.exp(logits - max) / np.sum(np.exp(logits - max), axis=1, keepdims=True)
        b_labels = labels.astype(bool)[:, np.newaxis]
        b_idx = np.concatenate((~b_labels, b_labels), axis=1)
        target = b_idx.astype(int)
        for i, t in enumerate(threshold):
            bool_index = prob > t
            self.samples_list[i] += np.count_nonzero(bool_index)
            self.right_list[i] += np.count_nonzero(target[bool_index])


def run():
    with tf.Graph().as_default(), tf.Session() as session:
        model = Model(session)
        """init variables"""

        model.run()


if __name__ == '__main__':
    run()
