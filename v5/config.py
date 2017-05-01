time_step = 40
is_training = True
batch_size = 1200
rec_file = '/home/daiab/code/uqer-data/train.tfrecords'
rec_file_val = '/home/daiab/code/uqer-data/valid.tfrecords'
state_size = 200
hidden_layers = 2
rnn_keep_prop = 1.0
num_samples = 4237825  # 4237825 after 2006-01-01, total 6203371
epochs = 10
iter_num = int(num_samples / batch_size) * epochs
learning_rate = 0.008
momentum = 0.9
power = 0.6
weght_decay = 0.0002
decay_steps = iter_num
ckpt_file = None
class_num = 2
idx_num=7