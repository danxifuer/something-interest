import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)
#
# class Option:
#     def __init__(self):
#         """时间跨度"""
#         self.time_step = 20
#         """RNN每层个数"""
#         self.hidden_cell_num = 300
#         """每个code的迭代次数"""
#         self.epochs = 1
#         """批处理大小"""
#         self.batch_size = 400
#         """RNN输出之后的隐藏层层数"""
#         self.hidden_layer_num = 1
#         """RNN每层dropout保留比例"""
#         self.rnn_keep_prop = 0.95
#         """RNN输出之后的隐藏层dropout保留比例"""
#         self.hidden_layer_keep_prop = 0.95
#         """学习率"""
#         self.learning_rate = 0.001
#         """RNN输出之后的隐藏层单元个数"""
#         self.output_cell_num = 400
#         """预测指标"""
#         self.predict_index_type = "closePrice"  # could be one of ["open", "close", "high", "low"]
#         """LSTM forget gate forget bias"""
#         self.forget_bias = 1  # 最好不要动
#         """所有的code迭代多少轮"""
#         self.loop_time = 200
#         """训练数据的norm类型"""
#         self.train_data_norm_type = "rate"  # could be ["zscore", "rate"]
#         """预测数据的norm类型"""
#         self.target_data_norm_type = "rate"  # could be ["zscore", "rate"]
#         """是否checkpoint保存文件"""
#         self.is_save_file = True
#         logger.info("options:::::\n%s", self)
#
#     def __str__(self):
#         return "time_step: " + str(self.time_step) + "\n" \
#                 "hidden_cell_num: " + str(self.hidden_cell_num) + "\n" \
#                 "epochs: " + str(self.epochs) + "\n" \
#                 "batch_size: " + str(self.batch_size) + "\n" \
#                 "hidden_layer_num: " + str(self.hidden_layer_num) + "\n" \
#                 "rnn_keep_prop: " + str(self.rnn_keep_prop) + "\n" \
#                 "hidden_layer_keep_prop: " + str(self.hidden_layer_keep_prop) + "\n" \
#                 "learning_rate: " + str(self.learning_rate) + "\n" \
#                 "output_cell_num: " + str(self.output_cell_num) + "\n" \
#                 "predict_index_type: " + self.predict_index_type + "\n" \
#                 "forget_bias: " + str(self.forget_bias) + "\n" \
#                 "loop_time: " + str(self.loop_time) + "\n" \
#                 "train_data_norm_type: " + self.train_data_norm_type + "\n" \
#                 "target_data_norm_type: " + self.target_data_norm_type + "\n" \
#                 "is_save_file: " + str(self.is_save_file)


"""时间跨度"""
time_step = 50
"""RNN每层个数"""
hidden_cell_num = 200
"""每个code的迭代次数"""
epochs = 200
"""批处理大小"""
batch_size = 500
"""RNN输出之后的隐藏层层数"""
hidden_layer_num = 2
"""RNN每层dropout保留比例"""
rnn_keep_prop = 0.9
"""RNN输出之后的隐藏层dropout保留比例"""
hidden_layer_keep_prop = 1
"""学习率"""
learning_rate = 0.001
"""RNN输出之后的隐藏层单元个数"""
output_cell_num = 400
"""预测指标"""
predict_index_type = 1  # could be one of {"open":0, "close":1, "high":2, "low":3}
"""LSTM forget gate forget bias"""
forget_bias = 1  # 最好不要动
"""训练数据的norm类型"""
train_data_norm_type = "rate"  # could be ["zscore", "rate"]
"""预测数据的norm类型"""
target_data_norm_type = "none"  # could be ["zscore", "rate", "none"]
"""是否checkpoint保存文件"""
is_save_file = False


def config_print():
    string = "time_step: " + str(time_step) + "\n" \
            "hidden_cell_num: " + str(hidden_cell_num) + "\n" \
            "epochs: " + str(epochs) + "\n" \
            "batch_size: " + str(batch_size) + "\n" \
            "hidden_layer_num: " + str(hidden_layer_num) + "\n" \
            "rnn_keep_prop: " + str(rnn_keep_prop) + "\n" \
            "hidden_layer_keep_prop: " + str(hidden_layer_keep_prop) + "\n" \
            "learning_rate: " + str(learning_rate) + "\n" \
            "output_cell_num: " + str(output_cell_num) + "\n" \
            "predict_index_type: " + str(predict_index_type) + "\n" \
            "forget_bias: " + str(forget_bias) + "\n" \
            "train_data_norm_type: " + train_data_norm_type + "\n" \
            "target_data_norm_type: " + target_data_norm_type + "\n" \
            "is_save_file: " + str(is_save_file)
    logger.info("config :\n %s", string)

