import logging
import os


def get_logger(file_name):
    logging.basicConfig(level=logging.DEBUG,
                              format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                              datefmt='%b %d %Y %H:%M:%S',
                              filename='/home/daiab/log/quantlog.log',
                              filemode='w')

    return logging.getLogger(file_name)

logger = get_logger(__name__)

# -----------enum------------
"""
case 1: 线下训练
case 2: 在线训练
case 3: 预测
"""
OFFLINE_TRAIN = 1
ONLINE_TRAIN = 2
PREDICT = 3

# ------------some file path------------
code_csv_file_path = "/home/daiab/code/ml/something-interest/v4/csv_data/all_code.csv"
update_uqer_csv_file_path = "/home/daiab/Public/update_data/download-2016-12-03.csv"

# ------------model parameter------------
"""时间跨度"""
time_step = 40
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
output_cell_num = 512
"""预测指标"""
predict_index_type = 1  # could be one of {"open":0, "close":1, "high":2, "low":3}
"""LSTM forget gate forget bias"""
forget_bias = 1  # 最好不要动
"""训练数据的norm类型"""
train_data_norm_type = "rate"  # could be ["zscore", "rate"]
"""预测数据的norm类型"""
target_data_norm_type = "none"  # could be ["zscore", "rate", "none"]
"""是否checkpoint保存文件"""
is_save_file = True
# -------------online predict parameter: op prefix--------------
"""训练文件的保存路径"""
op_ckpt_file_path = "/home/daiab/ckpt/2016-12-16-15-53.ckpt"
"""预测结果导出excel的路径"""
op_export_excel_file_path = "/home/daiab/ckpt/predict-outcome.csv"
"""最邻近数据的日期(且这一天必须是交易日),格式必须： 1994-09-07 """
op_last_transaction_date = '2016-12-02'
# -------------online train parameter: ot prefix--------------
"""训练文件的保存路径"""
ot_ckpt_file_path = "/home/daiab/ckpt/2016-12-16-15-53.ckpt"
ot_limit = time_step + 0
ot_last_transaction_date = '2016-12-02'




def config_print():
    config_file_path = os.path.abspath(__file__)
    print(config_file_path)
    with open(config_file_path, 'r', encoding='utf-8') as config_file:
        for line in config_file:
            logger.info("|||  " + line)

