import logging

import numpy as np
from v4.config import config
from v4.entry.train_data_struct import DD

"""
一定要注意，处理数据时避免在源数据上进行赋值,以及类型转化可能隐含的错误
如果时间跨度是5天，取'2015-01-10'的训练数据时返回['2015-01-06','2015-01-07','2015-01-08','2015-01-09','2015-01-10'],
也就是包含'2015-01-10'这一天的数据。但是取目标数据时返回['2015-01-11']，也就是第二天的数据。这样方便训练时能用同一日期索引
"""

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)


shape_0 = 0

class DataPreprocess:
    def __init__(self, time_step):
        self.time_step = time_step

    """
    input: pandas DataFrame type, shape like [sample_size, variable_size]
    """
    def process(self, origin_data, date_range):
        global shape_0
        shape_0 = origin_data.shape[0]
        start_index = self.time_step - 1 + 1
        end_index = shape_0 - 2
        filter_date_range = date_range[start_index: end_index]

        """generate_train_serial 返回的数据长度是(sample_size - time_step + 1)"""
        if config.train_data_norm_type == "zscore":
            train_data = generate_train_serial(origin_data, self.time_step, norm_type="zscore")
        elif config.train_data_norm_type == "rate":
            train_data = generate_train_serial(origin_data, self.time_step, norm_type="rate")
        else:
            raise Exception("norm data type error")

        origin_target_data = origin_data[:, config.predict_index_type]

        if config.target_data_norm_type == "zscore":
            target = norm_to_zscore(origin_target_data)
        elif config.target_data_norm_type == "rate":
            target = norm_to_rate(origin_target_data)
        elif config.target_data_norm_type == "none":
            target = None
        else:
            raise Exception("norm data type error")

        softmax = generate_softmax_target(origin_target_data)

        """DD object, 因为最开始第一天的数据，如果是使用rate norm的方式，第一天的数据是有问题的，所以这里把他排除出去"""
        if target:
            return DD(filter_date_range, train_data=train_data[1:-1], softmax=softmax[self.time_step + 1:], target=target[self.time_step + 1:])
        else:
            return DD(filter_date_range, train_data=train_data[1:-1], softmax=softmax[self.time_step + 1:])


"""
input: pandas DataFrame type data, shape like [sample_size, variable_size]
output: [sample_size, variable_size], but the output is shifted by -1, so the last row is [NaN, ..., NaN]
"""
def norm_to_zscore(origin_data):
    return (origin_data - origin_data.mean(axis=0)) / origin_data.std(axis=0)


"""
input: pandas DataFrame type, shape like [sample_size, 1], just handle one variable per time
output: [sample_size, 2], the first sample will handled to [1, 0] or [0, 1] by probability,
        because it have no reference for contrast
        increase will denote as [1, 0], instead decrease will denote as [0, 1]
for example:
input: [[2],
        [3],
        [5],
        [4]]
output:[[0, 0],
        [1, 0],
        [1, 0],
        [0, 1]]
"""
def generate_softmax_target(origin_data):
    # warn: 注意append的顺序
    softmax = [[0, 0]]
    for row in range(1, shape_0):
        if origin_data[row] >= origin_data[row - 1]:
            softmax.append([1, 0])
        else:
            softmax.append([0, 1])
    assert len(softmax) == shape_0
    return np.array(softmax)


"""
input: pandas DataFrame type, shape like [sample_size, variable_size]
        the first sample will handled to be [1, 1, ..., 1]
output: [sample_size, variable_size]
"""
def norm_to_rate(orgin_data):
    rate = np.ones_like(orgin_data)
    for row in range(shape_0 - 1, 0, -1):
        row_rate = orgin_data[row] / orgin_data[row - 1] - 1
        rate[row] = row_rate
    return rate


"""
input: pandas DataFrame type, shape like [sample_size, variable_size]
        the input will be normalized to z-score for preprocess
        norm_type = "zscore" or "rate"
     time_step: how many samples per serial
output: [sample_size - length + 1, sample_size, variable_size]
example: length = 2
input: [[1, 2],
        [2, 3],
        [3, 4],
        [4, 5]]
output: [[[1, 2],
          [2, 3]],
         [[2, 3],
          [3, 4]],
         [[3, 4],
          [4, 5]]]
notification: 使用索引查询时，比如查询'2014-01-20', 时间跨度是5天，
那么返回的是'2014-01-16至'2014-01-20'的数据
"""
def generate_train_serial(origin_data, time_step, norm_type="zscore"):
    if norm_type == "zscore":
        norm_data = norm_to_zscore(origin_data)
    elif norm_type == "rate":
        norm_data = norm_to_rate(origin_data)
    else:
        raise Exception("norm_type is not found")

    train_data = []
    for row in range(time_step, shape_0 + 1):
        step_data = norm_data[row - time_step: row]
        train_data.append(step_data)

    assert len(train_data) == shape_0 - time_step + 1
    return np.array(train_data)


if __name__ == "__main__":
    datahandle = DataPreprocess(2)

