import tensorflow as tf
from v4.config import config
from v4.model.model_12_15 import LstmModel
from v4.util.load_all_code import load_all_code
from v4.service.construct_train_data import GenTrainData
import numpy as np
import pandas as pd

all_stock_code = load_all_code()

with tf.Graph().as_default(), tf.Session() as session:
    config.config_print()
    lstmModel = LstmModel(session)
    lstmModel.build_graph()
    lstmModel.saver.restore(lstmModel.session, config.ckpt_file_path)
    dd_list = GenTrainData(all_stock_code, config.time_step).dd_list

    last_transaction_date = config.last_transaction_date
    code_list = []
    probability = []
    for dd in dd_list:
        if last_transaction_date != dd.date_range[-1]:
            print("stock code : %d, 预测数据有误, 数据最新日期== %s，但last_transaction_date== %s"
                  % (dd.code, dd.date_range[-1], last_transaction_date))
            continue
        predict = lstmModel.session.run(lstmModel.predict_target,
                              feed_dict={lstmModel.one_train_data: [dd.train_data[-1]],
                                         lstmModel.rnn_keep_prop: 1.0,
                                         lstmModel.hidden_layer_keep_prop: 1.0})

        predict = np.exp(predict)
        prob = predict / np.sum(predict, axis=1)[:, np.newaxis]
        code_list.append(dd.code)
        probability.append(prob.max())
    series = pd.Series(probability, index=code_list, name="probability")
    series.to_csv(config.export_excel_file_path)
    print("predict over..., export csv file path == %s" % config.export_excel_file_path)

    session.close()
