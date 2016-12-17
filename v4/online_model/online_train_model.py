from v4.model.model_12_15 import *
from v4.util.load_all_code import load_all_code
import sys

all_stock_code = load_all_code()

with tf.Graph().as_default(), tf.Session() as session:
    config.config_print()
    print("please confirm!!!!\n最后交易日期：%s, 取数据天数：%s" % (config.ot_last_transaction_date, config.ot_limit))
    print("please type 'yes' to continue, and type Ctl+D to finish:\n")
    message = sys.stdin.readlines()
    if message != "yes":
        print("exit success")
        sys.exit()

    print("start to online training")
    lstmModel = LstmModel(session)
    lstmModel.build_graph()
    lstmModel.load_data(config.ONLINE_TRAIN, end_date=config.ot_last_transaction_date, limit=config.ot_limit)
    lstmModel.saver.restore(lstmModel.session, config.op_ckpt_file_path)
    print("restore ckpt file over")
    lstmModel.train_model(operate_type=config.ONLINE_TRAIN)
    print("online training over...")
    session.close()
