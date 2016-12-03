import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)

class Option:
    def __init__(self):
        self.time_step = 100
        self.hidden_cell_num = 200
        self.epochs = 200
        self.batch_size = 50
        self.hidden_layer_num = 1
        self.rnn_keep_prop = 1
        self.hidden_layer_keep_prop = 0.9
        self.learning_rate = 0.001
        self.output_cell_num = 512
        self.predict_type = "closePrice"  # could be one of ["open", "close", "high", "low"]
        self.forget_bias = 0.8
        self.loop_time = 50
        self.data_norm_type = "rate"  # could be ["zscore", "rate"]
        self.is_save_file = False
        logger.info("options:::::\n%s", self)

    def __str__(self):
        return "time_step: " + str(self.time_step) + "\n" \
                "hidden_cell_num: " + str(self.hidden_cell_num) + "\n" \
                "epochs: " + str(self.epochs) + "\n" \
                "batch_size: " + str(self.batch_size) + "\n" \
                "hidden_layer_num: " + str(self.hidden_layer_num) + "\n" \
                "rnn_keep_prop: " + str(self.rnn_keep_prop) + "\n" \
                "hidden_layer_keep_prop: " + str(self.hidden_layer_keep_prop) + "\n" \
                "learning_rate: " + str(self.learning_rate) + "\n" \
                "output_cell_num: " + str(self.output_cell_num) + "\n" \
                "predict_type: " + self.predict_type + "\n" \
                "forget_bias: " + str(self.forget_bias) + "\n" \
                "loop_time: " + str(self.loop_time) + "\n" \
                "train_data_type: " + self.data_norm_type + "\n" \
                "is_save_file: " + str(self.is_save_file)
