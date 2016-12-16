import numpy as np


class DD:
    def __init__(self, date_index, code="", train_data=None, softmax=None, target=None):
        assert train_data.shape[0] == softmax.shape[0]
        self.date_index = date_index
        self.train_data = train_data
        self.softmax = softmax
        self.target = target
        self.code = code
        self.days = len(date_index)
        train_days = self.days - 100
        index = list(range(self.days))
        np.random.shuffle(index)
        self.train_index = index[:train_days]
        self.test_index = index[train_days:]


    # TODO:convert date to index and index to date
