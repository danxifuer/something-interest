class DD:
    def __init__(self, date_index, code="", train_data=None, softmax=None, target=None):
        self.date_index = date_index
        self.train_data = train_data
        self.softmax = softmax
        self.target = target
        self.code = code
        self.days = len(date_index)
        self.train_days = self.days - 80

    # TODO:convert date to index and index to date
