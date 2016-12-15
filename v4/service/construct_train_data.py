from v4.util.data_preprocess import DataPreprocess
from v4.db.db_service.read_mongodb import ReadDB


class GenTrainData:
    def __init__(self, all_code, time_step):
        read_db = ReadDB()
        preprocess = DataPreprocess(time_step)
        """list of DD object"""
        self.dd_list = []
        for code in all_code:
            db_data, date_range = read_db.read_one_stock_data(code)
            dd = preprocess.process(db_data, date_range)
            dd.code = code
            self.dd_list.append(dd)
        read_db.destory()
