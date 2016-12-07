from pymongo import MongoClient
import pymongo
from v3.service.data_preprocess import DataPreprocess
import logging
import pandas as pd
from db.db_connect import DBConnectManage

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='/home/daiab/log/quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)


class ReadDB:
    def __init__(self, data_preprocess):
        self.client = DBConnectManage()
        self.collection = self.client.get_collection()
        self.data_preprocess = data_preprocess
        self.read_count = 0

    def read_one_stock_data(self, code):
        logger.info("read count == %d", self.read_count)
        self.read_count += 1
        dbData = self.collection.find({"ticker": code, "isOpen": 1}).sort("tradeDate", pymongo.ASCENDING)
        data = []
        for dataDict in dbData:
            tmp = []
            """过滤掉异常数据"""
            if float(dataDict["openPrice"]) < 0.001:
                continue
            if float(dataDict["closePrice"]) < 0.001:
                continue
            if float(dataDict["highestPrice"]) < 0.001:
                continue
            if float(dataDict["lowestPrice"]) < 0.001:
                continue
            # if float(dataDict["turnoverVol"]) < 0.001:
            #     continue
            # if float(dataDict["turnoverValue"]) < 0.001:
            #     continue
            # if float(dataDict["turnoverRate"]) < 0.001:
            #     continue
            # if float(dataDict["actPreClosePrice"]) < 0.001:
            #     continue
            tmp.append(dataDict["openPrice"])
            tmp.append(dataDict["closePrice"])
            tmp.append(dataDict["highestPrice"])
            tmp.append(dataDict["lowestPrice"])
            # tmp.append(dataDict["actPreClosePrice"])
            tmp.append(dataDict["tradeDate"])
            # tmp.append(dataDict["turnoverVol"])
            # tmp.append(dataDict["turnoverValue"])
            # tmp.append(dataDict["turnoverRate"])
            # print(dataDict["tradeDate"])
            data.append(tmp)
            # print(tmp)
        count = len(data)
        logger.info("stock code == %s, count == %d", code, count)
        data = pd.DataFrame(
            data, columns=["openPrice", "closePrice", "highestPrice", "lowestPrice", "tradeDate"]).\
            set_index("tradeDate", append=False)
            # data, columns=["openPrice", "closePrice", "highestPrice", "lowestPrice", "tradeDate", "turnoverVol", "turnoverValue", "turnoverRate"])


        # print("origin mongodb data>>>>>>")
        # print(data.loc['2016-11-10':'2016-11-18'])
        self.data_preprocess.process(data)

    def destory(self):
        self.client.close()

if __name__=='__main__':
    data_process = DataPreprocess(2)
    readData = ReadDB(data_process)
    readData.read_one_stock_data(1)
    print("train data>>>>>>")
    for date in data_process.train_data.loc['2016-11-11':'2016-11-17']:
        print("date time == %s" % date)
        print(data_process.train_data[date])
    print("target data>>>>>>")
    print(data_process.target.loc['2016-11-11':'2016-11-17'])
    print("rate data>>>>>>")
    print(data_process.rate.loc['2016-11-11':'2016-11-17'])
    print("softmax data>>>>>>")
    print(data_process.softmax.loc['2016-11-11':'2016-11-17'])

