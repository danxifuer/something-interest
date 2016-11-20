from pymongo import MongoClient
import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)

class ReadDB:
    def __init__(self, stockCodeList, threshold, datahandle):
        self.stockCodeList = stockCodeList
        self.totalNum = len(stockCodeList)
        self.index = 0
        self.client = MongoClient('mongodb://localhost:27017/')
        self.collection = self.client.quant.day_price
        self.threshold = threshold
        self.datahandle = datahandle


    def readOneStockData(self):
        if self.index >= self.totalNum:
            self.index = 0
        stockCode = self.stockCodeList[self.index]
        logger.info("stock code == %s index == %d", stockCode, self.index)
        count = self.collection.find({"stock_code": stockCode}).count()
        logger.info("stock code == %s, stock count == %d",stockCode , count)
        if count < self.threshold:
            logger.info("skip current stock, count == %d", count)
            self.index += 1
            self.readOneStockData()
        dbData = self.collection.find({"stock_code": stockCode}).sort("date")
        data = []
        for dataDict in dbData:
            tmp = []
            tmp.append(dataDict["open"])
            tmp.append(dataDict["close"])
            tmp.append(dataDict["high"])
            tmp.append(dataDict["low"])
            data.append(tmp)
        self.index += 1
        self.datahandle.formatDataDim(np.array(data))


    def destory(self):
        self.client.close()

if __name__=='__main__':
    readData = ReadDB(['000001'], 200)
    logger.info("data from db %s", readData.readOneStockData())
    # print(readData.readOneStockData())
