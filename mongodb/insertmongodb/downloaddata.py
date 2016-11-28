from pymongo import MongoClient
import tushare as ts
import json
import pandas as pd
import time

stock_code = '000001'

client = MongoClient('mongodb://localhost:27017/')
db = client.quant
collection = db.day_price


def downloadOneStock(stock_code):
    price = ts.get_hist_data(stock_code)
    rows = price.shape[0]
    print("rows >>>> %d" % rows)
    price['stock_code'] = pd.Series([stock_code] * rows, index=price.index)
    price['date'] = pd.Series(price.index.values, index=price.index)
    collection.insert(json.loads(price.to_json(orient='records')))

def insertByAPI():
    allStock = ts.get_today_all()
    # allStock['code'].to_csv("allstockcode.csv")
    stockCodes = allStock['code'].values
    print(stockCodes)

    for code in stockCodes:
        print("download %s" % code)
        downloadOneStock(code)
        time.sleep(0.1)

def insertByCsv():
    path = "/home/daiab/code/ml/something-interest/datasource/000001.csv"
    data = pd.read_csv(path, index_col=0)
    rows = data.shape[0]
    print("rows >>>> %d" % rows)
    data['stock_code'] = pd.Series([stock_code] * rows, index=data.index)
    data['date'] = pd.Series(data.index.values, index=data.index)
    print(data[1:4])
    # print(json.loads(data.to_json(orient='records')))
    collection.insert(json.loads(data.to_json(orient='records')))


# insertByAPI()
insertByCsv()
client.close()

