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

allStock = ts.get_today_all()
# allStock['code'].to_csv("allstockcode.csv")
stockCodes = allStock['code'].values
# index = 0
# for i in stockCodes:
#     if i == '300253':
#         break
#     index += 1

print(stockCodes)

for code in stockCodes:
    print("download %s" % code)
    downloadOneStock(code)
    time.sleep(0.1)

client.close()
