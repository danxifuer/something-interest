"""
使用多层lstm神经层,重新使用减去均值除以标准差的方式正则化数据,仅预测close price
尝试测试多个stock
"""
import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DataHandle:
    def __init__(self, filePath, timeStep):
        self.TIME_STEP = timeStep
        originData = self.readCsv(filePath)
        self.formatDataDim(originData)
        # print(self.trainData[:5])
        # print("==================")
        # print(self.data[0:60])

    def readCsv(self, file_path):
        csv = pd.read_csv(file_path, index_col=0)
        return csv.reindex(index=csv.index[::-1])

    def formatDataDim(self, data):
        data = self.concatenateData(data)
        self.data = self.normalization(data)
        self.days = self.data.shape[0]
        self.target = self.data[:, 1:2]
        # print("target >>>>>>>>>>>>>>>>>>>>")
        # print(self.target)
        self.trainData = self.buildSample(self.data)

    def concatenateData(self, data):
        data = np.concatenate(
            [data.open.values[:, np.newaxis],
             data.close.values[:, np.newaxis],
             data.high.values[:, np.newaxis],
             data.low.values[:, np.newaxis]], 1)
        # print("concat data==========>")
        # print(data)
        return data

    def normalization(self, data):
        # rows = data.shape[0]
        # norm = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0))
        norm = (data - data.mean(axis=0)) / data.var(axis=0)
        return norm

    def buildSample(self, data):
        result = []
        rows = data.shape[0]
        for i in range(self.TIME_STEP, rows + 1):
            tmp = []
            tmp.append(data[i - self.TIME_STEP: i, :])
            result.append(np.array(tmp))
        return result



class LstmModel:
    def __init__(self):
        filePath = '/home/daiab/code/ml/something-interest/data/000001.csv'
        # filePath = '/home/daiab/code/ml/something-interest/data/601988.csv'
        # filePath = '/home/daiab/code/ml/something-interest/data/000068.csv'
        self.TIME_STEP = 19
        self.NUM_HIDDEN = 50
        self.epochs = 200
        self._session = tf.Session()
        self.predictFutureDay = 1
        dataHandle = DataHandle(filePath, self.TIME_STEP)
        self.data = dataHandle.data
        self.trainData = dataHandle.trainData
        self.target = dataHandle.target
        self.days = dataHandle.days
        self.testDays = (int)(self.days / 4)
        print("all days is %d" % self.days)
        self.trainDays = self.days - self.testDays
        self.isPlot = True
        del dataHandle


    # 当前天前timeStep天的数据，包含当前天
    def getOneEpochTrainData(self, day):
        assert day >= self.TIME_STEP - 1
        index = day - (self.TIME_STEP - 1)
        # print("get_one_epoch_data >>>>>>>>>>>>>>")
        # print(self.trainData[index])
        return self.trainData[index]

    # 当前天后一天的数据
    def getOneEpochTarget(self, day):
        target = self.target[day + 1:day + 1 + self.predictFutureDay, :]
        # target = np.hsplit(target, [1])[1]
        # print("get_one_epoch_target >>>>>>>>>>>>>>")
        # print(np.array(target))
        return np.reshape(target, [1, self.predictFutureDay])

    def buildGraph(self):
        self.oneTrainData = tf.placeholder(tf.float32, [1, self.TIME_STEP, 4])
        self.targetPrice = tf.placeholder(tf.float32, [1, self.predictFutureDay])
        self.yesterdayPrice = tf.placeholder(tf.float32, [1, self.predictFutureDay])
        cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN)

        cell_2 = tf.nn.rnn_cell.MultiRNNCell([cell] * 1)
        # cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=dropout)
        val, self.states = tf.nn.dynamic_rnn(cell_2, self.oneTrainData, dtype=tf.float32)
        # tf.nn.xw_plus_b()
        # tf.get_variable()


        self.val = tf.transpose(val, [1, 0, 2])
        self.lastTime = tf.gather(self.val, self.val.get_shape()[0] - 1)

        self.weight = tf.Variable(tf.truncated_normal([self.NUM_HIDDEN, self.predictFutureDay], dtype=tf.float32))
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, self.predictFutureDay]))
        self.predictPrice = tf.matmul(self.lastTime, self.weight) + self.bias
        self.diff = tf.sqrt(tf.reduce_mean(tf.squared_difference(self.predictPrice, self.targetPrice)))

        inse = tf.reduce_sum(self.predictPrice * self.targetPrice)
        l = tf.reduce_sum(self.predictPrice * self.predictPrice)
        r = tf.reduce_sum(self.targetPrice * self.targetPrice)
        self.dice = 2 * inse / (l + r)

        # trend = (self.predictPrice - self.yesterdayPrice) * (self.targetPrice - self.yesterdayPrice)
        # self.sign = tf.reduce_sum(tf.sign(trend))

        # tf.clip_by_value()

        self.minimize = tf.train.AdamOptimizer().minimize(self.diff - self.dice)

    with tf.device("/cpu:0"):
        def trainModel(self):
            self._session.run(tf.initialize_all_variables())
            for epoch in range(self.epochs):
                print("epoch %i >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" % epoch)
                diffSum = 0
                count = 0
                for day in range(self.TIME_STEP, self.trainDays):
                    # print("day %d" % day)
                    oneEpochTrainData = self.getOneEpochTrainData(day)
                    realPrice = self.getOneEpochTarget(day)
                    yesterdayPrice = self.getOneEpochTarget(day - 1)

                    self._session.run(self.minimize, {self.oneTrainData: oneEpochTrainData,
                                                      self.targetPrice: realPrice,
                                                      self.yesterdayPrice: yesterdayPrice})

                    predictPrice = self._session.run(self.predictPrice,
                                                     {self.oneTrainData: oneEpochTrainData,
                                                      self.targetPrice: realPrice,
                                                      self.yesterdayPrice: yesterdayPrice})

                    diff = self._session.run(self.diff,
                                                     {self.oneTrainData: oneEpochTrainData,
                                                      self.targetPrice: realPrice,
                                                      self.yesterdayPrice: yesterdayPrice})

                    dice = self._session.run(self.dice,
                                                     {self.oneTrainData: oneEpochTrainData,
                                                      self.targetPrice: realPrice,
                                                      self.yesterdayPrice: yesterdayPrice})

                    # states = self._session.run(self.states,
                    #                          {self.oneTrainData: oneEpochTrainData,
                    #                           self.targetPrice: realPrice,
                    #                           self.yesterdayPrice: yesterdayPrice})


                    diffSum += diff
                    count += 1

                    if day % 100 == 0:
                        print("...................................................")
                        print("predictPrice is %s" % predictPrice)
                        print("real price is %s" % realPrice)
                        print("diff is %s" % diff)
                        print("dice is %s" % dice)
                        # print("state is %s" % states)
                        # print(states.shape)
                print("diff mean >>>>>>>>>>>>>> %f" % (diffSum / count))
                if epoch % 20 == 0:
                    self.test()

    with tf.device("/cpu:1"):
        def test(self):
            predict = np.zeros([1])
            real = np.zeros([1])
            dayIndex = [self.trainDays - 1]

            for day in range(self.trainDays, self.days - self.predictFutureDay):
                trainData = self.getOneEpochTrainData(day)
                # target = self.getOneEpochTarget(day)

                predictPrice = self._session.run(self.predictPrice,
                                                 {self.oneTrainData: trainData})[:, 0]

                realPrice = self.getOneEpochTarget(day)[:, 0]

                # print(">>>>>>>>>>>>>>>,,,,,,,,,,,,, day %d" % day)
                # print(trainData)
                # print(target)
                # print(realPrice)

                # print(predict_price)
                predict = np.concatenate([predict, predictPrice])
                # print(predict)
                real = np.concatenate([real, realPrice])
                # print(real)
                dayIndex.append(day)
            # print(predict[:, 0])
            if self.isPlot:
                self.plotLine(dayIndex[1:], predict[1:], real[1:])

    with tf.device("/cpu:3"):
        def plotLine(self, days, predict, real):
            plt.ylabel("close")
            plt.grid(True)
            plt.plot(days, predict, 'r-')
            plt.plot(days, real, 'b-')
            plt.show()


    def run(self):
        self.buildGraph()
        self.trainModel()
        self.test()
        self._session.close()

if __name__ == '__main__':
    lstmModel = LstmModel()
    lstmModel.run()



