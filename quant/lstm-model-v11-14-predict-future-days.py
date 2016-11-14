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
        filePath = '/home/daiab/code/ml/something-interest/data/000001-minute.csv'
        # filePath = '/home/daiab/code/ml/something-interest/data/601988.csv'
        # filePath = '/home/daiab/code/ml/something-interest/data/000068.csv'
        self.TIME_STEP = 20
        self.NUM_HIDDEN = 50
        self.epochs = 200
        self._session = tf.Session()
        self.predictFutureDay = 2
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
        return np.reshape(target, [1, 2])

    def buildGraph(self):
        self.oneTrainData = tf.placeholder(tf.float32, [1, self.TIME_STEP, 4])
        self.targetPrice = tf.placeholder(tf.float32, [1, 2])
        cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN)

        cell_2 = tf.nn.rnn_cell.MultiRNNCell([cell] * 1)
        val, _ = tf.nn.dynamic_rnn(cell_2, self.oneTrainData, dtype=tf.float32)

        self.val = tf.transpose(val, [1, 0, 2])
        self.lastTime = tf.gather(self.val, self.TIME_STEP - 1)


        self.weight = tf.Variable(tf.truncated_normal([self.NUM_HIDDEN, 2], dtype=tf.float32))
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, 2]))
        self.predictPrice = tf.matmul(self.lastTime, self.weight) + self.bias

        self.diff = tf.sqrt(tf.reduce_mean(tf.square(self.predictPrice - self.targetPrice)))

        self.minimize = tf.train.AdamOptimizer().minimize(self.diff)

    def trainModel(self):
        init_op = tf.initialize_all_variables()
        self._session.run(init_op)
        for epoch in range(self.epochs):
            print("epoch %i >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>" % epoch)
            diffSum = 0
            count = 0
            for day in range(self.TIME_STEP, self.trainDays):
                # print("day %d" % day)
                realPrice = self.getOneEpochTarget(day)
                oneEpochTrainData = self.getOneEpochTrainData(day)

                self._session.run(self.minimize,
                                  {self.oneTrainData: oneEpochTrainData,
                                   self.targetPrice: realPrice})

                predictPrice = self._session.run(self.predictPrice,
                                                 {self.oneTrainData: oneEpochTrainData,
                                                  self.targetPrice: realPrice})

                diff = self._session.run(self.diff,
                                         {self.oneTrainData: oneEpochTrainData,
                                          self.targetPrice: realPrice})
                diffSum += diff
                count += 1

                if day % 100 == 0:
                    print("...................................................")
                    print("predictPrice is %s" % predictPrice)
                    print("real price is %s" % self.getOneEpochTarget(day))
                    print("diff is %s" % diff)
            print("diff mean >>>>>>>>>>>>>> %f" % (diffSum / count))
            if epoch % 20 == 0:
                self.test()

    def test(self):
        predict = np.zeros([1])
        real = np.zeros([1])
        dayIndex = [self.trainDays - 1]
        for day in range(self.trainDays, self.days - self.predictFutureDay):
            predictPrice = self._session.run(self.predictPrice,
                                              {self.oneTrainData: self.getOneEpochTrainData(day),
                                               self.targetPrice: self.getOneEpochTarget(day)})[:, 0]
            realPrice = self.getOneEpochTarget(day-1)[:, 0]

            # print(predict_price)
            predict = np.concatenate([predict, predictPrice])
            # print(predict)
            real = np.concatenate([real, realPrice])
            # print(real)
            dayIndex.append(day)
        # print(predict[:, 0])
        if self.isPlot:
            self.plotLine(dayIndex[1:], predict[1:], real[1:])


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



