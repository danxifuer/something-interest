import pandas as pd
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

class DataHandle:
    def __init__(self, filePath, timeStep):
        self.TIME_STEP = timeStep
        originData = self.readCsv(filePath)
        self.formatDataDim(originData)
        # print("==================")
        # print(self.trainData[:5])
        # print(self.data[120:140])

    def readCsv(self, file_path):
        csv = pd.read_csv(file_path, index_col=0)
        return csv.reindex(index=csv.index[::-1])

    def formatDataDim(self, data):
        data = self.concatenateData(data)
        self.data = self.normalization(data)
        self.days = self.data.shape[0]
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
        rows = data.shape[0]
        minus = np.concatenate([[data[0, :]], data[0:rows - 1, :]], 0)
        # print("normalization data==========>")
        # print(data / minus - 1)
        return data / minus - 1

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
        filePath = '/home/daiab/code/ml/something-interest/data/2016.csv'
        self.TIME_STEP = 20
        self.NUM_HIDDEN = 10
        self.epochs = 1
        self.testDays = 40
        self._session = tf.Session()
        dataHandle = DataHandle(filePath, self.TIME_STEP)
        self.data = dataHandle.data
        self.trainData = dataHandle.trainData
        self.days = dataHandle.days

    # 当前天前timeStep天的数据，包含当前天
    def getOneEpochTrainData(self, day):
        assert day >= self.TIME_STEP - 1
        index = day - (self.TIME_STEP - 1)
        # print("get_one_epoch_data >>>>>>>>>>>>>>")
        # print(self.trainData[index])
        return self.trainData[index]

    # 当前天后一天的数据
    def getOneEpochTarget(self, day):
        target = self.data[day + 1]
        # print("get_one_epoch_target >>>>>>>>>>>>>>")
        # print(np.array(target))
        return target[np.newaxis, :]

    def buildGraph(self):
        self.oneTrainData = tf.placeholder(tf.float32, [1, self.TIME_STEP, 4])
        # self.train_target = tf.placeholder(tf.float32, [1, 4])
        self.targetPrice = tf.placeholder(tf.float32, [1, 4])
        cell = tf.nn.rnn_cell.LSTMCell(self.NUM_HIDDEN)
        val, state = tf.nn.dynamic_rnn(cell, self.oneTrainData, dtype=tf.float32)
        self.val = tf.transpose(val, [1, 0, 2])
        self.last_time = tf.gather(self.val, self.TIME_STEP - 1)

        self.weight = tf.Variable(tf.truncated_normal([self.NUM_HIDDEN, 4], dtype=tf.float32))
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, 4]))
        self.predictPrice = tf.matmul(self.last_time, self.weight) + self.bias

        self.diff = tf.sqrt(tf.reduce_sum(tf.square(self.predictPrice - self.targetPrice)))

        self.minimize = tf.train.AdamOptimizer(learning_rate=0).minimize(self.diff)

    def trainModel(self):
        init_op = tf.initialize_all_variables()
        self._session.run(init_op)
        for i in range(self.epochs):
            print("epoch %i " % i)
            for day in range(self.TIME_STEP, self.days - self.testDays):
                self._session.run(self.minimize,
                                  {self.oneTrainData: self.getOneEpochTrainData(day),
                                   self.targetPrice: self.getOneEpochTarget(day)})

                if day % 100 == 0:
                    predictPrice = self._session.run(self.predictPrice,
                                                      {self.oneTrainData: self.getOneEpochTrainData(day),
                                                       self.targetPrice: self.getOneEpochTarget(day)})

                    diff = self._session.run(self.diff,
                                             {self.oneTrainData: self.getOneEpochTrainData(day),
                                              self.targetPrice: self.getOneEpochTarget(day)})


                    print(">>>>>>>>>>>>>>>>>>>")
                    print(predictPrice)
                    # print(self.getOneEpochTrainData(day))
                    print(self.getOneEpochTarget(day))
                    print(diff)

    def test(self):
        predict = np.array([[0, 0, 0, 0]])
        real = np.array([[0, 0, 0, 0]])
        from_index = self.days - self.testDays
        day = [from_index - 1]
        for i in range(from_index, self.days - 1):
            predict_price = self._session.run(self.predictPrice,
                                              {self.oneTrainData: self.getOneEpochTrainData(i),
                                               self.targetPrice: self.getOneEpochTarget(i)})
            real_price = self.getOneEpochTarget(i)
            # print(predict_price)
            predict = np.concatenate([predict, predict_price], 0)
            # print(predict)
            real = np.concatenate([real, real_price], 0)
            # print(real)
            day.append(i)
        # print(predict[:, 0])

        self.plotLine(day[1:], predict[1:, :], real[1:, :])

    def plotLine(self, days, predict, real):
        plt.plot(days, predict[:, 0], 'r-')
        plt.plot(days, real[:, 0], 'b-')
        plt.show()
        plt.plot(days, predict[:, 1], 'r-')
        plt.plot(days, real[:, 1], 'b-')
        plt.show()
        plt.plot(days, predict[:, 2], 'r-')
        plt.plot(days, real[:, 2], 'b-')
        plt.show()
        plt.plot(days, predict[:, 3], 'r-')
        plt.plot(days, real[:, 3], 'b-')
        plt.show()

    def run(self):
        self.buildGraph()
        self.trainModel()
        self.test()
        self._session.close()

if __name__ == '__main__':
    lstmModel = LstmModel()
    lstmModel.run()



