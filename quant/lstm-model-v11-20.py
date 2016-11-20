"""
use softmax regression and read data from mongodb
"""
import tensorflow as tf
import numpy as np
from mongodb.readmongodb import ReadDB
from mongodb.datahandle import DataHandle
from mongodb.readallstockcode import readallcode
import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)


def batch(batch_size, data=None, target=None, ratio=None, softmax=None, shuffle=False):
    assert len(data) == len(target)
    if shuffle:
        indices = np.arange(len(data), dtype=np.int32)
        # indices = range(len(datasource))
        np.random.shuffle(indices)
    for start_idx in range(0, len(data) - batch_size + 1, batch_size):
        if shuffle:
            excerpt = indices[start_idx:start_idx + batch_size]
        else:
            excerpt = slice(start_idx, start_idx + batch_size)
        yield data[excerpt], target[excerpt], ratio[excerpt], softmax[excerpt]



class LstmModel:
    def __init__(self):
        self.timeStep = 10
        self.hiddenNum = 50
        self.epochs = 200
        self._session = tf.Session()

        self.allStockCode = readallcode()
        self.dataHandle = DataHandle(self.timeStep)
        self.readDb = ReadDB(stockCodeList=self.allStockCode, threshold=2 * self.timeStep, datahandle=self.dataHandle)
        # 从数据库取出一次数据后，重复利用几次
        self.reuseTime = 10
        self.isPlot = True
        self.batchSize = 10

    def updateData(self):
        self.readDb.readOneStockData()
        self.trainData = self.dataHandle.trainData
        self.target = self.dataHandle.target
        self.ratio = self.dataHandle.ratio
        self.softmax = self.dataHandle.softmax
        self.days = self.target.shape[0]
        self.testDays = (int)(self.days / 5)
        self.trainDays = self.days - self.testDays


    # 当前天前timeStep天的数据，包含当前天
    def getOneEpochTrainData(self, day):
        assert day >= 0
        return self.trainData[day]

    # 当前天后一天的数据
    def getOneEpochTarget(self, day):
        target = self.target[day:day + 1, :]
        return np.reshape(target, [1, 1])

    def getOneEpochRatio(self, day):
        ratio = self.ratio[day:day + 1, :]
        return np.reshape(ratio, [1, 1])

    def getOneEpochSoftmax(self, day):
        softmax = self.softmax[day:day + 1, :]
        return softmax

    def buildGraph(self):
        self.oneTrainData = tf.placeholder(tf.float32, [None, self.timeStep, 4])
        self.targetPrice = tf.placeholder(tf.float32, [None, 2])
        cell = tf.nn.rnn_cell.LSTMCell(self.hiddenNum)

        cell_2 = tf.nn.rnn_cell.MultiRNNCell([cell] * 1)
        val, self.states = tf.nn.dynamic_rnn(cell_2, self.oneTrainData, dtype=tf.float32)

        self.val = tf.transpose(val, [1, 0, 2])
        self.lastTime = tf.gather(self.val, self.val.get_shape()[0] - 1)

        self.weight = tf.Variable(tf.truncated_normal([self.hiddenNum, 2], dtype=tf.float32))
        self.bias = tf.Variable(tf.constant(0.0, dtype=tf.float32, shape=[1, 2]))
        self.predictPrice = tf.matmul(self.lastTime, self.weight) + self.bias

        self.cross_entropy = tf.nn.softmax_cross_entropy_with_logits(self.predictPrice, self.targetPrice)

        self.minimize = tf.train.AdamOptimizer().minimize(self.cross_entropy)

    with tf.device("/gpu:0"):
        def trainModel(self):
            self._session.run(tf.initialize_all_variables())
            for epoch in range(self.epochs):

                for i in range(len(self.allStockCode) * self.reuseTime):
                    logger.info("epoch == %d, time == %d", epoch, i)
                    self.updateData()
                    batchData = batch(self.batchSize,
                                      self.trainData[:self.trainDays],
                                      self.target[:self.trainDays],
                                      self.ratio[:self.trainDays],
                                      self.softmax[:self.trainDays], shuffle=False)

                    count = 1
                    dict = {}
                    for oneEpochTrainData, _, _, softmax in batchData:
                        count += 1
                        dict = {self.oneTrainData: oneEpochTrainData, self.targetPrice: softmax}
                        # logger.info("dict == %s", dict)

                        self._session.run(self.minimize, feed_dict=dict)

                    if len(dict) != 0:
                        crossEntropy = self._session.run(self.cross_entropy, feed_dict=dict).sum()
                        logger.info("crossEntropy == %f", crossEntropy)

                    if epoch % 20 == 0:
                        self.test()

    with tf.device("/cpu:1"):
        def test(self):
            count, right = 1, 0
            logger.info("test begin >>>>>>>>>>>>>>>>>>>>>>> ")

            for day in range(self.trainDays, self.days - 1):
                trainData = [self.getOneEpochTrainData(day)]
                predictPrice = self._session.run(self.predictPrice,
                                                 {self.oneTrainData: trainData})

                realPrice = self.getOneEpochSoftmax(day)

                if np.argmax(predictPrice) == np.argmax(realPrice): right += 1
                count += 1

            logger.info("test right ratio == %f", (right / count))


    def run(self):
        self.buildGraph()
        self.trainModel()
        self.test()
        self._session.close()
        self.readDb.destory()

if __name__ == '__main__':
    lstmModel = LstmModel()
    lstmModel.run()



