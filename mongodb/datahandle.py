import numpy as np
import logging

logging.basicConfig(level=logging.DEBUG,
                format='%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(message)s',
                datefmt='%b %d %Y %H:%M:%S',
                filename='quantlog.log',
                filemode='w')
logger = logging.getLogger(__name__)

class DataHandle:
    def __init__(self, timeStep):
        self.timeStep = timeStep

    def formatDataDim(self, data):
        zscoreData = self.zscore(data)
        rateNormData = self.rateNorm(data)
        self.target = zscoreData[self.timeStep:, 1:2]
        self.ratio = rateNormData[self.timeStep:, 1:2]
        self.softmax = self.softmaxTarget(self.ratio)
        self.days = self.target.shape[0]
        self.trainData = self.buildSample(zscoreData)[:-1]
        # logger.info("zscore data %s", zscoreData[:30])
        # logger.info("target %s", self.target[:30])
        # logger.info("ratio %s", self.ratio[:30])
        # logger.info("softmax %s", self.softmax[:30])
        # logger.info("trainData %s", self.trainData[:30])
        assert self.ratio.shape[0] == self.softmax.shape[0]
        assert self.ratio.shape[0] == self.trainData.shape[0]
        assert self.ratio.shape[0] == self.target.shape[0]



    def zscore(self, data):
        norm = (data - data.mean(axis=0)) / data.var(axis=0)
        return norm

    def rateNorm(self, data):
        norm = np.zeros_like(data)
        for i in range(data.shape[0] - 1, 0, -1):
            norm[i] = (data[i] / data[i - 1]) - 1
        norm[0] = 0
        return norm

    def softmaxTarget(self, ratio):
        softmax = np.zeros(shape=[ratio.shape[0], 2])
        softmax[np.where(ratio[:, 0] >= 0), 0] = 1
        softmax[np.where(ratio[:, 0] < 0), 1] = 1
        return softmax

    def buildSample(self, data):
        result = []
        rows = data.shape[0]
        for i in range(self.timeStep, rows + 1):
            result.append(np.array(data[i - self.timeStep: i, :]))
        return np.array(result)