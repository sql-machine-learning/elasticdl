import numpy as np
import os
from tensorflow.keras.preprocessing.sequence import pad_sequences

class LoadData(object):
    def __init__(self, path):
        self.trainfile = os.path.join(path, "train.libfm")
        self.testfile = os.path.join(path, "test.libfm")
        self.validationfile = os.path.join(path, "validation.libfm")
        self.feature_num = self.gen_feature_map()
        print('feature_num:%d' % self.feature_num)

        self.train = self.read_data(self.trainfile)
        maxlen_train = max([len(i) for i in self.train[0]])

        self.validation = self.read_data(self.validationfile)
        maxlen_val = max([len(i) for i in self.validation[0]])

        self.test = self.read_data(self.testfile)
        maxlen_test = max([len(i) for i in self.test[0]])

        self.maxlen = max(maxlen_train, maxlen_val, maxlen_test)
        print('maxlen:%d' % self.maxlen)

        self.train = self.to_numpy(self.train, self.maxlen)
        self.validation = self.to_numpy(self.validation, self.maxlen)
        self.test = self.to_numpy(self.test, self.maxlen)

    def gen_feature_map(self):
        self.features = {}
        self._read_features(self.trainfile)
        self._read_features(self.testfile)
        self._read_features(self.validationfile)
        return len(self.features) + 1

    def _read_features(self, filepath):
        with open(filepath, 'r') as fp:
            for line in fp:
                for item in line.strip().split(' ')[1:]:
                    # 0 for pad_sequences
                    self.features.setdefault(item, len(self.features) + 1)
            
    def read_data(self, datafile):
        x, y = [], []
        with open(datafile, 'r') as fp:
            for line in fp:
                arr = line.strip().split(' ')
                if float(arr[0]) > 0:
                    y.append(1)
                else:
                    y.append(0)
                x.append([self.features[item] for item in arr[1:]])
        return x, y

    def to_numpy(self, data, maxlen):
        x, y = data
        maxlen = max([len(i) for i in x])
        x = pad_sequences(x, maxlen=maxlen)
        return (
            np.array(x, dtype=np.int64),  
            np.array(y, dtype=np.int64)
        )
