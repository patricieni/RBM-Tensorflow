import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np

class Batch(object):
    def __init__(self, X, batch_size, y=None):
        self.X = X
        self.batch_size = batch_size
        self.size = X.shape[0]
        self.y = y


    def get_batch(self):
        indices = np.random.choice(range(self.size), self.batch_size)
        if self.y is None:
            return self.X[indices, :]
        return self.X[indices, :], self.y[indices, :]

    def get_tensor_batch(self):
        bt = self.get_batch()
        return tf.stack(bt)
