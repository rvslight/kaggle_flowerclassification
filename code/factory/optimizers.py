import tensorflow as tf
import os
os.environ['TF_KERAS'] = '1'
from keras_radam import RAdam

def adam(parameters, lr):
    return tf.keras.optimizers.Adam(learning_rate=lr)

def radam(_, lr):

    return RAdam(learning_rate = lr)

def sgd(parameters, lr, momentum=0.9, weight_decay=0, **_):
    return tf.keras.optimizers.SGD(lr=lr,  momentum=momentum,decay=weight_decay)


def get_optimizer(config, parameters=None):
    print('optimizer name:', config.OPTIMIZER.NAME)
    f = globals().get(config.OPTIMIZER.NAME)
    if config.OPTIMIZER.PARAMS is None:
        return f(parameters, config.OPTIMIZER.LR)
    else:
        return f(parameters, config.OPTIMIZER.LR, **config.OPTIMIZER.PARAMS)
