import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa

class FocalLoss(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def focal_loss_with_logits(self, logits, targets, alpha, gamma, y_pred):
        weight_a = alpha * (1 - y_pred) ** gamma * targets
        weight_b = (1 - alpha) * y_pred ** gamma * (1 - targets)

        return (tf.math.log1p(tf.exp(-tf.abs(logits))) + tf.nn.relu(-logits)) * (weight_a + weight_b) + logits * weight_b


    def call(self, y_true, y_pred):
        y_pred = tf.clip_by_value(y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon())
        logits = tf.math.log(y_pred / (1 - y_pred))

        loss = self.focal_loss_with_logits(logits=logits, targets=y_true, alpha=self.alpha, gamma=self.gamma, y_pred=y_pred)

        # or reduce_sum and/or axis=-1
        return tf.reduce_mean(loss)

def binary_crossentropy():
    return 'binary_crossentropy'

def binary_focal_loss(gamma=2., alpha=.25):

    @tf.function()
    def binary_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred:  A tensor resulting from a sigmoid
        :return: Output tensor.
        """

        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
        pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))

        epsilon = tf.keras.backend.epsilon()
        # clip to prevent NaN's and Inf's
        pt_1 = tf.clip_by_value(pt_1, epsilon, 1. - epsilon)
        pt_0 = tf.clip_by_value(pt_0, epsilon, 1. - epsilon)

        return -tf.keras.backend.sum(alpha * tf.math.pow(1. - pt_1, gamma) * tf.math.log(pt_1)) \
               - tf.keras.backend.sum((1 - alpha) * tf.math.pow(pt_0, gamma) * tf.math.log(1. - pt_0))

    return binary_focal_loss_fixed


def categorical_focal_loss(gamma=2., alpha=.25):
    """
    Softmax version of focal loss.
           m
      FL = ∑  -alpha * (1 - p_o,c)^gamma * y_o,c * log(p_o,c)
          c=1
      where m = number of classes, c = class and o = observation
    Parameters:
      alpha -- the same as weighing factor in balanced cross entropy
      gamma -- focusing parameter for modulating factor (1-p)
    Default value:
      gamma -- 2.0 as mentioned in the paper
      alpha -- 0.25 as mentioned in the paper
    References:
        Official paper: https://arxiv.org/pdf/1708.02002.pdf
        https://www.tensorflow.org/api_docs/python/tf/keras/backend/categorical_crossentropy
    Usage:
     model.compile(loss=[categorical_focal_loss(alpha=.25, gamma=2)], metrics=["accuracy"], optimizer=adam)
    """
    def categorical_focal_loss_fixed(y_true, y_pred):
        """
        :param y_true: A tensor of the same shape as `y_pred`
        :param y_pred: A tensor resulting from a softmax
        :return: Output tensor.
        """
        y_pred = tf.cast(y_pred, tf.float32)
        y_true = tf.cast(y_true, tf.float32)
        # Scale predictions so that the class probas of each sample sum to 1
        y_pred /= tf.keras.backend.sum(y_pred, axis=-1, keepdims=True)

        # Clip the prediction value to prevent NaN's and Inf's
        epsilon = tf.keras.backend.epsilon()
        y_pred = tf.clip_by_value(y_pred, epsilon, 1. - epsilon)

        # Calculate Cross Entropy
        cross_entropy = -y_true * tf.math.log(y_pred)

        # Calculate Focal Loss
        loss = alpha * tf.math.pow(1 - y_pred, gamma) * cross_entropy

        # Sum the losses in mini_batch
        return tf.keras.backend.sum(loss, axis=1)

    return categorical_focal_loss_fixed


def binary_lovasz_loss2():
    def func(input, target):

        loss = 0.
        for n in range(input.shape[0]):
            for c in range(input.shape[1]):
                iflat = input[n, c].view(-1)
                tflat = target[n, c].view(-1)

                loss += lovasz_hinge_flat(iflat, tflat)

        loss = loss / (input.shape[0] * input.shape[1])
        return loss

    return func



def dice():
    @tf.function()
    def dice_loss(y_true, y_pred):
        numerator = 2 * tf.reduce_sum(y_true * y_pred, axis=-1)
        denominator = tf.reduce_sum(y_true + y_pred, axis=-1)
        loss = 1 - (numerator + 1) / (denominator + 1)
        return loss
    return dice_loss

def sparse_categorical_crossentropy():
    return 'sparse_categorical_crossentropy'

def categorical_crossentropy():
    return 'categorical_crossentropy'

def binary_crossentropy():
    return 'binary_crossentropy'

def get_loss(config):
    loss_name = config.LOSS.NAME
    print('loss name:', loss_name)
    f = globals().get(loss_name)
    return f()

if __name__ == '__main__':
    import time

    input = torch.rand(8, 4, 256, 512).cuda()
    target = torch.rand(8, 4, 256, 512).cuda()

    print(binary_lovasz_loss()(input, target).item())
    print(binary_lovasz(per_image=True)(input, target).item())
    print(binary_lovasz2(per_image=False)(input, target).item())


    # start = time.time()
    # for i in range(100):
        # out = binary_lovasz_loss()(input, target)
        # out = binary_lovasz()(input, target)
        # out = binary_lovasz2()(input, target)
    # print(time.time() - start)
