import tensorflow as tf
import numpy as np



def LearningRateScheduler(config, last_epoch=-1, num_replicas_in_sync=1):
    class LRFN():
        def __init__(self, lr, LB_MAX = 0.00005, LB_MIN= 0.00001, num_replicas_in_sync=num_replicas_in_sync):
            self.LR_START = lr
            self.LR_MAX = LB_MAX * num_replicas_in_sync
            self.LR_MIN = LB_MIN
            self.LR_RAMPUP_EPOCHS = 4 #5
            self.LR_SUSTAIN_EPOCHS = 6 #0
            self.LR_EXP_DECAY = .8

        def lrfn(self, epoch):
            if epoch < self.LR_RAMPUP_EPOCHS:
                lr = np.random.random_sample() * self.LR_START
            elif epoch < self.LR_RAMPUP_EPOCHS + self.LR_SUSTAIN_EPOCHS:
                lr = self.LR_MAX
            else:
                lr = (self.LR_MAX - self.LR_MIN) * self.LR_EXP_DECAY ** (
                            epoch - self.LR_RAMPUP_EPOCHS - self.LR_SUSTAIN_EPOCHS) + self.LR_MIN
            # tf.summary.scalar('learning rate', data=lr, step=epoch)
            return lr

        def get_lrfn(self):
            return self.lrfn

    return tf.keras.callbacks.LearningRateScheduler(LRFN(config.OPTIMIZER.LR, config.SCHEDULER.PARAMS.LB_MAX, config.SCHEDULER.PARAMS.LB_MIN, num_replicas_in_sync).get_lrfn(), verbose=True)

def multi_step(optimizer, last_epoch, milestones=[500, 5000], gamma=0.1, **_):
    if isinstance(milestones, str):
        milestones = eval(milestones)
    return lr_scheduler.MultiStepLR(optimizer, milestones=milestones,
                                     gamma=gamma, last_epoch=last_epoch)


def none(optimizer, last_epoch, **_):
    return lr_scheduler.StepLR(optimizer, step_size=10000000, last_epoch=last_epoch)


def get_scheduler(config,last_epoch=-1, num_replicas_in_sync=1):
    print('scheduler name:', config.SCHEDULER.NAME)
    f = globals().get(config.SCHEDULER.NAME)
    # if config.SCHEDULER.PARAMS is None:
    return f(config, last_epoch, num_replicas_in_sync)
    # else:
    #     return f(config, last_epoch, num_replicas_in_sync, **config.SCHEDULER.PARAMS)
