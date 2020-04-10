import os
import sys
import numpy as np
from matplotlib import pyplot as plt
from utils.global_var import CLASSES
import math

def prepare_train_directories(config):
    out_dir = config.TRAIN_DIR+config.RECIPE
    os.makedirs(os.path.join(out_dir, 'checkpoints'), exist_ok=True)

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


class Logger(object):
    def __init__(self):
        self.terminal = sys.stdout  #stdout
        self.file = None

    def open(self, file, mode=None):
        if mode is None: mode ='a'
        self.file = open(file, mode)

    def write(self, message, is_terminal=0, is_file=1 ):
        if '\r' in message: is_file=0

        if is_terminal == 1:
            self.terminal.write(message)
            self.terminal.flush()
            #time.sleep(1)

        if is_file == 1:
            self.file.write(message)
            self.file.flush()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass


def mask2contour(mask, width=1):
    w = mask.shape[1]
    h = mask.shape[0]
    mask2 = np.concatenate([mask[:,width:],np.zeros((h,width))],axis=1)
    mask2 = np.logical_xor(mask,mask2)
    mask3 = np.concatenate([mask[width:,:],np.zeros((width,w))],axis=0)
    mask3 = np.logical_xor(mask,mask3)
    return np.logical_or(mask2,mask3)


def annotate_to_images(images, labels, preds,thresh=0.5):
  preds = preds>=thresh
  # preds[preds > thresh] = 1
  # preds[preds <= thresh] = 0
  annotated_images = []
  for item in zip(images, labels, preds):
    image = item[0]
    mask = item[1]
    pred = item[2]
    #
    ### image ###
    if image.shape[0] == 3:
      image = np.transpose(image, [1, 2, 0])

    image = input_to_image(image)

    # mask = np.transpose(mask, [1, 2, 0])
    # pred = np.transpose(pred, [1, 2, 0])

    image = image.astype('uint8')
    for index in range(1):
        image_with_mask = image.copy()
        mask_line, pred_line = mask2contour(mask[:,:,index]), mask2contour(pred[:,:,index])

        image_with_mask[mask_line == 1, :2] = 0
        image_with_mask[pred_line == 1, :1] = 255

        ## one channel
        # image_with_mask[np.expand_dims(mask_line,axis=0) == 1] = 0
        # image_with_mask[np.expand_dims(pred_line,axis=0) == 1] = 255
        del mask_line, pred_line
        image_with_mask = np.transpose(image_with_mask, [2, 0, 1]) # change to C,H,W

        annotated_images.append(image_with_mask)

        ## one_channel
        # annotated_images.append(image_with_mask) # C,H,W order..
    del mask, pred
  return annotated_images

MEAN = [0.485, 0.456, 0.406]
STD  = [0.229, 0.224, 0.225]
def input_to_image(input):#, rgb_mean=MEAN, rgb_std=STD):
    #input h,w,c
    input = (input+1) / 2
    image = (input - np.min(input)) / (np.max(input) - np.min(input))*255


    # input = input * 0.5 + 0.5
    # # input[:,:,0] = (input[:,:,0]*rgb_std[0]+rgb_mean[0])
    # # input[:,:,1] = (input[:,:,1]*rgb_std[1]+rgb_mean[1])
    # # input[:,:,2] = (input[:,:,2]*rgb_std[2]+rgb_mean[2])
    # image = (input*255).astype(np.uint8)
    return image



def batch_to_numpy_images_and_labels(data):
    images, labels = data
    numpy_images = images.numpy()
    numpy_labels = labels.numpy()
    if numpy_labels.dtype == object: # binary string in this case, these are image ID strings
        numpy_labels = [None for _ in enumerate(numpy_images)]
    # If no labels, only image IDs, return None for labels (this is the case for test data)
    return numpy_images, numpy_labels

def display_one_flower(image, title, subplot, red=False, titlesize=16):
    plt.subplot(*subplot)
    plt.axis('off')
    plt.imshow(image)
    if len(title) > 0:
        plt.title(title, fontsize=int(titlesize) if not red else int(titlesize/1.2), color='red' if red else 'black', fontdict={'verticalalignment':'center'}, pad=int(titlesize/1.5))
    return (subplot[0], subplot[1], subplot[2]+1)

def title_from_label_and_target(label, correct_label):
    if correct_label is None:
        return CLASSES[label], True
    correct = (label == correct_label)
    return "{} [{}{}{}]".format(CLASSES[label], 'OK' if correct else 'NO', u"\u2192" if not correct else '',
                                CLASSES[correct_label] if not correct else ''), correct



def display_batch_of_images(databatch, predictions=None, figsize=13.0):
    """This will work with:
    display_batch_of_images(images)
    display_batch_of_images(images, predictions)
    display_batch_of_images((images, labels))
    display_batch_of_images((images, labels), predictions)
    """
    # data
    images, labels = batch_to_numpy_images_and_labels(databatch)
    if labels is None:
        labels = [None for _ in enumerate(images)]

    # auto-squaring: this will drop data that does not fit into square or square-ish rectangle
    rows = int(math.sqrt(len(images)))
    cols = len(images) // rows

    # size and spacing
    FIGSIZE = figsize
    SPACING = 0.1
    subplot = (rows, cols, 1)
    if rows < cols:
        plt.figure(figsize=(FIGSIZE, FIGSIZE / cols * rows))
    else:
        plt.figure(figsize=(FIGSIZE / rows * cols, FIGSIZE))

    # display
    for i, (image, label) in enumerate(zip(images[:rows * cols], labels[:rows * cols])):
        title = '' if label is None else CLASSES[label]
        correct = True
        if predictions is not None:
            title, correct = title_from_label_and_target(predictions[i], label)
        dynamic_titlesize = FIGSIZE * SPACING / max(rows,
                                                    cols) * 40 + 3  # magic formula tested to work from 1x1 to 10x10 images
        subplot = display_one_flower(image, title, subplot, not correct, titlesize=dynamic_titlesize)

    # layout
    plt.tight_layout()
    if label is None and predictions is None:
        fig = plt.subplots_adjust(wspace=0, hspace=0)
    else:
        fig = plt.subplots_adjust(wspace=SPACING, hspace=SPACING)
    # plt.show()
    # plt.draw()
    # plt.waitforbuttonpress(0)
    plt.close(fig)


def display_confusion_matrix(cmat, score, precision, recall):
    plt.figure(figsize=(15, 15))
    ax = plt.gca()
    ax.matshow(cmat, cmap='Reds')
    ax.set_xticks(range(len(CLASSES)))
    ax.set_xticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left", rotation_mode="anchor")
    ax.set_yticks(range(len(CLASSES)))
    ax.set_yticklabels(CLASSES, fontdict={'fontsize': 7})
    plt.setp(ax.get_yticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    titlestring = ""
    if score is not None:
        titlestring += 'f1 = {:.3f} '.format(score)
    if precision is not None:
        titlestring += '\nprecision = {:.3f} '.format(precision)
    if recall is not None:
        titlestring += '\nrecall = {:.3f} '.format(recall)
    if len(titlestring) > 0:
        ax.text(101, 1, titlestring,
                fontdict={'fontsize': 18, 'horizontalalignment': 'right', 'verticalalignment': 'top',
                          'color': '#804040'})
    plt.show()


def display_training_curves(training, validation, title, subplot):
    if subplot % 10 == 1:  # set up the subplots on the first call
        plt.subplots(figsize=(10, 10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model ' + title)
    ax.set_ylabel(title)
    # ax.set_ylim(0.28,1.05)
    ax.set_xlabel('epoch')