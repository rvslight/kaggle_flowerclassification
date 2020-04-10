import pandas as pd
import os
import tensorflow as tf
import cv2

import math, re, os
import tensorflow as tf, tensorflow.keras.backend as K
import numpy as np
from matplotlib import pyplot as plt
import numpy as np
import tensorflow.keras as keras
import cv2
import os
from imgaug import augmenters as iaa
from utils.global_var import CLASSES
from sklearn.model_selection import KFold
# BASE_DIR = os.path.dirname(__file__)
# IMAGES_DIR = os.path.join(BASE_DIR, 'dataset', 'images')


class DataSet():
    'Generates data for Keras'
    ##jitter is transform true or false
    def __init__(self, config, num_replicas_in_sync=1, AUTO=None, isOneHot=False):
        'Initialization'
        self.config = config

        self.image_h = config.DATA.IMG_H
        self.image_w = config.DATA.IMG_W
        self.AUTO = AUTO
        self.tta = config.TTA
        print('TTA: ',self.tta)
        if num_replicas_in_sync == -1: ## test
            self.BATCH_SIZE = self.config.EVAL.BATCH_SIZE
        else:
            self.BATCH_SIZE = self.config.TRAIN.BATCH_SIZE * num_replicas_in_sync ## num_replicas_in_sync is 2

        GCS_DS_PATH = config.DATA_DIR  # KaggleDatasets().get_gcs_path()
        GCS_PATH_SELECT = {  # available image sizes
            192: GCS_DS_PATH + '/tfrecords-jpeg-192x192',
            224: GCS_DS_PATH + '/tfrecords-jpeg-224x224',
            331: GCS_DS_PATH + '/tfrecords-jpeg-331x331',
            512: GCS_DS_PATH + '/tfrecords-jpeg-512x512'
        }
        GCS_PATH = GCS_PATH_SELECT[self.config.DATA.IMG_H]

        self.TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/train/*.tfrec')
        self.VALIDATION_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/val/*.tfrec')
        self.TEST_FILENAMES = tf.io.gfile.glob(
            GCS_PATH + '/test/*.tfrec')  # predictions on this dataset should be submitted for the competition

        # watch out for overfitting!
        self.SKIP_VALIDATION = self.config.SKIP_VALIDATION

        if self.config.DEBUG:
            if self.config.DATA.KFOLD != -1:
                self.TRAINING_FILENAMES = self.TRAINING_FILENAMES[:4]
                self.VALIDATION_FILENAMES = self.VALIDATION_FILENAMES[:1]
            else:
                self.TRAINING_FILENAMES = self.TRAINING_FILENAMES[:1]
                self.VALIDATION_FILENAMES = self.VALIDATION_FILENAMES[:1]
        else:
            if self.SKIP_VALIDATION:
                self.TRAINING_FILENAMES = self.TRAINING_FILENAMES + self.VALIDATION_FILENAMES

        self.VALIDATION_MISMATCHES_IDS = []
        self.KFOLD_TRAINING_FILENAMES = []
        self.trn_ind = None
        self.val_ind = None

        self.isOneHot = isOneHot

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.ceil(float(len(self.dataset)) / self.batch_size))

    def data_augment(self, image, label):
        # data augmentation. Thanks to the dataset.prefetch(AUTO) statement in the next function (below),
        # this happens essentially for free on TPU. Data pipeline code is executed on the "CPU" part
        # of the TPU while the TPU itself is computing gradients.
        image = tf.image.random_flip_left_right(image)
        # image = tf.image.random_brightness(image,0.15,seed=27)
        # image = tf.image.random_contrast(image, 0, 2) #lower, upper
        # if np.random.uniform()>0.8:
        #     image = self.transform(image,label)
        # image = tf.image.random_saturation(image, 0, 2)
        return image, label

    def get_mat(self, rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
        # returns 3x3 transformmatrix which transforms indicies

        # CONVERT DEGREES TO RADIANS
        rotation = math.pi * rotation / 180.
        shear = math.pi * shear / 180.

        # ROTATION MATRIX
        c1 = tf.math.cos(rotation)
        s1 = tf.math.sin(rotation)
        one = tf.constant([1], dtype='float32')
        zero = tf.constant([0], dtype='float32')
        rotation_matrix = tf.reshape(tf.concat([c1, s1, zero, -s1, c1, zero, zero, zero, one], axis=0), [3, 3])

        # SHEAR MATRIX
        c2 = tf.math.cos(shear)
        s2 = tf.math.sin(shear)
        shear_matrix = tf.reshape(tf.concat([one, s2, zero, zero, c2, zero, zero, zero, one], axis=0), [3, 3])

        # ZOOM MATRIX
        zoom_matrix = tf.reshape(
            tf.concat([one / height_zoom, zero, zero, zero, one / width_zoom, zero, zero, zero, one], axis=0), [3, 3])

        # SHIFT MATRIX
        shift_matrix = tf.reshape(tf.concat([one, zero, height_shift, zero, one, width_shift, zero, zero, one], axis=0),
                                  [3, 3])

        return K.dot(K.dot(rotation_matrix, shear_matrix), K.dot(zoom_matrix, shift_matrix))

    def cutmix(self, image, label, PROBABILITY=1.0):
        # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
        # output - a batch of images with cutmix applied
        DIM = self.config.DATA.IMG_W
        CLASSES = 104

        imgs = []
        labs = []
        for j in range(self.BATCH_SIZE):
            # DO CUTMIX WITH PROBABILITY DEFINED ABOVE
            P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.int32)
            # CHOOSE RANDOM IMAGE TO CUTMIX WITH
            k = tf.cast(tf.random.uniform([], 0, self.BATCH_SIZE), tf.int32)
            # CHOOSE RANDOM LOCATION
            x = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
            y = tf.cast(tf.random.uniform([], 0, DIM), tf.int32)
            b = tf.random.uniform([], 0, 1)  # this is beta dist with alpha=1.0
            WIDTH = tf.cast(DIM * tf.math.sqrt(1 - b), tf.int32) * P
            ya = tf.math.maximum(0, y - WIDTH // 2)
            yb = tf.math.minimum(DIM, y + WIDTH // 2)
            xa = tf.math.maximum(0, x - WIDTH // 2)
            xb = tf.math.minimum(DIM, x + WIDTH // 2)
            # MAKE CUTMIX IMAGE
            one = image[j, ya:yb, 0:xa, :]
            two = image[k, ya:yb, xa:xb, :]
            three = image[j, ya:yb, xb:DIM, :]
            middle = tf.concat([one, two, three], axis=1)
            img = tf.concat([image[j, 0:ya, :, :], middle, image[j, yb:DIM, :, :]], axis=0)
            imgs.append(img)
            # MAKE CUTMIX LABEL
            a = tf.cast(WIDTH * WIDTH / DIM / DIM, tf.float32)
            if len(label.shape) == 1:
                lab1 = tf.one_hot(label[j], CLASSES)
                lab2 = tf.one_hot(label[k], CLASSES)
            else:
                lab1 = label[j,]
                lab2 = label[k,]
            labs.append((1 - a) * lab1 + a * lab2)

        # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
        image2 = tf.reshape(tf.stack(imgs), (self.BATCH_SIZE, DIM, DIM, 3))
        label2 = tf.reshape(tf.stack(labs), (self.BATCH_SIZE, CLASSES))
        return image2, label2

    def mixup(self, image, label, PROBABILITY=1.0):
        # input image - is a batch of images of size [n,dim,dim,3] not a single image of [dim,dim,3]
        # output - a batch of images with mixup applied
        DIM = self.config.DATA.IMG_W
        CLASSES = 104

        imgs = []
        labs = []
        for j in range(self.BATCH_SIZE):
            # DO MIXUP WITH PROBABILITY DEFINED ABOVE
            P = tf.cast(tf.random.uniform([], 0, 1) <= PROBABILITY, tf.float32)
            # CHOOSE RANDOM
            k = tf.cast(tf.random.uniform([], 0, self.BATCH_SIZE), tf.int32)
            a = tf.random.uniform([], 0, 1) * P  # this is beta dist with alpha=1.0
            # MAKE MIXUP IMAGE
            img1 = image[j,]
            img2 = image[k,]
            imgs.append((1 - a) * img1 + a * img2)
            # MAKE CUTMIX LABEL
            if len(label.shape) == 1:
                lab1 = tf.one_hot(label[j], CLASSES)
                lab2 = tf.one_hot(label[k], CLASSES)
            else:
                lab1 = label[j,]
                lab2 = label[k,]
            labs.append((1 - a) * lab1 + a * lab2)

        # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
        image2 = tf.reshape(tf.stack(imgs), (self.BATCH_SIZE, DIM, DIM, 3))
        label2 = tf.reshape(tf.stack(labs), (self.BATCH_SIZE, CLASSES))
        return image2, label2


    def cutmix_mixup_transform(self, image, label):
        # THIS FUNCTION APPLIES BOTH CUTMIX AND MIXUP
        DIM = self.config.DATA.IMG_W
        CLASSES = 104
        SWITCH = 0.5
        CUTMIX_PROB = 0.666
        MIXUP_PROB = 0.666
        # FOR SWITCH PERCENT OF TIME WE DO CUTMIX AND (1-SWITCH) WE DO MIXUP
        image2, label2 = self.cutmix(image, label, CUTMIX_PROB)
        image3, label3 = self.mixup(image, label, MIXUP_PROB)
        imgs = []
        labs = []
        for j in range(self.BATCH_SIZE):
            P = tf.cast(tf.random.uniform([], 0, 1) <= SWITCH, tf.float32)
            imgs.append(P * image2[j,] + (1 - P) * image3[j,])
            labs.append(P * label2[j,] + (1 - P) * label3[j,])
        # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR (maybe use Python typing instead?)
        image4 = tf.reshape(tf.stack(imgs), (self.BATCH_SIZE, DIM, DIM, 3))
        label4 = tf.reshape(tf.stack(labs), (self.BATCH_SIZE, CLASSES))
        return image4, label4

    def transform(self, image, label):
        # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
        # output - image randomly rotated, sheared, zoomed, and shifted
        DIM = self.config.DATA.IMG_W
        XDIM = DIM % 2  # fix for size 331

        rot = 15. * tf.random.normal([1], dtype='float32')
        shr = 5. * tf.random.normal([1], dtype='float32')
        h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
        w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / 10.
        h_shift = 16. * tf.random.normal([1], dtype='float32')
        w_shift = 16. * tf.random.normal([1], dtype='float32')

        # GET TRANSFORMATION MATRIX
        m = self.get_mat(rot, shr, h_zoom, w_zoom, h_shift, w_shift)

        # LIST DESTINATION PIXEL INDICES
        x = tf.repeat(tf.range(DIM // 2, -DIM // 2, -1), DIM)
        y = tf.tile(tf.range(-DIM // 2, DIM // 2), [DIM])
        z = tf.ones([DIM * DIM], dtype='int32')
        idx = tf.stack([x, y, z])

        # ROTATE DESTINATION PIXELS ONTO ORIGIN PIXELS
        idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
        idx2 = K.cast(idx2, dtype='int32')
        idx2 = K.clip(idx2, -DIM // 2 + XDIM + 1, DIM // 2)

        # FIND ORIGIN PIXEL VALUES
        idx3 = tf.stack([DIM // 2 - idx2[0,], DIM // 2 - 1 + idx2[1,]])
        d = tf.gather_nd(image, tf.transpose(idx3))

        return tf.reshape(d, [DIM, DIM, 3]), label

    def decode_image(self, image_data):
        image = tf.image.decode_jpeg(image_data, channels=3)
        image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
        image = tf.reshape(image, [self.config.DATA.IMG_H, self.config.DATA.IMG_W, 3])  # explicit size needed for TPU
        return image

    def read_unlabeled_tfrecord(self, example):
        UNLABELED_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
            # class is missing, this competitions's challenge is to predict flower classes for the test dataset
        }
        example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
        image = self.decode_image(example['image'])
        idnum = example['id']
        return image, idnum  # returns a dataset of image(s)

    def read_labeled_id_tfrecord(self,example):
        LABELED_ID_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
            "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        }
        example = tf.io.parse_single_example(example, LABELED_ID_TFREC_FORMAT)
        image = self.decode_image(example['image'])
        label = tf.cast(example['class'], tf.int32)
        idnum = example['id']
        return image, label, idnum  # returns a dataset of (image, label, idnum) triples

    def read_labeled_id_tfrecord_onehot(self,example):
        LABELED_ID_TFREC_FORMAT = {
            "image": tf.io.FixedLenFeature([], tf.string),  # tf.string means bytestring
            "class": tf.io.FixedLenFeature([], tf.int64),  # shape [] means single element
            "id": tf.io.FixedLenFeature([], tf.string),  # shape [] means single element
        }
        example = tf.io.parse_single_example(example, LABELED_ID_TFREC_FORMAT)
        image = self.decode_image(example['image'])
        label = tf.cast(example['class'], tf.int32)
        label = tf.one_hot(label, depth=len(CLASSES))
        idnum = example['id']
        return image, label, idnum  # returns a dataset of (image, label, idnum) triples

    def load_dataset(self, filenames, labeled=True, ordered=False):
        # Read from TFRecords. For optimal performance, reading from multiple files at once and
        # disregarding data order. Order does not matter since we will be shuffling the data anyway.

        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False  # disable order, increase speed

        dataset = tf.data.TFRecordDataset(filenames,
                                          num_parallel_reads=self.AUTO)  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order)  # uses data as soon as it streams in, rather than in its original order
        if self.isOneHot:
            dataset = dataset.map(self.read_labeled_id_tfrecord_onehot if labeled else self.read_unlabeled_tfrecord, num_parallel_calls=self.AUTO)
        else:
            dataset = dataset.map(self.read_labeled_id_tfrecord if labeled else self.read_unlabeled_tfrecord, num_parallel_calls=self.AUTO)

        return dataset

    def load_dataset_with_id(self, filenames, ordered=False):
        # Read from TFRecords. For optimal performance, reading from multiple files at once and
        # disregarding data order. Order does not matter since we will be shuffling the data anyway.

        ignore_order = tf.data.Options()
        if not ordered:
            ignore_order.experimental_deterministic = False  # disable order, increase speed

        dataset = tf.data.TFRecordDataset(filenames,
                                          num_parallel_reads=self.AUTO)  # automatically interleaves reads from multiple files
        dataset = dataset.with_options(
            ignore_order)  # uses data as soon as it streams in, rather than in its original order
        dataset = dataset.map(self.read_labeled_id_tfrecord, num_parallel_calls=self.AUTO)
        # returns a dataset of (image, label) pairs if labeled=True or (image, id) pairs if labeled=False
        return dataset

    def get_training_dataset(self):
        if self.trn_ind is not None: # kfold
            self.TRAINING_FILENAMES = list(pd.DataFrame({'TRAINING_FILENAMES': self.KFOLD_TRAINING_FILENAMES}).loc[self.trn_ind]['TRAINING_FILENAMES'])
        dataset = self.load_dataset(self.TRAINING_FILENAMES, labeled=True)
        dataset = dataset.filter(
            lambda image, label, idnum: tf.reduce_sum(tf.cast(idnum == self.VALIDATION_MISMATCHES_IDS, tf.int32)) == 0)
        dataset = dataset.map(lambda image, label, idnum: [image, label])
        # dataset = dataset.map(self.transform, num_parallel_calls=self.AUTO) ## spear, rotate, shift..
        dataset = dataset.map(self.data_augment, num_parallel_calls=self.AUTO) ## flip

        dataset = dataset.repeat()  # the training dataset must repeat for several epochs
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(self.AUTO)  # prefetch next batch while training (autotune prefetch buffer size)

        NUM_TRAINING_IMAGES = self.count_data_items(self.TRAINING_FILENAMES)
        STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // self.BATCH_SIZE
        return dataset, NUM_TRAINING_IMAGES, STEPS_PER_EPOCH

    def get_training_dataset_cutmix(self):
        if self.trn_ind is not None: # kfold
            self.TRAINING_FILENAMES = list(pd.DataFrame({'TRAINING_FILENAMES': self.KFOLD_TRAINING_FILENAMES}).loc[self.trn_ind]['TRAINING_FILENAMES'])
        dataset = self.load_dataset(self.TRAINING_FILENAMES, labeled=True)
        dataset = dataset.filter(
            lambda image, label, idnum: tf.reduce_sum(tf.cast(idnum == self.VALIDATION_MISMATCHES_IDS, tf.int32)) == 0)
        dataset = dataset.map(lambda image, label, idnum: [image, label])
        dataset = dataset.map(self.data_augment, num_parallel_calls=self.AUTO) ## flip and spear,rotate,scale transform include
        dataset = dataset.repeat()  # the training dataset must repeat for several epochs

        ## for cut mix
        dataset = dataset.batch(self.BATCH_SIZE)
        if self.config.DATA.CUTMIX_MIXUP:
            dataset = dataset.map(self.cutmix_mixup_transform,
                                  num_parallel_calls=self.AUTO)  ## cutmix, mixup, none 33/33/33/
        dataset = dataset.unbatch()
        ## end of cutmix
        dataset = dataset.shuffle(2048)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(self.AUTO)  # prefetch next batch while training (autotune prefetch buffer size)

        NUM_TRAINING_IMAGES = self.count_data_items(self.TRAINING_FILENAMES)
        STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // self.BATCH_SIZE
        return dataset, NUM_TRAINING_IMAGES, STEPS_PER_EPOCH

    def get_validation_dataset(self,ordered=False):
        if self.val_ind is not None:  # kfold
            self.VALIDATION_FILENAMES = list(
                pd.DataFrame({'TRAINING_FILENAMES': self.KFOLD_TRAINING_FILENAMES}).loc[self.val_ind][
                    'TRAINING_FILENAMES'])

        dataset = self.load_dataset(self.VALIDATION_FILENAMES, labeled=True, ordered=ordered)
        dataset = dataset.filter(
            lambda image, label, idnum: tf.reduce_sum(tf.cast(idnum == self.VALIDATION_MISMATCHES_IDS, tf.int32)) == 0)
        dataset = dataset.map(lambda image, label, idnum: [image, label])
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
        NUM_VALIDATION_IMAGES = (1 - self.SKIP_VALIDATION) * self.count_data_items(self.VALIDATION_FILENAMES)
        return dataset, NUM_VALIDATION_IMAGES

    ##kfold..
    def set_train_cross_validate_ind(self, folds=5, fold_index=0):
        kfold = KFold(folds, shuffle=True, random_state=42)
        self.KFOLD_TRAINING_FILENAMES = self.TRAINING_FILENAMES + self.VALIDATION_FILENAMES
        for fold, (trn_ind, val_ind) in enumerate(kfold.split(self.KFOLD_TRAINING_FILENAMES)):
            if fold == fold_index:
                self.trn_ind = trn_ind
                self.val_ind = val_ind
            else:
                continue


    def get_validation_dataset_with_id(self,ordered=False):
        dataset = self.load_dataset_with_id(self.VALIDATION_FILENAMES, ordered=ordered)
        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.cache()
        dataset = dataset.prefetch(self.AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
        return dataset

    def get_test_dataset(self, ordered=False):
        dataset = self.load_dataset(self.TEST_FILENAMES, labeled=False, ordered=ordered)
        if self.tta > 2:
            dataset = dataset.map(self.data_augment, num_parallel_calls=self.AUTO)  ## flip

        dataset = dataset.batch(self.BATCH_SIZE)
        dataset = dataset.prefetch(self.AUTO)  # prefetch next batch while training (autotune prefetch buffer size)
        NUM_TEST_IMAGES = self.count_data_items(self.TEST_FILENAMES)
        return dataset, NUM_TEST_IMAGES

    # counting number of unique ids in dataset
    def count_data_items(self,filenames):
        dataset = self.load_dataset(filenames, labeled=False)
        dataset = dataset.map(lambda image, idnum: idnum)
        dataset = dataset.filter(
            lambda idnum: tf.reduce_sum(tf.cast(idnum == self.VALIDATION_MISMATCHES_IDS, tf.int32)) == 0)
        uids = next(iter(dataset.batch(21000))).numpy().astype('U')
        return len(np.unique(uids))



    def size(self):
        return len(self.dataset)


if __name__ == '__main__':
    yml = 'configs/fastscnn_mv3_sj_add_data_1024.yml'
    config = utils.config.load(yml)
    dataset = Dataset(config, 'train', None)
    # dataloader = dataset.getDataloader()
    dataloader = dataset.DataGenerator("membrane/train", batch_size=2)
    for step, inputs in enumerate(dataloader):
        # input_image, target = tf.split(inputs, num_or_size_splits=[3, 3], axis=3)
        print(inputs[0].shape, inputs[1].shape)