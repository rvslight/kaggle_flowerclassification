import os
import random
import pprint
import numpy as np
from dataset import DataSet
import tensorflow as tf
from factory.optimizers import get_optimizer
from factory.schedulers import get_scheduler
from factory.losses import get_loss
import utils.config
import utils.checkpoint
from utils.tools import prepare_train_directories, display_batch_of_images
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard, Callback
from model.model_factory import get_model
from pathlib import Path
import shutil
import psutil

isLocal = False
if "D:" in os.getcwd():
    isLocal = True


psutil.cpu_count(logical = True)
total_logical_cpu_count = psutil.cpu_count(logical = True)
print("logical_cpu_count: ",total_logical_cpu_count)

if isLocal:
    num_cores = total_logical_cpu_count ##// 2
else: #server
    print("!!!!!!!!!!!Server .. So half logical core!!!!!!!!!!!!!!!!!!")
    num_cores = total_logical_cpu_count // 2
print("set_num_cores: ",num_cores)
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)


def train(config, dataset, optimizer, criterion, scheduler, metric, strategy):
        if config.DATA.KFOLD is not -1:
            print('kfold: ',config.DATA.KFOLD,' fold_ind: ', config.DATA.FOLD_IND)
            dataset.set_train_cross_validate_ind(folds=config.DATA.KFOLD,fold_index=config.DATA.FOLD_IND)
        train_loader, NUM_TRAINING_IMAGES, step_per_epoch = dataset.get_training_dataset()
        val_loader, NUM_VALIDATION_IMAGES = dataset.get_validation_dataset(True)

        if config.TRAIN.DISPLAY:
            train_batch = iter(train_loader.unbatch().batch(3))
            for fi in range(NUM_TRAINING_IMAGES):
                x = next(train_batch)
                display_batch_of_images(x, figsize=8)

        if config.SKIP_VALIDATION: # only loss
            checkpoints_path = str(Path(
                config.TRAIN_DIR) / config.RECIPE / 'checkpoints') + '\\epoch_{epoch:03d}_{loss:.4f}.h5'
        else:
            if  config.MODEL.METRIC == 'accuracy':
                checkpoints_path = str(Path(
                    config.TRAIN_DIR) / config.RECIPE / 'checkpoints') + '\\epoch_{epoch:03d}_{val_accuracy:.4f}.h5'
            elif config.MODEL.METRIC == 'categorical_accuracy':
                checkpoints_path = str(Path(
                    config.TRAIN_DIR) / config.RECIPE / 'checkpoints') + '\\epoch_{epoch:03d}_{val_categorical_accuracy:.4f}.h5'
            else:
                checkpoints_path = str(Path(
                    config.TRAIN_DIR) / config.RECIPE / 'checkpoints') + '\\epoch_{epoch:03d}_{val_sparse_categorical_accuracy:.4f}.h5'

        checkpoint_all = ModelCheckpoint(
            checkpoints_path,
            # monitor='val_'+config.MODEL.METRIC,
            monitor='loss',
            mode='min',
            verbose=1,
            save_best_only=True,
            period=1
        )

        # checkpoint_all = ModelCheckpoint(
        #     checkpoints_path,
        #     monitor='val_'+config.MODEL.METRIC,
        #     mode='max',
        #     verbose=1,
        #     save_best_only=True,
        #     period=1
        # )

        log_dir = Path(config.TRAIN_DIR)/config.RECIPE/'logs'
        print(log_dir)
        tensorboard = TensorBoard(log_dir = log_dir,
            histogram_freq=0, write_graph=False, write_images=False,
            update_freq='epoch', profile_batch=0, embeddings_freq=0,
            embeddings_metadata=None
        )
        # with tf.device('/CPU:0'):
        with strategy.scope():
            model = get_model(config)
            checkpoint = utils.checkpoint.get_initial_checkpoint(config)
            if checkpoint is not None:
                utils.checkpoint.load_checkpoint(model, checkpoint)
            else:
                print('[*] no checkpoint found')

            model.compile(
                optimizer=optimizer,
                loss=criterion,
                metrics=metric
            )
            # print(model.metrics_name)
            # model.summary()

        history = model.fit(
            train_loader,
            steps_per_epoch=step_per_epoch,
            epochs=config.TRAIN.NUM_EPOCHS,
            callbacks=[scheduler, tensorboard,checkpoint_all],#checkpoint_all, ],
            validation_data=None if config.SKIP_VALIDATION and not config.DEBUG else val_loader
        )

        print("End! Goodluck~")


def setStrategy(config):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    ## TPU Strategy check
    # TPU detection
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    except ValueError:
        tpu = None

    # TPUStrategy for distributed training
    if tpu:
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.experimental.TPUStrategy(tpu)
    else:
        if not config.PARALLEL:
            os.environ["CUDA_VISIBLE_DEVICES"] = "1" ## just single model
        # default strategy that works on CPU and single GPU
            strategy = tf.distribute.get_strategy()
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0,1" ## just single model
            strategy = tf.distribute.MirroredStrategy(
                    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
    print('Strategy: ',strategy)

    AUTO = tf.data.experimental.AUTOTUNE
    return strategy, AUTO

def run(config):
    strategy, AUTO = setStrategy(config)

    optimizer = get_optimizer(config)
    scheduler = get_scheduler(config, num_replicas_in_sync = strategy.num_replicas_in_sync)
    criterion = get_loss(config)
    # if 'focal' in config.LOSS.NAME:
    #     if 'categorical_focal_loss' == config.LOSS.NAME:
    #         metric = ['sparse_categorical_accuracy']
    #     else:
    #         metric = ['accuracy']
    # else:
    #     metric = ['sparse_categorical_accuracy']
    metric = [config.MODEL.METRIC]
    # writer = tf.summary.create_file_writer(os.path.join(config.TRAIN_DIR+config.RECIPE, 'logs'))
    # writer.set_as_default()
    if 'focal' in config.LOSS.NAME or 'categorical_crossentropy' == config.LOSS.NAME:
        isOneHot=True
    else:
        isOneHot=False

    dataset = DataSet(config,num_replicas_in_sync = strategy.num_replicas_in_sync, AUTO = AUTO, isOneHot=isOneHot)


    train(config, dataset, optimizer,
          criterion, scheduler, metric, strategy)


def seed_everything():
    seed = 2019
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def main():

    import warnings
    warnings.filterwarnings("ignore")

    print('start training.')
    seed_everything()

    ymls = ['configs/base.yml']
    yml = ymls[0]
    config = utils.config.load(yml)

    if not isLocal:
        ## server so single.. not paraller
        print("!!!!!!!!!!!No Parallel!!!!!!!!!!")
        config.PARALLEL = False

    # exp. input_size
    input_size_list = config.EXPERIMENT.INPUT_SIZE_LIST
    batch_size_list = config.EXPERIMENT.BATCH_SIZE_LIST
    model_list = config.EXPERIMENT.MODEL_LIST
    original_recipe = config.RECIPE

    for input_size in input_size_list:

        config.DATA.IMG_H = input_size
        config.DATA.IMG_W = input_size

        for batch_size in batch_size_list:
            config.TRAIN.BATCH_SIZE = batch_size
            config.EVAL.BATCH_SIZE = batch_size * 2

            for selected_model_name in model_list:
                # config.TRAIN.BATCH_SIZE = batch_size
                config.MODEL.NAME = selected_model_name
                if selected_model_name == "EfficientNetB7":
                    config.TRAIN.BATCH_SIZE = 4
                    config.EVAL.BATCH_SIZE = config.TRAIN.BATCH_SIZE * 2
                    config.MODEL.WEIGHT= 'noisy-student'
                if config.DEBUG:
                    config.RECIPE = "debug_"+original_recipe
                else:
                    config.RECIPE = original_recipe
                ## recipe set batch_input_
                config.RECIPE = "_".join((config.RECIPE,config.MODEL.NAME, 'B'+str(config.TRAIN.BATCH_SIZE), 'S'+str(config.DATA.IMG_W)+'x'+str(config.DATA.IMG_H), 'LR'+str(config.OPTIMIZER.LR)))

                ##delete log_settings
                if config.DEBUG or config.DELETE_LOGPATH:
                    path = os.path.join(config.TRAIN_DIR,config.RECIPE)
                    if os.path.exists(path):
                        shutil.rmtree(path)
                        print("Delete path: ",path)

                ##delete all debug path
                train_path_list = os.listdir(config.TRAIN_DIR)
                for path_recipe in train_path_list:
                    if 'debug' in path_recipe:
                        path = os.path.join(config.TRAIN_DIR, path_recipe)
                        if os.path.exists(path):
                            shutil.rmtree(path)
                            print("Delete debug_path: "+path)

                prepare_train_directories(config)
                pprint.pprint(config, indent=2)
                utils.config.save_config(yml, config.TRAIN_DIR+config.RECIPE)
                run(config)

    print('success!')


if __name__ == '__main__':
    main()
