import os
import sys
import random
import pprint
import numpy as np
from dataset import DataSet
import tensorflow as tf
import efficientnet.tfkeras
from tensorflow.keras.models import load_model
from matplotlib import pyplot as plt
import utils.config
import utils.checkpoint
from utils.tools import prepare_train_directories, AverageMeter, Logger, display_batch_of_images, display_confusion_matrix, display_training_curves
from sklearn.metrics import f1_score, precision_score, recall_score, confusion_matrix
AUTO = tf.data.experimental.AUTOTUNE
from pathlib import Path
from utils.global_var import CLASSES
import pandas as pd

def get_wrong_label_check_and_f1_score_check(dataset, cmdataset, NUM_VALIDATION_IMAGES, cm_predictions, cm_correct_labels):
    cmdataset_with_id = dataset.get_validation_dataset_with_id(ordered=True)
    ids_ds = cmdataset_with_id.map(lambda image, label, idnum: idnum).unbatch()
    ids = next(iter(ids_ds.batch(NUM_VALIDATION_IMAGES))).numpy().astype('U')  # get everything as one batch

    val_batch = iter(cmdataset.unbatch().batch(1))
    noip = sum(cm_predictions != cm_correct_labels)
    print('Number of incorrect predictions: ' + str(noip) + ' (' + str(
        round(noip / NUM_VALIDATION_IMAGES * 100, 1)) + '%)')
    for fi in range(NUM_VALIDATION_IMAGES):
        x = next(val_batch)
        if cm_predictions[fi] != cm_correct_labels[fi]:
            print("Image id: '" + ids[fi] + "'")
            display_batch_of_images(x, np.array([cm_predictions[fi]]), figsize=2)

    cmat = confusion_matrix(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)))
    score = f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
    precision = precision_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
    recall = recall_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro')
    # cmat = (cmat.T / cmat.sum(axis=1)).T # normalized
    display_confusion_matrix(cmat, score, precision, recall)
    print('f1 score: {:.3f}, precision: {:.3f}, recall: {:.3f}'.format(score, precision, recall))

def get_best_weight(config, dataset, model, model2):
    cmdataset, NUM_VALIDATION_IMAGES = dataset.get_validation_dataset(True)
    images_ds = cmdataset.map(lambda image, label: image)
    labels_ds = cmdataset.map(lambda image, label: label).unbatch()
    cm_correct_labels = next(
        iter(labels_ds.batch(NUM_VALIDATION_IMAGES))).numpy()  # get everything as one batch
    m = model.predict(images_ds)
    m2 = model2.predict(images_ds)
    scores = []
    for alpha in np.linspace(0, 1, 100):
        cm_probabilities = alpha * m + (1 - alpha) * m2
        cm_predictions = np.argmax(cm_probabilities, axis=-1)
        scores.append(f1_score(cm_correct_labels, cm_predictions, labels=range(len(CLASSES)), average='macro'))

    print("Correct   labels: ", cm_correct_labels.shape, cm_correct_labels)
    print("Predicted labels: ", cm_predictions.shape, cm_predictions)
    plt.plot(scores)
    plt.show()
    best_alpha = np.argmax(scores) / 100
    cm_probabilities = best_alpha * m + (1 - best_alpha) * m2
    cm_predictions = np.argmax(cm_probabilities, axis=-1)
    print(best_alpha)
    if config.get_wrong_label_check_and_f1_score_check:
        get_wrong_label_check_and_f1_score_check(dataset,cmdataset,NUM_VALIDATION_IMAGES,cm_predictions, cm_correct_labels)
    return best_alpha



def run(config):
    # input_shape = (config.DATA.IMG_H, config.DATA.IMG_W, 3)
    # resnet50 = ResNet50(
    #     input_shape=input_shape,
    #     weights=None,
    #     include_top=False)
    # model = tf.keras.Sequential([
    #     resnet50,
    #     tf.keras.layers.GlobalAveragePooling2D(),
    #     tf.keras.layers.Dense(len(CLASSES), activation='softmax')
    # ])

    model1_path = config.MODEL1_INFER_DIR
    model = load_model(model1_path)
    assert model.input_shape[1:][0] == config.DATA.IMG_H

    model2 = None
    model2_path = config.MODEL2_INFER_DIR
    if model2_path != 'None':
        model2 = load_model(model2_path)
    dataset = DataSet(config, num_replicas_in_sync=-1, AUTO=AUTO, isOneHot=False)

    best_alpha = 0.50
    if config.get_best_weight:
        best_alpha = get_best_weight(config,dataset,model,model2)

    print('Computing predictions...')


    tta_probabilities = []
    if config.TTA > 0:
        if model2 is None: #single model
            print(f'TTA num: {config.TTA} 1 model predict')
            for i in range(config.TTA):
                print('TTA_Num: ', i)
                test_ds, NUM_TEST_IMAGES = dataset.get_test_dataset(
                    ordered=True)  # since we are splitting the dataset and iterating separately on images and ids, order matters.
                test_images_ds = test_ds.map(lambda image, idnum: image)
                tta_probabilities.append(model.predict(test_images_ds))
            probabilities = np.mean(tta_probabilities,axis=0)
        else:
            print(f'TTA num: {config.TTA} 2 model ensemble')
            for i in range(config.TTA):
                print('first model TTA_Num: ', i)
                test_ds, NUM_TEST_IMAGES = dataset.get_test_dataset(
                    ordered=True)  # since we are splitting the dataset and iterating separately on images and ids, order matters.
                test_images_ds = test_ds.map(lambda image, idnum: image)
                tta_probabilities.append(model.predict(test_images_ds))
            probabilities_1 = np.mean(tta_probabilities, axis=0)
            tta_probabilities.clear()
            for i in range(config.TTA):
                print('second model TTA_Num: ', i)
                test_ds, NUM_TEST_IMAGES = dataset.get_test_dataset(
                    ordered=True)  # since we are splitting the dataset and iterating separately on images and ids, order matters.
                test_images_ds = test_ds.map(lambda image, idnum: image)
                tta_probabilities.append(model.predict(test_images_ds))
            probabilities_2 = np.mean(tta_probabilities, axis=0)
            probabilities = best_alpha * probabilities_1 + (1 - best_alpha) * probabilities_2

    else:
        test_ds, NUM_TEST_IMAGES = dataset.get_test_dataset(
            ordered=True)  # since we are splitting the dataset and iterating separately on images and ids, order matters.
        test_images_ds = test_ds.map(lambda image, idnum: image)

        if model2 is None: #single model
            print("no tta 1 model predict")
            probabilities = model.predict(test_images_ds)
        else: #multi model
            print("no tta 2 model ensemble")
            probabilities = best_alpha * model.predict(test_images_ds) + (1 - best_alpha) * model2.predict(test_images_ds)

    predictions = np.argmax(probabilities, axis=-1)
    print(predictions)
    del test_images_ds
    print('Generating submission.csv file...')
    test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
    test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U')  # all in one batch
    np.savetxt('submission.csv', np.rec.fromarrays([test_ids, predictions]), fmt=['%s', '%d'], delimiter=',',
               header='id,label', comments='')

    ##make direct_submit csv file
    print("Generate direct_submission!!")
    make_direct_submit('direct_submission.csv')

def make_direct_submit(output_file):
    df = pd.read_csv('submission.csv')
    df = df.astype(str)
    df['direct'] = df[['id','label']].apply(lambda x: ", ".join(x), axis=1) #axis column direction. apply each row.
    df.to_csv(output_file, index=False)

def main():
    import warnings
    warnings.filterwarnings("ignore")

    print('start training.')

    ymls = ['configs/infer.yml']
    for yml in ymls:
        config = utils.config.load(yml)

        os.environ["CUDA_VISIBLE_DEVICES"] = '0'#,1'  # config.GPU
        prepare_train_directories(config)
        pprint.pprint(config, indent=2)
        utils.config.save_config(yml, config.TRAIN_DIR + config.RECIPE)
        run(config)



if __name__ == "__main__":
    main()

