from backend.utils import get_dataset, save_for_inference
from backend.MobileFaceNet import mobile_face_net_train, mobile_face_net, preprocess_input
import os
import argparse
from datetime import datetime
import time
import sys
import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adamax
from tensorflow.keras.models import Model

sys.path.append('./backend')

gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)


def main(args):

    config_dict = {'train': [None, 'softmax', 1e-3], 'retrain': [
        args.weights, 'arcface', 1e-3], 'tune': [args.weights, 'arcface', 1e-4]}

    dataset_info, dataset_train, dataset_val = get_dataset(
        args.tfrecords, config_dict[args.session][1], args.data_split, args.batch_size)

    strategy = tf.distribute.MirroredStrategy()

    with strategy.scope():
        model = mobile_face_net_train(num_labels=dataset_info[0], loss=config_dict[args.session][1],
                                      input_shape=dataset_info[4], alpha=args.alpha,
                                      weights=config_dict[args.session][0], variant=args.variant,
                                      backend=keras.backend, layers=keras.layers,
                                      models=keras.models, utils=keras.utils)
        model.compile(optimizer=args.optimizer(
            config_dict[args.session][2], epsilon=1e-8), loss='categorical_crossentropy', metrics=['accuracy'])

    model.summary()

    train_date = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    path = os.path.join('models', train_date)
    basename = model.name + "_best_" + METRIC
    print('Current training session folder is {}'.format(path))
    print('''
    Dataset info:
    Number of labels {0[0]}
    Number of examples {0[1]}
    Number of training examples {0[2]}
    Number of validation examples {0[3]}
    '''.format(dataset_info))
    os.makedirs(path)
    save_weights_name = os.path.join(path, basename + '.h5')
    save_weights_name_ctrlc = os.path.join(path, basename + '_ctrlc.h5')
    print('\n')

    # Save the model after every epoch
    check_pointer = ModelCheckpoint(
        filepath=save_weights_name, verbose=1, save_best_only=True)

    # Interrupt the training when the validation loss is not decreasing
    early_stopping = EarlyStopping(monitor=METRIC, patience=6)

    # Reduce learning rate when a metric has stopped improving
    reduce_lr = ReduceLROnPlateau(
        monitor=METRIC, factor=0.2, patience=3, min_lr=0.000001)

    print('Training the model')
    hist = model.fit(
        dataset_train,
        validation_data=dataset_val,
        epochs=TOTAL_EPOCHS,
        callbacks=[check_pointer, early_stopping, reduce_lr])

    print(hist.history)
    save_for_inference(model, path)

if __name__ == '__main__':

    # default params, not used, left here for reference
    BATCH_SIZE = 256
    NUM_LABELS = 35275
    DATA_SPLIT = 0.001
    TOTAL_EPOCHS = 30
    LOSS_TYPE = 'softmax'
    OPTIMIZER = Adam
    LR = 0.001
    ALPHA = 1
    METRIC = "val_accuracy"

    argparser = argparse.ArgumentParser(
        description='Train MobileFaceNet face recognition model')

    argparser.add_argument(
        '-b',
        '--batch_size',
        default=256,
        help='batch_size')

    argparser.add_argument(
        '-d',
        '--data_split',
        default=0.001,
        help='data_split')

    argparser.add_argument(
        '-e',
        '--num_epochs',
        default=30,
        help='data_split')

    argparser.add_argument(
        '-s',
        '--session',
        default='train',
        help='train - Starting from scratch with softmax, retrain - Retraining with arcface or tune - Fine-tuning with arcface')

    argparser.add_argument(
        '-o',
        '--optimizer',
        default=Adam,
        help='optimizer')

    argparser.add_argument(
        '-a',
        '--alpha',
        default=1,
        help='alpha parameter for the network')

    argparser.add_argument(
        '-m',
        '--metric',
        default='val_loss',
        help='main metric')

    argparser.add_argument(
        '-w',
        '--weights',
        default=None,
        help='path to weights file')

    argparser.add_argument(
        '-v',
        '--variant',
        default='MobileFaceNet-S',
        help='backbone variant: MobileFaceNet-M, MobileFaceNet-S or Full')

    argparser.add_argument(
        '-t',
        '--tfrecords',
        default='/home/ubuntu/datasets/msra_tf',
        help='path to tfrecords dir')

    args = argparser.parse_args()
    main(args)
