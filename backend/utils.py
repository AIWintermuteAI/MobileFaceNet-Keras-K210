import numpy as np
import os
import sys
import argparse
from tensorflow.keras.models import Model, load_model
import tensorflow as tf
import tensorflow.keras as keras
sys.path.append('./')
from .MobileFaceNet import mobile_face_net_train, mobile_face_net, preprocess_input

AUTOTUNE = tf.data.experimental.AUTOTUNE

def get_dataset(tfrecord_save_dir, loss, data_split, batch_size):
    print('Importing the TFRecord(s)')
    num_labels = 10571 # 10571 for CASIA and 35275 for MSRA
    tfrecord_path_list = []
    for file_name in os.listdir(tfrecord_save_dir):
        if file_name[-9: ] == '.tfrecord':
            tfrecord_path_list.append(os.path.join(tfrecord_save_dir, file_name))

    dataset_size =  490871 #490871 for CASIA and 1763750 for MSRA
    # can also count with sum(1 for _ in tf.data.TFRecordDataset(tfrecord_path_list))
    # but that will take a long time

    image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            }

    if loss == 'arcface':
        def _read_tfrecord(serialized_example):

            example = tf.io.parse_single_example(serialized_example, image_feature_description)

            img = tf.image.decode_jpeg(example['image_raw'], channels = 3) # RGB rather than BGR!!!
            img = tf.cast(img, tf.float32)
            img = preprocess_input(img, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
            img_shape = [example['height'], example['width'], example['depth']]
            img = tf.reshape(img, img_shape)

            label = example['label']
            one_hot_label = tf.one_hot(label, num_labels)

            return {'input_1': img, 'input_2': one_hot_label}, one_hot_label

    else:
        def _read_tfrecord(serialized_example):

            example = tf.io.parse_single_example(serialized_example, image_feature_description)

            img = tf.image.decode_jpeg(example['image_raw'], channels = 3) # RGB rather than BGR!!!
            img = tf.cast(img, tf.float32)
            img = preprocess_input(img, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
            img_shape = [example['height'], example['width'], example['depth']]
            img = tf.reshape(img, img_shape)
            label = example['label']
            one_hot_label = tf.one_hot(label, num_labels)
            return img, one_hot_label

    train_size = int((1 - data_split) * dataset_size)
    val_size = int(data_split * dataset_size)

    ignore_order = tf.data.Options()
    ignore_order.experimental_deterministic = False

    image_dataset = tf.data.TFRecordDataset(tfrecord_path_list, num_parallel_reads=4)

    image_dataset = image_dataset.with_options(ignore_order)
    image_dataset = image_dataset.map(_read_tfrecord)

    dataset_train = image_dataset.take(train_size)
    dataset_val = image_dataset.skip(train_size)

    dataset_train = dataset_train.shuffle(2048)
    dataset_train = dataset_train.batch(batch_size)
    dataset_train = dataset_train.prefetch(buffer_size=AUTOTUNE)

    dataset_val = dataset_val.shuffle(1024)
    dataset_val = dataset_val.batch(batch_size)
    dataset_val = dataset_val.prefetch(buffer_size=AUTOTUNE)

    for image_feature in image_dataset:
        img_shape = image_feature[0].numpy().shape
        break

    return (num_labels, dataset_size, train_size, val_size, img_shape), dataset_train, dataset_val

def save_for_inference(model, path):
    print('Converting to inference model')
    pred_model = mobile_face_net(alpha=1, variant = 'MobileFaceNet-S', backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
    pred_model.summary()

    print('Extracting the weights & transfering to the prediction model')
    temp_weights_list = []
    for layer in model.layers:

        if 'dropout' in layer.name:
            continue
        temp_layer = model.get_layer(layer.name)
        temp_weights = temp_layer.get_weights()
        temp_weights_list.append(temp_weights)

    for i in range(len(pred_model.layers)):

        pred_model.get_layer(pred_model.layers[i].name).set_weights(temp_weights_list[i])

    print('Verifying the results')
    x = np.random.rand(1, 112, 112, 3)
    dense1_layer_model = Model(inputs=model.input, outputs=model.get_layer('flatten').output)
    y1 = dense1_layer_model.predict(x)[0]
    y2 = pred_model.predict(x)[0]
    for i in range(128):
        assert (y1[i] - y2[i]) < 1e-4

    print('Saving the model')
    new_path = os.path.join(path.split('.')[0] + '_infer.h5')
    pred_model.save(new_path)

if __name__ == '__main__':
    argparser = argparse.ArgumentParser(description='Convert training model to inference model')

    argparser.add_argument(
        '-p',
        '--model_path',
        default=None,
        help='Path to model')

    args = argparser.parse_args()
    print('Loading the training model')
    model = load_model(args.model_path)
    model.summary()
    save_for_inference(model, args.model_path)
