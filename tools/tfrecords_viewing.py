# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:43:44 2020

@author: TMaysGGS
"""

"""Last updated on 2020.03.21 13:58"""
"""Importing the libraries"""
import os
import cv2
import sys
import random
import tensorflow as tf
import numpy as np
from tqdm import tqdm

IMG_SHAPE = (128, 128, 3)

"""Building helper functions"""
def _bytes_feature(value):

    '''Returns a bytes_list from a string / byte. '''
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy() # BytesList will not unpack a string from an EagerTensor.

    return tf.train.Feature(bytes_list = tf.train.BytesList(value = [value]))

def _float_feature(value):

    '''Returns a float_list from a float / double. '''
    if not isinstance(value, list):
        value = [value]
    return tf.train.Feature(float_list = tf.train.FloatList(value = value))

def _int64_feature(value):

    '''Returns an int64_list from bool / enum / int / uint. '''
    return tf.train.Feature(int64_list = tf.train.Int64List(value = [value]))

def convert_image_info_to_tfexample(anno):

    img_path = anno[0]
    label = int(anno[1])

    #img = cv2.imread(img_path)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #img = cv2.resize(img, (IMG_SHAPE[0], IMG_SHAPE[1]))
    image_bytes = open(img_path, 'rb').read()
    #image_bytes = tf.io.encode_jpeg(tf.cast(img, tf.uint8))

    feature = {
            'height': _int64_feature(IMG_SHAPE[0]),
            'width': _int64_feature(IMG_SHAPE[1]),
            'depth': _int64_feature(IMG_SHAPE[2]),
            'label': _int64_feature(label),
            'image_raw': _bytes_feature(image_bytes),
            }

    return tf.train.Example(features = tf.train.Features(feature = feature))


if __name__ == '__main__':
    # """Reading the images from TFRecord"""
    import tensorflow as tf

    tfrecord_save_path = sys.argv[1] # Source tfrecord path
    raw_image_dataset = tf.data.TFRecordDataset(tfrecord_save_path)

    image_feature_description = {
            'height': tf.io.FixedLenFeature([], tf.int64),
            'width': tf.io.FixedLenFeature([], tf.int64),
            'depth': tf.io.FixedLenFeature([], tf.int64),
            'label': tf.io.FixedLenFeature([], tf.int64),
            'image_raw': tf.io.FixedLenFeature([], tf.string),
            }

    def _parse_image_function(example_proto):

        return tf.io.parse_single_example(example_proto, image_feature_description)

    parsed_image_dataset = raw_image_dataset.map(_parse_image_function)

    for image_feature in parsed_image_dataset:
        img = tf.image.decode_jpeg(image_feature['image_raw'], channels = 3).numpy()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        label = image_feature['label']
        print(image_feature['height'].numpy(), image_feature['width'].numpy())
        print(label.numpy())
        cv2.imshow('image', img)
        k = cv2.waitKey(0)
        if k == 27:
            cv2.destroyAllWindows()
            break