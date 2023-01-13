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
    data_dir = sys.argv[1] # Source image folder path
    tfrecord_save_prefix = sys.argv[2] # Destination tfrecord

    print('Getting the folder info')
    label_list = os.listdir(data_dir)
    img_info_list = []
    print('Getting the img info')

    for label in label_list:
        img_name_list = os.listdir(os.path.join(data_dir, label))
        for img_name in img_name_list:
            img_path = os.path.join(data_dir, label, img_name)
            img_info_list.append([img_path, label])
    del label, img_name, img_path, label_list, img_name_list
    print('Preparing TFRecords')
    random.shuffle(img_info_list)
    img_num_per_tfrecord = 500000
    img_info_sections = []
    for i in range(int(np.ceil(len(img_info_list) / img_num_per_tfrecord))):
        temp_list = img_info_list[i * img_num_per_tfrecord: min((i + 1) * img_num_per_tfrecord, len(img_info_list))]
        img_info_sections.append(temp_list)

    print('Writing the images & labels into TFRecord')
    for i in range(len(img_info_sections)):
        print("Section: ", i)
        tfrecord_save_path = tfrecord_save_prefix + str(i) + r'.tfrecord'
        with tf.io.TFRecordWriter(tfrecord_save_path) as writer:
            for anno in tqdm(img_info_sections[i]):
                tf_example = convert_image_info_to_tfexample(anno)
                writer.write(tf_example.SerializeToString())