import os
import cv2
import skimage
import skimage.transform
import random
import numpy as np
import sys
import tensorflow as tf
from tqdm import trange

DEBUG = True
ALLIGN = True

if ALLIGN:
    from mtcnn.mtcnn import MTCNN

"""Building helper functions"""
def face_detection(img, detector):

    info = []
    results = detector.detect_faces(img)
    if len(results) == 0:
        return []
    elif len(results) > 0:
        for i in range(len(results)):
            result = results[i]
            confidence = result['confidence']
            box = np.array(result['box'], np.float32)
            keypoints_dict = result['keypoints']
            keypoints = []
            for key in keypoints_dict:
                keypoints.append(keypoints_dict[key])
            keypoints = np.array(keypoints, dtype = np.float32)

            info.append([confidence, box, keypoints])

        return info

def face_alignment(img, detector):

    info = face_detection(img, detector)
    if len(info) <= 0 or len(info) > 1:
        return None
    else:
        face_info = info[0]
        assert(len(face_info) == 3)
        keypoints = face_info[2]
        if DEBUG: print(keypoints)
        transformer = skimage.transform.SimilarityTransform()
        transformer.estimate(keypoints, src)
        M = transformer.params[0: 2, : ]
        warped_img = cv2.warpAffine(img, M, (IMG_SHAPE[1], IMG_SHAPE[0]), borderValue = 0.0)

        return warped_img

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

class TFRecordWriter:

    def __init__(self, output_path):

        self.img_num_per_tfrecord = 50000
        self.cur_img = 0
        self.cur_record = 0
        self.tfrecord_save_path = os.path.join(output_path, 'casia_' + str(self.cur_record) + r'.tfrecord')
        self.writer = tf.io.TFRecordWriter(self.tfrecord_save_path)
        self.img_shape = (112, 112, 3)

    def save_to_tfrecord(self, img, label):
        if self.cur_img == self.img_num_per_tfrecord:
            self.writer.close()
            self.cur_record += 1
            self.cur_img = 0
            self.tfrecord_save_path = os.path.join(output_path, 'casia_' + str(self.cur_record) + r'.tfrecord')
            self.writer = tf.io.TFRecordWriter(self.tfrecord_save_path)
        tf_example = self.convert_image_info_to_tfexample(img, label)
        self.writer.write(tf_example.SerializeToString())
        self.cur_img += 1

    def convert_image_info_to_tfexample(self, img, label):
        img_string = cv2.imencode('.jpg', img)[1].tostring()
        feature = {
                'height': _int64_feature(self.img_shape[0]),
                'width': _int64_feature(self.img_shape[1]),
                'depth': _int64_feature(self.img_shape[2]),
                'label': _int64_feature(label),
                'image_raw': _bytes_feature(img_string),
                }

        return tf.train.Example(features = tf.train.Features(feature = feature))


if __name__ == '__main__':
    input_path = sys.argv[1] # Source image folder path
    output_path = sys.argv[2] # Destination image folder path

    IMG_SHAPE = (128, 128) # in HW form
    if ALLIGN:
        detector = MTCNN()
    src = np.array([[52, 51],
                    [92, 51],
                    [72, 74],
                    [55, 87],
                    [89, 87]], dtype = np.float32)
    if IMG_SHAPE == (112, 112):
        src[:, 0] = src[:, 0] + 8.0

    tf_writer = TFRecordWriter(output_path)
    nums_of_imgs = []
    paths_dir = []
    clean_files = []
    clean_labels = []

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    with open(os.path.join(input_path, 'cleaned_list.txt')) as fp:
        samples = fp.readlines()
        for i in trange(len(samples)):
            file_path, label = samples[i].split(' ')
            folder, file_name = file_path.split('\\')
            img = cv2.imread(os.path.join(input_path, folder, file_name))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if ALLIGN:
                img = img[14:222, 59:207]
                img = cv2.resize(img, (128,128))
                warped_img = face_alignment(img, detector)
            else:
                img = img[14:222, 59:207]
                warped_img = cv2.resize(img, IMG_SHAPE)
            if warped_img is not None:
                if DEBUG:
                    cv2.imshow('img', img)
                    cv2.waitKey(0)
                else:
                    tf_writer.save_to_tfrecord(warped_img, int(label.strip()))
