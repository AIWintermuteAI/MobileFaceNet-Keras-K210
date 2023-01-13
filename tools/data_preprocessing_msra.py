# -*- coding: utf-8 -*-
"""
Created on Wed Oct 23 16:50:39 2019

@author: TMaysGGS
"""

"""Data Preprocessing for MobileFaceNet"""
'''Last updated on 10/24/2019 11:09'''
"""Importing the libraries & basic settings"""
import os
import cv2
import random
import numpy as np
import sys
#from mtcnn.mtcnn import MTCNN
from tqdm import trange
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables import Keypoint, KeypointsOnImage

def create_augment_pipeline():

    aug_pipe = iaa.Sequential(
        [
            iaa.Fliplr(0.5),
            iaa.SomeOf((0, 1),
                       iaa.OneOf([
                               iaa.GaussianBlur((0, 3.0)),
                               iaa.AverageBlur(k=(2, 7)),
                               iaa.MedianBlur(k=(3, 11)),
                           ])),
                           iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)),
                           iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
                           iaa.OneOf([
                               iaa.Dropout((0.01, 0.1), per_channel=0.5),
                               iaa.CoarseDropout((0.03, 0.15), size_percent=(0.02, 0.05), per_channel=0.2),
                           ]),
                           iaa.Add((-10, 10), per_channel=0.5),
                           iaa.Multiply((0.5, 1.5), per_channel=0.5),
                           iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5)
        ],
        random_order=True
    )

    return aug_pipe



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
        transformer = skimage.transform.SimilarityTransform()
        transformer.estimate(keypoints, src)
        M = transformer.params[0: 2, : ]
        warped_img = cv2.warpAffine(img, M, (IMG_SHAPE[1], IMG_SHAPE[0]), borderValue = 0.0)

        return warped_img


def read_img(path):
    img = cv2.imread(path)
    img = img[91:347, 70:326]
    img = cv2.resize(img, (128, 128))
    return img

def data_clean_and_augmentation(input_path, output_path):
    augment = create_augment_pipeline()
    nums_of_imgs = []
    paths_dir = []
    folders = os.listdir(input_path)
    label = 0
    for i in trange(len(folders)):
        directory = folders[i]
        if os.path.isdir(input_path + directory) and not os.path.exists(output_path + str(label)):
            path_dir = os.listdir(input_path + directory)
            paths_dir.append(path_dir)

            num_of_imgs = len(path_dir)
            nums_of_imgs.append(num_of_imgs)
            os.makedirs(output_path + str(label))
            if num_of_imgs > 100:
                samples = random.sample(path_dir, 100)
            else:
                samples = path_dir
                while len(samples) < 100:
                    samples.append(random.sample(path_dir, 1)[0])

            for j in range(len(samples)):
                img = read_img(input_path + directory + '/' + samples[j])
                #warped_img = face_alignment(img, detector)
                if j >= num_of_imgs:
                    img = augment(image=img)
                if img is not None:
                    saving_path = output_path + str(label) + '/' + str(random.randint(0, 10000000)) + ".jpg"
                    success = cv2.imwrite(saving_path, img)
                    if not success:
                        raise Exception("img " + name + " saving failed. ")
        label += 1

        """
        for name in samples:
            img = cv2.imread(input_path + directory + '/' + name)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            warped_img = face_alignment(img, detector)
            if warped_img is not None:
                saving_path = output_path + directory + '/' + name
                print(saving_path)
                success = cv2.imwrite(saving_path, warped_img)
                if not success:
                    raise Exception("img " + name + " saving failed. ")

        """
if __name__ == '__main__':
    input_path = sys.argv[1] # Source image folder path
    output_path = sys.argv[2] # Destination image folder path
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    IMG_SHAPE = (128, 128) # in HW form
    #detector = MTCNN()
    src = np.array([[52, 51],
                    [92, 51],
                    [72, 74],
                    [55, 87],
                    [89, 87]], dtype = np.float32)
    if IMG_SHAPE == (112, 112):
        src[:, 0] = src[:, 0] + 8.0
    data_clean_and_augmentation(input_path, output_path)
