# -*- coding: utf-8 -*-
"""
Created on Mon May 20 10:16:45 2019

@author: TMaysGGS
"""

'''Last updated on 2020.04.21 09:45'''
import os
import cv2
from skimage import io

img_dir = r'/data/daiwei/processed_data/datasets_for_face_recognition'
label_list = os.listdir(img_dir)

for label in label_list:
    
    print('Directory: ' + str(label))
    label_dir = os.path.join(img_dir, label)
    for name in os.listdir(label_dir):
        
        img_path = os.path.join(label_dir, name)
        cv_img = cv2.imread(img_path)
        try:
            assert(cv_img.shape == (112, 112, 3))
        except:
            print('Error dir: ' + str(label) + ', error img: ' + name + ' (shape error)')
        try:
            sk_img = io.imread(img_path)
            assert(sk_img.shape == (112, 112, 3))
        except:
            print('Error dir: ' + str(label) + ', error img: ' + name + ' (image io read error)')
            cv2.imwrite(img_path, cv_img)
            try:
                sk_img = io.imread(img_path)
                assert(sk_img.shape == (112, 112, 3))
            except:
                print('Re-saving failed. ')
print("Checking done.")