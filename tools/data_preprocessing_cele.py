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
import skimage 
import random 
import numpy as np 
from mtcnn.mtcnn import MTCNN 

input_path = '/data/face_data/glin_data/celebrity/' # path of raw image data set
output_path = '/data/daiwei/processed_data/cele/' # path of output image data set 
if not os.path.exists(output_path): 
    os.makedirs(output_path) 
os.environ['CUDA_VISIBLE_DEVICES'] = ''
IMG_SHAPE = (112, 112) # in HW form 
detector = MTCNN() 
src = np.array([
      [30.2946, 51.6963],
      [65.5318, 51.5014],
      [48.0252, 71.7366],
      [33.5493, 92.3655],
      [62.7299, 92.2041]], dtype = np.float32)
if IMG_SHAPE == (112, 112): 
    src[:, 0] = src[:, 0] + 8.0  

"""Building helper functions"""
def face_detection(img, detector): 
    
    info = []
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
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

"""Data clean & augmentation"""
def data_clean_and_augmentation(input_path, output_path):
     
    nums_of_imgs = []
    paths_dir = []
    kernel = np.ones((5, 5), dtype = np.uint8)
    
    for directory in os.listdir(input_path): 
        
        # directory: label folder name
        print('\n' + directory +'\n')
        if os.path.isdir(input_path + directory) and not os.path.exists(output_path + directory):
            
            path_dir = os.listdir(input_path + directory) # The name list of the images in each label folder
            paths_dir.append(path_dir)
            
            num_of_imgs = len(path_dir) # The number of images in each label folder
            nums_of_imgs.append(num_of_imgs) # A list of the numbers of images in each label folder
            
            if num_of_imgs > 30:
                os.makedirs(output_path + directory)
                
            # For each label folder: 
            
            # (1) num > 350
            if num_of_imgs > 350:
                
                samples = random.sample(path_dir, 350) # Randomly pick 350 out from the overall set
                
                for name in samples:
                    
                    img = cv2.imread(input_path + directory + '/' + name)
                    warped_img = face_alignment(img, detector)
                    if warped_img is not None: 
                        saving_path = output_path + directory + '/' + name 
                        print(saving_path) 
                        success = cv2.imwrite(saving_path, warped_img) 
                        if not success: 
                            raise Exception("img " + name + " saving failed. ") 
                    
            # (2) 200 < num <= 350
            elif num_of_imgs > 200 and num_of_imgs <= 350:
                
                for name in path_dir:
                    
                    img = cv2.imread(input_path + directory + '/' + name)
                    warped_img = face_alignment(img, detector)
                    if warped_img is not None: 
                        saving_path = output_path + directory + '/' + name 
                        print(saving_path) 
                        success = cv2.imwrite(saving_path, warped_img) 
                        if not success: 
                            raise Exception("img " + name + " saving failed. ") 
         
            # (3) 90 < num <= 200
            elif num_of_imgs > 90 and num_of_imgs <= 200:
                
                for name in path_dir:
                    
                    img = cv2.imread(input_path + directory + '/' + name)
                    warped_img = face_alignment(img, detector)
                    if warped_img is not None: 
                        saving_path_1 = output_path + directory + '/' + name 
                        print(saving_path_1) 
                        success = cv2.imwrite(saving_path_1, warped_img) 
                        if not success: 
                            raise Exception("img " + name + " saving failed. ") 
                    
                        # Opening
                        temp_img = cv2.morphologyEx(warped_img, cv2.MORPH_OPEN, kernel)
                        
                        temp_img_name = 'Open_' + name
                        saving_path_2 = output_path + directory + '/' + temp_img_name
                        print(saving_path_2)
                        success = cv2.imwrite(saving_path_2, temp_img) 
                        if not success: 
                            raise Exception("img " + temp_img_name + " saving failed. ") 
                            
            # (4) 30 < num <= 90
            elif num_of_imgs > 30 and num_of_imgs <= 90:
                
                for name in path_dir:
                    
                    img = cv2.imread(input_path + directory + '/' + name)
                    warped_img = face_alignment(img, detector)
                    if warped_img is not None: 
                        
                        saving_path_1 = output_path + directory + '/' + name
                        print(saving_path_1)
                        success = cv2.imwrite(saving_path_1, warped_img) 
                        if not success: 
                            raise Exception("img " + name + " saving failed. ") 
                        
                        # Opening
                        temp_img = cv2.morphologyEx(warped_img, cv2.MORPH_OPEN, kernel)
                        
                        temp_img_name = 'Open_' + name
                        saving_path_2 = output_path + directory + '/' + temp_img_name
                        print(saving_path_2) 
                        success = cv2.imwrite(saving_path_2, temp_img) 
                        if not success: 
                            raise Exception("img " + temp_img_name + " saving failed. ") 
                        
                        # Add Gaussian noise
                        temp_img_2 = skimage.util.random_noise(warped_img, mode = 'gaussian')
                        temp_img_2 = np.asarray(temp_img_2 * 255, dtype = np.uint8)
                        
                        temp_img_2_name = 'AddGaussian_' + name
                        saving_path_3 = output_path + directory + '/' + temp_img_2_name
                        print(saving_path_3) 
                        success = cv2.imwrite(saving_path_3, temp_img_2) 
                        if not success: 
                            raise Exception("img " + temp_img_2_name + " saving failed. ") 
                        
                        # Add Salt & Pepper noise 
                        temp_img_3 = skimage.util.random_noise(warped_img, mode = 's&p')
                        temp_img_3 = np.asarray(temp_img_3 * 255, dtype = np.uint8)
                        
                        temp_img_3_name = 'AddSP_' + name
                        saving_path_4 = output_path + directory + '/' + temp_img_3_name
                        print(saving_path_4)
                        success = cv2.imwrite(saving_path_4, temp_img_3) 
                        if not success: 
                            raise Exception("img " + temp_img_3_name + " saving failed. ") 
                    
            # (5) Drop the folder with num <= 30

data_clean_and_augmentation(input_path, output_path)

'''
if __name__ == '__main__':
	input_path = "./source/" # Source image folder path
	output_path = './result/' # Destination image folder path
	data_preprocessing_adjust_folders(input_path, output_path)
'''
