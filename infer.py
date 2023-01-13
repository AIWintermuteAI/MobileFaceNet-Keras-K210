import os
import sys
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, CSVLogger, ReduceLROnPlateau
from tensorflow.keras.optimizers import SGD, Adam, Adagrad, Adamax
from tensorflow.keras.models import Model
import cv2 
import skimage 
import skimage.transform
import random 
import numpy as np 
from mtcnn.mtcnn import MTCNN 

sys.path.append('./backend')
from backend.MobileFaceNet import mobile_face_net_train, mobile_face_net, preprocess_input

IMG_SHAPE = (128, 128) # in HW form 
detector = MTCNN()
offset_x = 8
offset_y = -8
offset_mouth = -10
src = np.array([(44+offset_x, 59+offset_y), 
                (84+offset_x, 59+offset_y), 
                (64+offset_x, 82+offset_y), 
                (47+offset_x, 105+offset_y+offset_mouth),
                (81+offset_x, 105+offset_y+offset_mouth)], dtype = np.float32)
        
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
    cropped_imgs = []
    info = face_detection(img, detector) 
    if len(info) <= 0: 
        return None 
    else: 
        for face_info in info: 
            assert(len(face_info) == 3) 
            keypoints = face_info[2] 
            transformer = skimage.transform.SimilarityTransform() 
            transformer.estimate(keypoints, src) 
            M = transformer.params[0: 2, : ] 
            warped_img = cv2.warpAffine(img, M, (IMG_SHAPE[1], IMG_SHAPE[0]), borderValue = 0.0) 
            cropped_imgs.append(warped_img)
        return cropped_imgs 

if __name__ == '__main__':
    data_dir = sys.argv[1] # Source image folder path
    model = tf.keras.models.load_model(filepath=sys.argv[2])
    prev_emb = np.zeros((128), dtype=np.float32)
    print('Getting the image info')
    img_file_list = os.listdir(data_dir)
    for image_file in img_file_list:
        img = cv2.imread(os.path.join(data_dir,image_file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) 
        faces = face_alignment(img, detector)
        for face in faces:
            cv2.imshow('image',face)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
            face = tf.cast(face, tf.float32)
            face = preprocess_input(face, backend=keras.backend, layers=keras.layers, models=keras.models, utils=keras.utils)
            emb = model.predict(np.expand_dims(face, 0))
            emb = np.squeeze(emb)
            diff = np.subtract(emb, prev_emb)
            dist = np.sum(np.square(diff))
            prev_emb = emb
            print(dist)
