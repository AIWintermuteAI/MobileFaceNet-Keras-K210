# -*- coding: utf-8 -*-
"""
Created on Mon May 13 16:54:33 2019

@author: TMaysGGS
"""

'''Last updated on 2020.03.30 11:15''' 
'''Importing the libraries & setting the configurations'''
import os
import sys
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping, Callback, CSVLogger, ReduceLROnPlateau
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras.optimizer_v2.adam import Adam
from tensorflow.python.keras.utils.multi_gpu_utils import multi_gpu_model
# import keras.backend.tensorflow_backend as KTF
# from tensorflow.python.keras.utils.vis_utils import plot_model

sys.path.append('../')
from Model_Structures.MobileFaceNet import mobile_face_net_train

os.environ['CUDA_VISIBLE_DEVICES'] = '3' # 如需多张卡设置为：'1, 2, 3'，使用CPU设置为：''
'''Set if the GPU memory needs to be restricted
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
session = tf.Session(config = config)

KTF.set_session(session)
'''
BATCH_SIZE = 128
old_m = 15090270
m = 15090270
DATA_SPLIT = 0.005
OLD_NUM_LABELS = 67960
NUM_LABELS = 67960
TOTAL_EPOCHS = 1000
OLD_LOSS_TYPE = 'arcface'
LOSS_TYPE = 'arcface'

'''Importing the data set'''
train_path = '/data/daiwei/processed_data/datasets_for_face_recognition'

train_datagen = ImageDataGenerator(rescale = 1. / 255, validation_split = DATA_SPLIT)

def mobilefacenet_input_generator(generator, directory, subset, loss = 'arcface'):
    
    gen = generator.flow_from_directory(
            directory, 
            target_size = (112, 112), 
            color_mode = 'rgb', 
            batch_size = BATCH_SIZE, 
            class_mode = 'categorical', 
            subset = subset)
    
    while True: 
        
        X = gen.next() 
        if loss == 'arcface':
            yield [X[0], X[1]], X[1] 
        else: 
            yield X[0], X[1] 

train_generator = mobilefacenet_input_generator(train_datagen, train_path, 'training', LOSS_TYPE) 
validate_generator = mobilefacenet_input_generator(train_datagen, train_path, 'validation', LOSS_TYPE) 

'''Loading the model & re-defining''' 
model = mobile_face_net_train(OLD_NUM_LABELS, loss = OLD_LOSS_TYPE)
print("Reading the pre-trained model... ")
model.load_weights(r'../Models/MobileFaceNet_train.h5')
print("Reading done. ")
model.summary()
# plot_model(model, to_file = r'../Models/training_model.png', show_shapes = True, show_layer_names = True) 
# model.layers

# SoftMax-ArcFace, FC layer change/not chage
if OLD_LOSS_TYPE == 'softmax' and LOSS_TYPE == 'arcface':
    customed_model = mobile_face_net_train(NUM_LABELS, loss = LOSS_TYPE)
    temp_weights_list = []
    for layer in model.layers:
        temp_layer = model.get_layer(layer.name)
        temp_weights = temp_layer.get_weights()
        temp_weights_list.append(temp_weights)
    for i in range(len(customed_model.layers) - 2):
        customed_model.get_layer(customed_model.layers[i].name).set_weights(temp_weights_list[i])

# ArcFace-SoftMax, FC layer change/not chage
elif OLD_LOSS_TYPE == 'arcface' and LOSS_TYPE == 'softmax':
    customed_model = mobile_face_net_train(NUM_LABELS, loss = LOSS_TYPE)
    temp_weights_list = []
    for layer in model.layers:
        temp_layer = model.get_layer(layer.name)
        temp_weights = temp_layer.get_weights()
        temp_weights_list.append(temp_weights)
    for i in range(len(customed_model.layers) - 1):
        customed_model.get_layer(customed_model.layers[i].name).set_weights(temp_weights_list[i])

# SoftMax-SoftMax/ArcFace-ArcFace, FC layer change 
elif OLD_LOSS_TYPE == LOSS_TYPE and OLD_NUM_LABELS != NUM_LABELS:
    customed_model = mobile_face_net_train(NUM_LABELS, loss = LOSS_TYPE)
    temp_weights_list = []
    for layer in model.layers:
        temp_layer = model.get_layer(layer.name)
        temp_weights = temp_layer.get_weights()
        temp_weights_list.append(temp_weights)
    for i in range(len(customed_model.layers) - 1):
        customed_model.get_layer(customed_model.layers[i].name).set_weights(temp_weights_list[i])

# SoftMax-SoftMax/ArcFace-ArcFace, FC layer not change
else:
    customed_model = model
customed_model.summary()

# Use multi-gpus to train the model 
num_gpus = len(os.environ['CUDA_VISIBLE_DEVICES'].split(',')) 
if num_gpus > 1:
    parallel_model = multi_gpu_model(customed_model, gpus = num_gpus)

'''Setting configurations for training the Model'''
if num_gpus <= 1:
    customed_model.compile(optimizer = Adam(lr = 0.01, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics = ['accuracy'])
    # Temporarily increase the learing rate to 0.01
else:
    parallel_model.compile(optimizer = Adam(lr = 0.01, epsilon = 1e-8), loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Save the model after every epoch
class ParallelModelCheckpoint(ModelCheckpoint): 
    
    def __init__(self, model, filepath, monitor = 'val_loss', verbose = 0, save_best_only = True, save_weights_only = False, mode = 'auto', period = 1): 
        self.single_model = model
        super(ParallelModelCheckpoint, self).__init__(filepath, monitor, verbose, save_best_only, save_weights_only, mode, period)
    
    def set_model(self, model):
        super(ParallelModelCheckpoint, self).set_model(self.single_model)

if num_gpus == 1:
    check_pointer = ModelCheckpoint(filepath = '../Models/MobileFaceNet_train.h5', verbose = 1, save_best_only = True)
elif num_gpus > 1:
    check_pointer = ParallelModelCheckpoint(customed_model, filepath = '../Models/MobileFaceNet_train.h5', monitor = 'val_loss', verbose = 1, save_best_only = True)

# Interrupt the training when the validation loss is not decreasing
early_stopping = EarlyStopping(monitor = 'val_loss', patience = 1000)

# Record the loss history
class LossHistory(Callback):
    def on_train_begin(self, logs = {}):
        self.losses = []
        
    def on_batch_end(self, batch, logs = {}):
        self.losses.append(logs.get('loss'))

history = LossHistory()

# Stream each epoch results into a .csv file
csv_logger = CSVLogger('training.csv', separator = ',', append = True)
# append = True append if file exists (useful for continuing training)
# append = False overwrite existing file

# Reduce learning rate when a metric has stopped improving
reduce_lr = ReduceLROnPlateau(monitor = 'val_loss', factor = 0.2, patience = 20, min_lr = 0) 

'''Importing the data & training the model'''
# Model.fit_generator is deprecated and will be removed in a future version, 
# Please use Model.fit, which supports generators.
if num_gpus <= 1:
    hist = customed_model.fit(
            train_generator,
            steps_per_epoch = int(m * (1 - DATA_SPLIT) / BATCH_SIZE), 
            epochs = TOTAL_EPOCHS,
            callbacks = [check_pointer, early_stopping, history, csv_logger, reduce_lr], 
            validation_data = validate_generator, 
            validation_steps = int(m * DATA_SPLIT / BATCH_SIZE), 
            workers = 1, 
            use_multiprocessing = False, 
            initial_epoch = 3)
elif num_gpus > 1:
    hist = parallel_model.fit(
            train_generator,
            steps_per_epoch = int(m * (1 - DATA_SPLIT) / BATCH_SIZE), 
            epochs = TOTAL_EPOCHS,
            callbacks = [check_pointer, early_stopping, history, csv_logger, reduce_lr], 
            validation_data = validate_generator, 
            validation_steps = int(m * DATA_SPLIT / BATCH_SIZE), 
            workers = 1, 
            use_multiprocessing = False, 
            initial_epoch = 3)
# For TensorFlow 2, Multi-Processing here is not able to use. Use tf.data API instead. 