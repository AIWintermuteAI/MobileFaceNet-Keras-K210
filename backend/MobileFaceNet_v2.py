# -*- coding: utf-8 -*-
"""
Created on Mon Dec  9 19:55:57 2019

@author: TMaysGGS
"""

'''Last updated on 2020.03.26 09:18'''
'''Importing the libraries''' 
import sys 
from tensorflow.python.keras import backend as K 
from tensorflow.python.keras.layers import Input, Conv2D, BatchNormalization, LeakyReLU, SeparableConv2D, DepthwiseConv2D, add, Flatten, Dense, Dropout 
from tensorflow.python.keras.models import Model 

sys.path.append('../') 
from tools.Keras_custom_layers import ArcFaceLossLayer 

'''Building Block Functions'''
def conv_block(inputs, filters, kernel_size, strides, padding):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = Conv2D(filters, kernel_size, strides = strides, padding = padding, use_bias = False)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    A = LeakyReLU()(Z)
    
    return A

def separable_conv_block(inputs, filters, kernel_size, strides):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = SeparableConv2D(filters, kernel_size, strides = strides, padding = "same", use_bias = False)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    A = LeakyReLU()(Z)
    
    return A

def bottleneck(inputs, filters, kernel, t, s, r = False):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    tchannel = K.int_shape(inputs)[channel_axis] * t
    
    Z1 = conv_block(inputs, tchannel, 1, 1, 'same')
    
    Z1 = DepthwiseConv2D(kernel, strides = s, padding = 'same', depth_multiplier = 1, use_bias = False)(Z1)
    Z1 = BatchNormalization(axis = channel_axis)(Z1)
    A1 = LeakyReLU()(Z1)
    
    Z2 = Conv2D(filters, 1, strides = 1, padding = 'same', use_bias = False)(A1)
    Z2 = BatchNormalization(axis = channel_axis)(Z2)
    
    if r:
        Z2 = add([Z2, inputs])
    
    return Z2

def inverted_residual_block(inputs, filters, kernel, t, strides, n):
    
    Z = bottleneck(inputs, filters, kernel, t, strides)
    
    for i in range(1, n):
        Z = bottleneck(Z, filters, kernel, t, 1, True)
    
    return Z

def linear_GD_conv_block(inputs, kernel_size, strides):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = DepthwiseConv2D(kernel_size, strides = strides, padding = 'valid', depth_multiplier = 1, use_bias = False)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    
    return Z

'''Building the MobileFaceNet Model'''
def mobile_face_net_train(num_labels, loss = 'arcface'):
    
    X = Input(shape = (112, 112, 3))
    label = Input((num_labels, ))

    M = conv_block(X, 64, 3, 2, 'same') # Output Shape: (56, 56, 64) 

    M = separable_conv_block(M, 64, 3, 1) # (56, 56, 64) 
    
    M = inverted_residual_block(M, 64, 3, t = 2, strides = 2, n = 5) # (28, 28, 64) 
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1) # (14, 14, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 6) # (14, 14, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1) # (7, 7, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 2) # (7, 7, 128) 
    
    M = conv_block(M, 512, 1, 1, 'valid') # (7, 7, 512) 
    
    M = linear_GD_conv_block(M, 7, 1) # (1, 1, 512) 
    # kernel_size = 7 for 112 x 112; 4 for 64 x 64
    
    M = conv_block(M, 128, 1, 1, 'valid')
    M = Dropout(rate = 0.2)(M)
    M = Flatten()(M)
    
    M = Dense(128, activation = None, use_bias = False, kernel_initializer = 'glorot_normal')(M) 
    
    if loss == 'arcface': 
        Y = ArcFaceLossLayer(class_num = num_labels)([M, label]) 
        model = Model(inputs = [X, label], outputs = Y, name = 'mobile_face_net') 
    else: 
        Y = Dense(units = num_labels, activation = 'softmax')(M) 
        model = Model(inputs = X, outputs = Y, name = 'mobile_face_net') 
    
    return model 

def mobile_face_net():
    
    X = Input(shape = (112, 112, 3)) 

    M = conv_block(X, 64, 3, 2, 'same') # Output Shape: (56, 56, 64) 

    M = separable_conv_block(M, 64, 3, 1) # (56, 56, 64) 
    
    M = inverted_residual_block(M, 64, 3, t = 2, strides = 2, n = 5) # (28, 28, 64) 
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1) # (14, 14, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 6) # (14, 14, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 4, strides = 2, n = 1) # (7, 7, 128) 
    
    M = inverted_residual_block(M, 128, 3, t = 2, strides = 1, n = 2) # (7, 7, 128) 
    
    M = conv_block(M, 512, 1, 1, 'valid') # (7, 7, 512) 
    
    M = linear_GD_conv_block(M, 7, 1) # (1, 1, 512) 
    # kernel_size = 7 for 112 x 112; 4 for 64 x 64
    
    M = conv_block(M, 128, 1, 1, 'valid')
    # M = Dropout(rate = 0.1)(M)
    M = Flatten()(M)
    
    M = Dense(128, activation = None, use_bias = False, kernel_initializer = 'glorot_normal')(M) 
    
    model = Model(inputs = X, outputs = M, name = 'mobile_face_net')
    
    return model 
