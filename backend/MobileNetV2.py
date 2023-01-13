# -*- coding: utf-8 -*-
"""
Created on Tue Oct 15 09:19:03 2019

@author: TMaysGGS
"""

'''Importing libraries & configurations''' 
import tensorflow.keras.backend as K 
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, ReLU, Add, GlobalAveragePooling2D, Reshape, Dropout, Softmax, Flatten 
from tensorflow.keras.models import Model 

NUM_CLASSES = 1000 

'''Building block funcitons''' 
def make_divisible(v, divisor, min_value = None): 
    
    if min_value == None: 
        min_value = divisor 
    
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor) 
    
    if new_v < 0.9 * v: 
        new_v = new_v + divisor 
    
    return new_v 

def conv_block(inputs, filters, kernel_size, strides):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    
    Z = Conv2D(filters, kernel_size, padding = "same", strides = strides)(inputs)
    Z = BatchNormalization(axis = channel_axis)(Z)
    A = ReLU(max_value = 6.0)(Z)
    
    return A

def bottleneck(inputs, filters, kernel_size, t, alpha, s, r = False):
    
    channel_axis = 1 if K.image_data_format() == 'channels_first' else -1
    # Depth 
    tchannel = K.int_shape(inputs)[channel_axis] * t
    # Width 
    cchannel = int(filters * alpha) 
    
    Z1 = conv_block(inputs, tchannel, 1, 1)
    
    Z1 = DepthwiseConv2D(kernel_size, strides = s, padding = "same", depth_multiplier = 1)(Z1)
    Z1 = BatchNormalization(axis = channel_axis)(Z1)
    A1 = ReLU(max_value = 6.0)(Z1)
    
    Z2 = Conv2D(cchannel, 1, strides = 1, padding = "same")(A1)
    Z2 = BatchNormalization(axis = channel_axis)(Z2)
    
    if r:
        Z2 = Add()([Z2, inputs])
    
    return Z2

def inverted_residual_block(inputs, filters, kernel_size, t, alpha, strides, n):
    
    Z = bottleneck(inputs, filters, kernel_size, t, alpha, strides)
    
    for i in range(1, n):
        Z = bottleneck(Z, filters, kernel_size, t, alpha, 1, True)
    
    return Z

'''Building the model''' 
def MobileNetV2(num_classes, alpha = 1.0): 
    
    num_first_filters = make_divisible(32, 8) 
    
    X = Input(shape = (224, 224, 3), name = 'input') 
    
    M = conv_block(X, num_first_filters, 3, 2) 
    
    M = inverted_residual_block(M, 16, 3, 1, alpha, 1, 1) 
    
    M = inverted_residual_block(M, 24, 3, 6, alpha, 2, 2) 
    
    M = inverted_residual_block(M, 32, 3, 6, alpha, 2, 3) 
    
    M = inverted_residual_block(M, 64, 3, 6, alpha, 2, 4) 
    
    M = inverted_residual_block(M, 96, 3, 6, alpha, 1, 3) 
    
    M = inverted_residual_block(M, 160, 3, 6, alpha, 2, 3) 
    
    M = inverted_residual_block(M, 320, 3, 6, alpha, 1, 1) 
    
    if alpha > 1.0: 
        num_last_filters = make_divisible(1280 * alpha, 8) 
    else: 
        num_last_filters = 1280 
    
    M = conv_block(M, num_last_filters, 1, 1) 
    
    M = GlobalAveragePooling2D()(M) 
    M = Reshape((1, 1, num_last_filters))(M) 
    M = Dropout(0.3)(M) 
    
    M = Conv2D(num_classes, 1, padding = 'same')(M) 
    A = Flatten()(M) 
    Y = Softmax()(A) 
    
    model = Model(X, Y) 
    
    return model 

model = MobileNetV2(num_classes = NUM_CLASSES)
model.summary() 
