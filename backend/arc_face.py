# -*- coding: utf-8 -*-
"""
Created on Tue Nov 12 13:18:31 2019

@author: TMaysGGS
"""

'''Last updated on 2020.03.26 09:18'''
import math
import tensorflow as tf
import tensorflow.python.keras.backend as K
from tensorflow.python.keras.layers import Layer

# Arc Face Loss Layer (Class)
class ArcFaceLossLayer(Layer):
    '''
    Arguments:
        inputs: the input embedding vectors
        class_num: number of classes
        s: scaler value (default as 64)
        m: the margin value (default as 0.5)
    Returns:
        the final calculated outputs
    '''
    def __init__(self, class_num, s = 16., m = 0.5):
        
        super(ArcFaceLossLayer, self).__init__()
        self.class_num = class_num
        #self.s = s
        self.s = math.sqrt(2)*(math.log(class_num-1))
        print("Scaling of ArcFace:", self.s)
        self.m = m
    
    def get_config(self):
        
        config = super().get_config().copy()
        config.update({
                'class_num': self.class_num, 
                's': self.s, 
                'm': self.m
                })
        
        return config
    
    def build(self, input_shape):
        
        assert len(input_shape[0]) == 2 and len(input_shape[1]) == 2
        self.W = self.add_weight(name = '{}_W'.format(self.name), shape = (input_shape[0][-1], self.class_num), 
                                 initializer = 'glorot_uniform', trainable = True) # Xavier uniform intializer
        
    def call(self, inputs, mask = None):
        
        cos_m = math.cos(self.m)
        sin_m = math.sin(self.m)
        mm = sin_m * self.m
        threshold = math.cos(math.pi - self.m)
        # features
        X = inputs[0] 
        # 1-D or one-hot label works as mask
        Y_mask = inputs[1]

        # If Y_mask is not in one-hot form, transfer it to one-hot form.
        if Y_mask.shape[-1] == 1: 
            Y_mask = K.cast(Y_mask, tf.int32)
            Y_mask = K.reshape(K.one_hot(Y_mask, self.class_num), (-1, self.class_num))

        X_normed = K.l2_normalize(X, axis = 1) # L2 Normalized X
        W_normed = K.l2_normalize(self.W, axis = 0) # L2 Normalized Weights
        
        # cos(theta + m)
        cos_theta = K.dot(X_normed, W_normed) # 矩阵乘法 
        cos_theta2 = K.square(cos_theta)
        sin_theta2 = 1. - cos_theta2
        sin_theta = K.sqrt(sin_theta2 + K.epsilon())
        cos_tm = self.s * ((cos_theta * cos_m) - (sin_theta * sin_m))
        
        # This condition controls the theta + m should in range [0, pi]
        #   0 <= theta + m < = pi
        #   -m <= theta <= pi - m
        cond_v = cos_theta - threshold
        cond = K.cast(K.relu(cond_v), dtype = tf.bool)
        keep_val = self.s * (cos_theta - mm)
        cos_tm_temp = tf.where(cond, cos_tm, keep_val)
        
        # mask by label
        # Y_mask =+ K.epsilon() # Why???
        inv_mask = 1. - Y_mask
        s_cos_theta = self.s * cos_theta
        output = K.softmax((s_cos_theta * inv_mask) + (cos_tm_temp * Y_mask))
        
        return output
    
    def compute_output_shape(self, input_shape):
        print(input_shape[0], self.class_num)
        return input_shape[0], self.class_num

