import os
import warnings
import sys

import tensorflow as tf
import numpy as np
import tensorflow.keras as keras

sys.path.append('.')
from .__init__ import get_submodules_from_kwargs
from .arc_face import ArcFaceLossLayer
from .imagenet_utils import preprocess_input

backend = None
layers = None
models = None
keras_utils = None

def conv_block(inputs, filters, kernel_size, strides, padding):

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    Z = layers.Conv2D(filters, kernel_size, strides = strides, padding = padding)(inputs)
    Z = layers.BatchNormalization(axis = channel_axis)(Z)

    return layers.LeakyReLU(name='conv_pw_%d_relu' % filters)(Z)

def _conv_block(inputs, filters, alpha, kernel=(3, 3), strides=(1, 1)):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    filters = int(filters * alpha)
    x = layers.ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv1_pad')(inputs)
    x = layers.Conv2D(filters, kernel,
                      padding='valid',
                      use_bias=False,
                      strides=strides,
                      name='conv1')(x)
    x = layers.BatchNormalization(axis=channel_axis, name='conv1_bn')(x)
    return layers.LeakyReLU(name='conv1_relu')(x)


def _depthwise_conv_block(inputs, pointwise_conv_filters, alpha,
                          depth_multiplier=1, strides=(1, 1), block_id=1):
    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1
    pointwise_conv_filters = int(pointwise_conv_filters * alpha)

    if strides == (1, 1):
        x = inputs
    else:
        x = layers.ZeroPadding2D(((1, 1), (1, 1)),
                                 name='conv_pad_%d' % block_id)(inputs)
    x = layers.DepthwiseConv2D((3, 3),
                               padding='same' if strides == (1, 1) else 'valid',
                               depth_multiplier=depth_multiplier,
                               strides=strides,
                               use_bias=False,
                               name='conv_dw_%d' % block_id)(x)
    x = layers.BatchNormalization(
        axis=channel_axis, name='conv_dw_%d_bn' % block_id)(x)
    x = layers.LeakyReLU(name='conv_dw_%d_relu' % block_id)(x)

    x = layers.Conv2D(pointwise_conv_filters, (1, 1),
                      padding='same',
                      use_bias=False,
                      strides=(1, 1),
                      name='conv_pw_%d' % block_id)(x)
    x = layers.BatchNormalization(axis=channel_axis,
                                  name='conv_pw_%d_bn' % block_id)(x)
    return layers.LeakyReLU(name='conv_pw_%d_relu' % block_id)(x)


def linear_GD_conv_block(inputs, kernel_size, strides):

    channel_axis = 1 if backend.image_data_format() == 'channels_first' else -1

    Z = layers.DepthwiseConv2D(kernel_size, strides = strides, padding = 'valid', depth_multiplier = 1)(inputs)
    Z = layers.BatchNormalization(axis = channel_axis)(Z)

    return Z

def preprocess(x, **kwargs):
    """Preprocesses a numpy array encoding a batch of images.

    # Arguments
        x: a 4D numpy array consists of RGB values within [0, 255].

    # Returns
        Preprocessed array.
    """
    return preprocess_input(x, mode='tf', **kwargs)

def mobile_face_net_train(num_labels,
              loss = 'arcface',
              input_shape=(112,112,3),
              alpha=1.0,
              depth_multiplier=1,
              dropout=0.1,
              weights=None,
              variant=None,
              **kwargs):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224


    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if backend.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        backend.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    img_input = layers.Input(shape=input_shape)
    label = layers.Input(num_labels)
    x = _conv_block(img_input, 64, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=0)

    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier,
                              strides=(2, 2), block_id=1)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=2)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=4)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=11)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=12)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=13)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=14)

    if variant == 'Full':
        x = conv_block(x, 512, 1, 1, 'valid')

    x = linear_GD_conv_block(x, min([x.shape[1:3]]), 1) # (1, 1, 512)

    if variant == 'MobileFaceNet-M' or variant == 'Full':
        x = conv_block(x, 128, 1, 1, 'valid')

    x = layers.Dropout(rate = dropout)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)

    if loss == 'arcface':
        Y = ArcFaceLossLayer(class_num = num_labels)([x, label])
        model = models.Model(inputs = [img_input, label], outputs = Y, name = 'mobile_face_net')
    else:
        Y = layers.Dense(units = num_labels, activation = 'softmax')(x)
        model = models.Model(inputs = img_input, outputs = Y, name = 'mobile_face_net')

    initial_weights = [layer.get_weights() for layer in model.layers]
    if weights is not None:
        model.load_weights(weights, by_name=True)

        for layer, initial in zip(model.layers, initial_weights):
            weights = layer.get_weights()
            if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                print(f'Checkpoint contained no weights for layer {layer.name}!')

    return model

def mobile_face_net(loss = 'arcface',
              input_shape=(112,112,3),
              alpha=1.0,
              depth_multiplier=1,
              dropout=0.1,
              weights=None,
              variant=None,
              **kwargs):

    global backend, layers, models, keras_utils
    backend, layers, models, keras_utils = get_submodules_from_kwargs(kwargs)

    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')

    # Determine proper input shape and default size.
    if input_shape is None:
        default_size = 224
    else:
        if backend.image_data_format() == 'channels_first':
            rows = input_shape[1]
            cols = input_shape[2]
        else:
            rows = input_shape[0]
            cols = input_shape[1]

        if rows == cols and rows in [128, 160, 192, 224]:
            default_size = rows
        else:
            default_size = 224

    if backend.image_data_format() == 'channels_last':
        row_axis, col_axis = (0, 1)
    else:
        row_axis, col_axis = (1, 2)
    rows = input_shape[row_axis]
    cols = input_shape[col_axis]

    if backend.image_data_format() != 'channels_last':
        warnings.warn('The MobileNet family of models is only available '
                      'for the input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height).'
                      ' You should set `image_data_format="channels_last"` '
                      'in your Keras config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        backend.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    img_input = layers.Input(shape=input_shape)
    x = _conv_block(img_input, 64, alpha, strides=(2, 2))
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=0)

    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier,
                              strides=(2, 2), block_id=1)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=2)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=3)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=4)
    x = _depthwise_conv_block(x, 64, alpha, depth_multiplier, block_id=5)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=6)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=7)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=8)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=9)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=10)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=11)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=12)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier,
                              strides=(2, 2), block_id=13)

    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=14)
    x = _depthwise_conv_block(x, 128, alpha, depth_multiplier, block_id=15)

    if variant == 'Full':
        x = conv_block(x, 512, 1, 1, 'valid')

    x = linear_GD_conv_block(x, min([x.shape[1:3]]), 1) # (1, 1, 512)

    if variant == 'MobileFaceNet-M' or variant == 'Full':
        x = conv_block(x, 128, 1, 1, 'valid')

    x = layers.Flatten()(x)
    x = layers.Dense(128)(x)
    Y = layers.BatchNormalization()(x)

    model = models.Model(inputs = img_input, outputs = Y, name = 'mobile_face_net')

    initial_weights = [layer.get_weights() for layer in model.layers]
    if weights is not None:
        model.load_weights(weights, by_name=True)
        for layer, initial in zip(model.layers, initial_weights):
            weights = layer.get_weights()
            if weights and all(tf.nest.map_structure(np.array_equal, weights, initial)):
                print(f'Checkpoint contained no weights for layer {layer.name}!')

    return model

