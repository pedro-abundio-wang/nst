"""Perceptual Losses for Real-Time Style Transfer and Super-Resolution.

Related papers
- https://arxiv.org/abs/1603.08155

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import applications


class ReflectionPadding2D(layers.Layer):
    """
      2D Reflection Padding
      Attributes:
        - padding: (padding_width, padding_height) tuple
    """
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def compute_output_shape(self, input_shape):
        return (input_shape[0],
                input_shape[1] + 2 * self.padding[0],
                input_shape[2] + 2 * self.padding[1],
                input_shape[3])

    def call(self, inputs, *args, **kwargs):
        padding_width, padding_height = self.padding
        return tf.pad(inputs,
                      [[0, 0], [padding_height, padding_height], [padding_width, padding_width], [0, 0]],
                      'REFLECT')


def residual_block(input_tensor,
                   filters):
    """
    Our residual blocks each contain two 3x3 convolutional layers with the same
    number of filters on both layer. We use the residual block design of Gross and
    Wilber [2] (shown in Figure 1), which differs from that of He et al [3] in that the
    ReLU nonlinearity following the addition is removed; this modified design was
    found in [2] to perform slightly better for image classification.
    """

    if backend.image_data_format() == 'channels_last':
        bn_axis = -1
    else:
        bn_axis = 1

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal')(input_tensor)
    x = layers.BatchNormalization(
        axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=bn_axis)(x)

    x = layers.add([x, input_tensor[:, 2:-2, 2:-2, :]])

    return x


def transformation_network():
    """
    Our image transformation networks roughly follow the architectural guidelines
    set forth by Radford et al [42]. We do not use any pooling layers, instead using
    strided and fractionally strided convolutions for in-network downsampling and
    upsampling. Our network body consists of five residual blocks [43] using the ar-
    chitecture of [44]. All non-residual convolutional layers are followed by spatial
    batch normalization [45] and ReLU nonlinearities with the exception of the out-
    put layer, which instead uses a scaled tanh to ensure that the output image has
    pixels in the range [0; 255]. Other than the first and last layers which use 9x9
    kernels, all convolutional layers use 3x3 kernels. The exact architectures of all
    our networks can be found in the supplementary material.

    Architecture:
    Layer                          Activation size
    ----------------------------------------------
    Input                          3 x 256 x 256
    Reflection Padding (40 x 40)   3 x 336 x 336
    32 x 9 x 9 conv, stride 1      32 x 336 x 336
    64 x 3 x 3 conv, stride 2      64 x 168 x 168
    128 x 3 x 3 conv, stride 2     128 x 84 x 84
    Residual block, 128 filters    128 x 80 x 80
    Residual block, 128 filters    128 x 76 x 76
    Residual block, 128 filters    128 x 72 x 72
    Residual block, 128 filters    128 x 68 x 68
    Residual block, 128 filters    128 x 64 x 64
    64 x 3 x 3 conv, stride 1/2    64 x 128 x 128
    32 x 3 x 3 conv, stride 1/2    32 x 256 x 256
    3 x 9 x 9 conv, stride 1       3 x 256 x 256
    """

    input_shape = (256, 256, 3)
    img_input = layers.Input(shape=input_shape)
    x = img_input

    if backend.image_data_format() == 'channels_first':
        x = layers.Permute((3, 1, 2))(x)
        channel_axis = 1
    else:  # channels_last
        channel_axis = -1

    x = ReflectionPadding2D(padding=(40, 40))(x)
    x = layers.Conv2D(
        filters=32,
        kernel_size=9,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)
    x = residual_block(x, filters=128)

    x = layers.Conv2DTranspose(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=3,
        kernel_size=9,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = layers.BatchNormalization(
        axis=channel_axis)(x)
    x = tf.nn.tanh(x) * 150 + 255. / 2

    # Create model.
    model = models.Model(img_input, x, name='image_transformation_network')

    model.summary()

    return model


def loss_network():
    # load pre-trained vgg model
    model = applications.vgg16.VGG16(
        include_top=False,
        weights="imagenet")

    model.trainable = False

    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # model (as a dict).
    feature_extractor = models.Model(inputs=model.inputs, outputs=outputs_dict)

    model.summary()

    return feature_extractor
