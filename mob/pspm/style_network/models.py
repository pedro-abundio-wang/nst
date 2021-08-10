"""Perceptual Losses for Real-Time Style Transfer and Super-Resolution.

Related papers
- https://arxiv.org/abs/1603.08155

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
import tensorflow as tf
import tensorflow_addons as tfa

from tensorflow.keras import backend
from tensorflow.keras import models
from tensorflow.keras import layers
from tensorflow.keras import applications


def vgg_layers(layer_names):
    vgg = applications.vgg19.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = [vgg.get_layer(name).output for name in layer_names]
    model = tf.keras.Model([vgg.input], outputs)
    return model


def gram_matrix(features, normalize=True):
    batch_size, height, width, filters = features.shape
    features = tf.reshape(features, (batch_size, height * width, filters))

    features_transpose = tf.transpose(features, perm=[0, 2, 1])
    gram = tf.matmul(features_transpose, features)
    if normalize:
        gram /= tf.cast(height * width, tf.float32)

    return gram


def style_loss(style_outputs, style_target):
    loss = tf.add_n([tf.reduce_mean((style_outputs[name] - style_target[name]) ** 2)
                     for name in style_outputs.keys()])

    return loss / len(style_outputs)


def content_loss(content_outputs, content_target):
    loss = tf.add_n([tf.reduce_mean((content_outputs[name] - content_target[name]) ** 2)
                     for name in content_outputs.keys()])

    return loss / len(content_outputs)


def total_variation_loss(img):
    x_var = img[:, :, 1:, :] - img[:, :, :-1, :]
    y_var = img[:, 1:, :, :] - img[:, :-1, :, :]

    return tf.reduce_mean(tf.square(x_var)) + tf.reduce_mean(tf.square(y_var))


class StyleContentModel(tf.keras.models.Model, abc.ABC):
    def __init__(self, style_layers, content_layers):
        super(StyleContentModel, self).__init__()
        self.vgg = vgg_layers(style_layers + content_layers)
        self.content_layers = content_layers
        self.style_layers = style_layers
        self.num_content_layers = len(content_layers)
        self.num_style_layers = len(style_layers)
        self.vgg.trainable = False

    def call(self, inputs, training=None, mask=None):
        preprocessed_input = applications.vgg19.preprocess_input(inputs)
        outputs = self.vgg(preprocessed_input)
        style_outputs, content_outputs = (outputs[:self.num_style_layers],
                                          outputs[self.num_style_layers:])

        # Compute the gram_matrix
        style_outputs = [gram_matrix(style_output) for style_output in style_outputs]

        # Features that extracted by VGG
        style_dict = {style_name: value
                      for style_name, value in zip(self.style_layers, style_outputs)}

        content_dict = {content_name: value
                        for content_name, value in zip(self.content_layers, content_outputs)}

        return {'content': content_dict, 'style': style_dict}


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
    x = tfa.layers.InstanceNormalization(
        axis=bn_axis)(x)
    # x = layers.BatchNormalization(
    #     axis=bn_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=1,
        padding='valid',
        kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(
        axis=bn_axis)(x)
    # x = layers.BatchNormalization(
    #     axis=bn_axis)(x)

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
    x = tfa.layers.InstanceNormalization(
        axis=channel_axis)(x)
    # x = layers.BatchNormalization(
    #     axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=64,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(
        axis=channel_axis)(x)
    # x = layers.BatchNormalization(
    #     axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=128,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(
        axis=channel_axis)(x)
    # x = layers.BatchNormalization(
    #     axis=channel_axis)(x)
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
    x = tfa.layers.InstanceNormalization(
        axis=channel_axis)(x)
    # x = layers.BatchNormalization(
    #     axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2DTranspose(
        filters=32,
        kernel_size=3,
        strides=2,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(
        axis=channel_axis)(x)
    # x = layers.BatchNormalization(
    #     axis=channel_axis)(x)
    x = layers.Activation('relu')(x)

    x = layers.Conv2D(
        filters=3,
        kernel_size=9,
        strides=1,
        padding='same',
        kernel_initializer='he_normal')(x)
    x = tfa.layers.InstanceNormalization(
        axis=channel_axis)(x)
    # x = layers.BatchNormalization(
    #     axis=channel_axis)(x)
    x = tf.nn.tanh(x) * 150 + 255. / 2

    # Create model.
    model = models.Model(img_input, x, name='image_transformation_network')

    model.summary()

    return model


class instance_norm(tf.keras.layers.Layer):
    def __init__(self, epsilon=1e-3):
        super(instance_norm, self).__init__()
        self.epsilon = epsilon

    def build(self, input_shape):
        self.beta = tf.Variable(tf.zeros([input_shape[3]]))
        self.gamma = tf.Variable(tf.ones([input_shape[3]]))

    def call(self, inputs):
        mean, var = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        x = tf.divide(tf.subtract(inputs, mean), tf.sqrt(tf.add(var, self.epsilon)))

        return self.gamma * x + self.beta


class conv_2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(conv_2d, self).__init__()
        pad = kernel // 2
        self.paddings = tf.constant([[0, 0], [pad, pad], [pad, pad], [0, 0]])
        self.conv2d = tf.keras.layers.Conv2D(filters, kernel, stride, use_bias=False, padding='valid')
        self.instance_norm = instance_norm()

    def call(self, inputs, relu=True):
        x = tf.pad(inputs, self.paddings, mode='REFLECT')
        x = self.conv2d(x)
        x = self.instance_norm(x)

        if relu:
            x = tf.nn.relu(x)
        return x


class resize_conv_2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(resize_conv_2d, self).__init__()
        self.conv = conv_2d(filters, kernel, stride)
        self.instance_norm = instance_norm()
        self.stride = stride

    def call(self, inputs):
        new_h = inputs.shape[1] * self.stride * 2
        new_w = inputs.shape[2] * self.stride * 2
        x = tf.image.resize(inputs, [new_h, new_w], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = self.conv(x)
        # return x

        """ Redundant """
        x = self.instance_norm(x)

        return tf.nn.relu(x)


class tran_conv_2d(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(tran_conv_2d, self).__init__()
        self.tran_conv = tf.keras.layers.Conv2DTranspose(filters, kernel, stride, padding='same')
        self.instance_norm = instance_norm()

    def call(self, inputs):
        x = self.tran_conv(inputs)
        x = self.instance_norm(x)

        return tf.nn.relu(x)


class residual(tf.keras.layers.Layer):
    def __init__(self, filters, kernel, stride):
        super(residual, self).__init__()
        self.conv1 = conv_2d(filters, kernel, stride)
        self.conv2 = conv_2d(filters, kernel, stride)

    def call(self, inputs):
        x = self.conv1(inputs)
        return inputs + self.conv2(x, relu=False)


class feed_forward(tf.keras.models.Model):
    def __init__(self):
        super(feed_forward, self).__init__()
        # [filters, kernel, stride]
        self.conv1 = conv_2d(32, 9, 1)
        self.conv2 = conv_2d(64, 3, 2)
        self.conv3 = conv_2d(128, 3, 2)
        self.resid1 = residual(128, 3, 1)
        self.resid2 = residual(128, 3, 1)
        self.resid3 = residual(128, 3, 1)
        self.resid4 = residual(128, 3, 1)
        self.resid5 = residual(128, 3, 1)
        # self.tran_conv1 = tran_conv_2d(64, 3, 2)
        # self.tran_conv2 = tran_conv_2d(32, 3, 2)
        self.resize_conv1 = resize_conv_2d(64, 3, 2)
        self.resize_conv2 = resize_conv_2d(32, 3, 2)
        self.conv4 = conv_2d(3, 9, 1)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.resid1(x)
        x = self.resid2(x)
        x = self.resid3(x)
        x = self.resid4(x)
        x = self.resid5(x)
        x = self.resize_conv1(x)
        x = self.resize_conv2(x)
        x = self.conv4(x, relu=False)
        return (tf.nn.tanh(x) * 150 + 255. / 2)  # for better convergence
