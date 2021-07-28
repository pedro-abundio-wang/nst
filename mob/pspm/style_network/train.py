"""Neural Style Transfer model for Keras.

Related papers
- https://arxiv.org/abs/1508.06576

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

import os
import numpy as np
from PIL import Image

import tensorflow as tf

from tensorflow.keras import optimizers
from tensorflow.keras import applications
from tensorflow.keras import preprocessing

from mob.pspm.style_network.style_network import loss_network
from mob.pspm.style_network.style_network import transformation_network
from mob.pspm.style_network import dataset_builder


def load_image(path_to_image):
    max_dim = 512
    image = Image.open(path_to_image)
    long = max(image.size)
    scale = max_dim / long
    image = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.ANTIALIAS)
    image = preprocessing.image.img_to_array(image)
    # We need to broadcast the image array such that it has a batch dimension
    image = np.expand_dims(image, axis=0)
    return image


def preprocess_image(images):
    images = applications.vgg19.preprocess_input(images)
    return images


def load_and_preprocess_image(path_to_image):
    # Util function to open, resize and format pictures into appropriate tensors
    images = load_image(path_to_image)
    images = preprocess_image(images)
    return tf.convert_to_tensor(images)


def deprocess_image(processed_img):
    # processed_img: numpy array
    image = processed_img.copy()
    if len(image.shape) == 4:
        image = np.squeeze(image, 0)
    assert len(image.shape) == 3, ("Input to deprocess image must be an image of "
                                   "dimension [1, height, width, channel] or [height, width, channel]")
    if len(image.shape) != 3:
        raise ValueError("Invalid input to deprocessing image")

    # Perform the inverse of the preprocessing step
    # Remove zero-center by mean pixel
    image[:, :, 0] += 103.939
    image[:, :, 1] += 116.779
    image[:, :, 2] += 123.68
    # 'BGR'->'RGB'
    image = image[:, :, ::-1]
    image = np.clip(image, 0, 255).astype("uint8")
    return image


def content_loss(content, combination):
    return 0.5 * tf.reduce_sum(tf.square(content - combination))


def gram_matrix(features, normalize=True):
    """
    Compute the Gram matrix from features.

    Inputs:
    - features: Tensor of shape (1, height, width, channel) giving features for
      a single image.
    - normalize: optional, whether to normalize the Gram matrix
        If True, divide the Gram matrix by the number of neurons (height * width)

    Returns:
    - gram: Tensor of shape (channel, channel) giving the (optionally normalized)
      Gram matrices for the input image.
    """

    _, height, width, channel = features.shape
    feature_maps = tf.reshape(features, (-1, channel))
    gram = tf.matmul(tf.transpose(feature_maps), feature_maps)
    if normalize:
        gram = tf.divide(gram, tf.cast(2.0 * height * width * channel, gram.dtype))

    return gram


def style_loss(style, combination):
    style_gram = gram_matrix(style, normalize=True)
    combination_gram = gram_matrix(combination, normalize=True)
    return tf.reduce_sum(tf.square(style_gram - combination_gram))


def total_variation_loss(image):
    """
    Compute total variation loss.

    Inputs:
    - img: Tensor of shape (1, height, width, 3) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.

    Returns:
    - loss: Tensor holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    image = tf.squeeze(image)
    height, width, channel = image.shape
    img_col_start = tf.slice(image, [0, 0, 0], [height, width - 1, channel])
    img_col_end = tf.slice(image, [0, 1, 0], [height, width - 1, channel])
    img_row_start = tf.slice(image, [0, 0, 0], [height - 1, width, channel])
    img_row_end = tf.slice(image, [1, 0, 0], [height - 1, width, channel])
    return tf.reduce_sum(tf.square(img_col_end - img_col_start)) + tf.reduce_sum(tf.square(img_row_end - img_row_start))


def compute_loss(transformation_model,
                 loss_model,
                 content_images,
                 style_image,
                 content_layer_name,
                 content_weight,
                 style_layer_names,
                 style_weight,
                 total_variation_weight):
    """
    Inputs:
    - combination_images: Tensor of shape (batch_size, height, width, channel).
    - content_images: Tensor of shape (batch_size, height, width, channel).
    - style_image: Tensor of shape (1, height, width, channel).
    """
    content_features = loss_model(content_images)
    style_features = loss_model(style_image)
    # random_images = tf.Variable(tf.random.uniform(content_images.shape))
    combination_images = transformation_model(content_images)
    combination_features = loss_model(combination_images)

    assert content_images.shape == combination_images.shape
    batch_size = content_images.shape[0]
    assert batch_size == 4

    # Initialize the loss
    loss = tf.zeros(shape=())
    c_loss = tf.zeros(shape=())
    s_loss = tf.zeros(shape=())
    v_loss = tf.zeros(shape=())

    # content loss
    if content_layer_name is not None:
        for i in range(batch_size):
            content_layer_features = content_features[content_layer_name][i:i+1]
            combination_layer_features = combination_features[content_layer_name][i:i+1]
            c_loss += content_weight * content_loss(content_layer_features, combination_layer_features)

        c_loss /= batch_size

    # style loss
    if style_layer_names is not None:
        for i in range(batch_size):
            weight_per_style_layer = 1.0 / float(len(style_layer_names))
            for _, style_layer_name in enumerate(style_layer_names):
                style_layer_features = style_features[style_layer_name]
                combination_layer_features = combination_features[style_layer_name][i:i+1]
                s_loss += style_weight * weight_per_style_layer * style_loss(style_layer_features, combination_layer_features)

        s_loss /= batch_size

    # total variation loss
    for i in range(batch_size):
        v_loss += total_variation_weight * total_variation_loss(combination_images[i:i+1])

    v_loss /= batch_size

    # total loss
    loss += c_loss + s_loss + v_loss

    return loss, c_loss, s_loss, v_loss


def compute_loss_and_grads(transformation_model,
                           loss_model,
                           content_images,
                           style_image,
                           content_layer_name,
                           content_weight,
                           style_layer_names,
                           style_weight,
                           total_variation_weight):
    """
    ## Add a tf.function decorator to loss & gradient computation
    To compile it, and thus make it fast.
    """
    with tf.GradientTape() as tape:
        loss, c_loss, s_loss, v_loss = compute_loss(transformation_model,
                                                    loss_model,
                                                    content_images,
                                                    style_image,
                                                    content_layer_name,
                                                    content_weight,
                                                    style_layer_names,
                                                    style_weight,
                                                    total_variation_weight)
    grads = tape.gradient(loss, transformation_model.trainable_variables)

    return loss, c_loss, s_loss, v_loss, grads


@tf.function
def train_step(optimizer,
               transformation_model,
               loss_model,
               content_images,
               style_image,
               content_layer_name,
               content_weight,
               style_layer_names,
               style_weight,
               total_variation_weight):
    # train step
    loss, c_loss, s_loss, v_loss, grads = compute_loss_and_grads(
        transformation_model,
        loss_model,
        content_images,
        style_image,
        content_layer_name,
        content_weight,
        style_layer_names,
        style_weight,
        total_variation_weight)
    optimizer.apply_gradients(zip(grads, transformation_model.trainable_variables))
    return loss, c_loss, s_loss, v_loss


def train(transformation_model,
          loss_model,
          coco_tfrecord_path,
          style_image_path,
          content_layer_name,
          content_weight,
          style_layer_names,
          style_weight,
          total_variation_weight):

    optimizer = optimizers.Adam(learning_rate=1)

    dataset = dataset_builder.build(coco_tfrecord_path, 'train')

    style_image = load_and_preprocess_image(style_image_path)

    iterations = 40000

    # metrics
    loss_metric = tf.keras.metrics.Mean()
    closs_metric = tf.keras.metrics.Mean()
    sloss_metric = tf.keras.metrics.Mean()
    vloss_metric = tf.keras.metrics.Mean()

    # tensorboard
    logdir = "./tb/"
    writer = tf.summary.create_file_writer(logdir)
    writer.set_as_default()

    for i in range(iterations):
        content_images, _ = next(iter(dataset))
        content_images = preprocess_image(content_images)
        loss, c_loss, s_loss, v_loss = train_step(optimizer,
                                                  transformation_model,
                                                  loss_model,
                                                  content_images,
                                                  style_image,
                                                  content_layer_name,
                                                  content_weight,
                                                  style_layer_names,
                                                  style_weight,
                                                  total_variation_weight)

        # metrics
        loss_metric(loss)
        closs_metric(c_loss)
        sloss_metric(s_loss)
        vloss_metric(v_loss)

        if i % 100 == 0:
            logging.info("Iteration %d: total_loss=%.4e, content_loss=%.4e, style_loss=%.4e, variation_loss=%.4e"
                         % (i, loss, c_loss, s_loss, v_loss))

            tf.summary.scalar('loss', loss, step=i)
            tf.summary.scalar('c_loss', c_loss, step=i)
            tf.summary.scalar('s_loss', s_loss, step=i)
            tf.summary.scalar('v_loss', v_loss, step=i)

            checkpoint_path = './ckpt/'
            if not os.path.exists(checkpoint_path):
                os.makedirs(checkpoint_path)
            checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=transformation_model)
            checkpoint.save(checkpoint_path)
            # checkpoint.restore(tf.train.latest_checkpoint(checkpoint_path))


def run():
    params = {
        'coco_tfrecord_path': '/home/pedro/datasets/coco',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': 'block2_conv2',
        'content_weight': 1e-3,
        'style_layer_names': [
            "block1_conv2",
            "block2_conv2",
            "block3_conv3",
            "block4_conv3"
        ],
        'style_weight': 1e0,
        'total_variation_weight': 1e-5
    }
    transformation_model = transformation_network()
    loss_model = loss_network()
    train(transformation_model, loss_model, **params)


def main(_):
    run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
