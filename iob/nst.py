"""Neural Style Transfer model for Keras.

Related papers
- https://arxiv.org/abs/1508.06576

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import logging

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import tensorflow as tf

from tensorflow.keras import models
from tensorflow.keras import optimizers
from tensorflow.keras import applications
from tensorflow.keras import preprocessing


def load_image(path_to_image):
    max_dim = 512
    image = Image.open(path_to_image)
    long = max(image.size)
    scale = max_dim/long
    image = image.resize((round(image.size[0]*scale), round(image.size[1]*scale)), Image.ANTIALIAS)
    image = preprocessing.image.img_to_array(image)
    # We need to broadcast the image array such that it has a batch dimension
    image = np.expand_dims(image, axis=0)
    return image


def imshow(image, title=None):
    # Remove the batch dimension
    out = np.squeeze(image, axis=0)
    # Normalize for display
    out = out.astype('uint8')
    plt.imshow(out)
    if title is not None:
        plt.title(title)
    plt.imshow(out)


def load_and_preprocess_image(path_to_image):
    # Util function to open, resize and format pictures into appropriate tensors
    image = load_image(path_to_image)
    image = applications.vgg19.preprocess_input(image)
    return tf.convert_to_tensor(image)


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


def load_model():
    """load pre-trained vgg model"""
    model = applications.vgg19.VGG19(include_top=False, weights="imagenet")
    model.trainable = False
    return model


def create_feature_extract(model):
    # Get the symbolic outputs of each "key" layer (we gave them unique names).
    outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

    # Set up a model that returns the activation values for every layer in
    # model (as a dict).
    feature_extractor = models.Model(inputs=model.inputs, outputs=outputs_dict)
    feature_extractor.summary()

    return feature_extractor


def vis_model_input_filters(model):
    # 1st conv filter weight
    input_filter_weight = [weight
                           for weight in model.weights
                           if 'kernel' in weight.name][0].numpy()
    vis_tensor_grid(input_filter_weight, 'input_filter_weight.png')


def vis_model_feature_maps(feature_extractor,
                           img_path='Green_Sea_Turtle_grazing_seagrass.jpg'):
    """vis model feature maps"""
    image = load_and_preprocess_image(img_path)
    feature_maps = feature_extractor.predict(image)

    for layer_name, feature_map in feature_maps.items():
        logging.info("visual feature map grid, layer_name = %s" % layer_name)
        vis_tensor_grid(feature_map, '%s_feature_map.png' % layer_name)


def vis_tensor_grid(tensor, save_path):
    assert len(tensor.shape) == 4, ("tensor.shape = (batch_size, height, width, channel_size) or "
                                    "tensor.shape = (in_channel, filter_size, filter_size, out_channel) "
                                    "batch_size = 1 for feature map vis"
                                    "in_channel = 3 for input filter weight vis")
    num_channel = tensor.shape[-1]
    grid_size = int(np.ceil(np.sqrt(num_channel)))
    w_min, w_max = np.min(tensor), np.max(tensor)
    for i in range(num_channel):
        plt.subplot(grid_size, grid_size, i + 1)
        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (tensor[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
    plt.savefig(save_path)
    plt.clf()


def content_loss(content, combination):
    return 0.5 * tf.reduce_sum(tf.square(combination - content))


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


def compute_loss(feature_extractor,
                 combination_image,
                 content_image,
                 style_image,
                 content_layer_name,
                 content_weight,
                 style_layer_names,
                 style_weight,
                 total_variation_weight):

    content_features = feature_extractor(content_image)
    style_features = feature_extractor(style_image)
    combination_features = feature_extractor(combination_image)

    # Initialize the loss
    loss = tf.zeros(shape=())
    c_loss = tf.zeros(shape=())
    s_loss = tf.zeros(shape=())
    v_loss = tf.zeros(shape=())

    # content loss
    if content_layer_name is not None:
        content_layer_features = content_features[content_layer_name]
        combination_layer_features = combination_features[content_layer_name]
        c_loss += content_loss(content_layer_features, combination_layer_features)
        c_loss *= content_weight

    # style loss
    if style_layer_names is not None:
        weight_per_style_layer = 1.0 / float(len(style_layer_names))
        for i, style_layer_name in enumerate(style_layer_names):
            style_layer_features = style_features[style_layer_name]
            combination_layer_features = combination_features[style_layer_name]
            s_loss += weight_per_style_layer * style_loss(style_layer_features, combination_layer_features)
        s_loss *= style_weight

    # total variation loss
    v_loss += total_variation_weight * total_variation_loss(combination_image)

    # total loss
    loss += c_loss + s_loss + v_loss

    return loss, c_loss, s_loss, v_loss


def compute_loss_and_grads(feature_extractor,
                           combination_image,
                           content_image,
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
        loss, c_loss, s_loss, v_loss = compute_loss(feature_extractor,
                                                    combination_image,
                                                    content_image,
                                                    style_image,
                                                    content_layer_name,
                                                    content_weight,
                                                    style_layer_names,
                                                    style_weight,
                                                    total_variation_weight)
    grads = tape.gradient(loss, combination_image)

    return loss, c_loss, s_loss, v_loss, grads


def neural_style_transfer(feature_extractor,
                          content_image_path,
                          style_image_path,
                          content_layer_name,
                          content_weight,
                          style_layer_names,
                          style_weight,
                          total_variation_weight,
                          result_prefix,
                          init_random=False):

    optimizer = optimizers.Adam(learning_rate=10)

    content_image = load_and_preprocess_image(content_image_path)
    style_image = load_and_preprocess_image(style_image_path)

    if init_random:
        combination_image = tf.Variable(tf.random.uniform(content_image.shape))
    else:
        combination_image = tf.Variable(load_and_preprocess_image(content_image_path), dtype=tf.float32)

    iterations = 1000
    # Store our best result
    best_loss, best_img = float('inf'), None
    norm_means = np.array([103.939, 116.779, 123.68])
    min_vals = -norm_means
    max_vals = 255 - norm_means

    for i in range(iterations):
        loss, c_loss, s_loss, v_loss, grads = compute_loss_and_grads(
            feature_extractor,
            combination_image,
            content_image,
            style_image,
            content_layer_name,
            content_weight,
            style_layer_names,
            style_weight,
            total_variation_weight)
        optimizer.apply_gradients([(grads, combination_image)])
        # clipping the image y to the range [0, 255] at each iteration
        clipped = tf.clip_by_value(combination_image, min_vals, max_vals)
        combination_image.assign(clipped)

        if loss < best_loss:
            # Update best loss and best image from total loss.
            best_loss = loss
            best_img = deprocess_image(combination_image.numpy())
            preprocessing.image.save_img("best.png", best_img)

        if i % 100 == 0:
            logging.info("Iteration %d: total_loss=%.4e, content_loss=%.4e, style_loss=%.4e, variation_loss=%.4e"
                         % (i, loss, c_loss, s_loss, v_loss))
            img = deprocess_image(combination_image.numpy())
            fname = "%s_iteration_%d.png" % (result_prefix, i)
            preprocessing.image.save_img(fname, img)


def content_reconstructions(feature_extractor):

    # block1_conv2
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': 'block1_conv2',
        'content_weight': 1,
        'style_layer_names': None,
        'style_weight': None,
        'total_variation_weight': 0,
        'result_prefix': 'content_reconstructions_block1_conv2',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)

    # block2_conv2
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': 'block2_conv2',
        'content_weight': 1,
        'style_layer_names': None,
        'style_weight': None,
        'total_variation_weight': 0,
        'result_prefix': 'content_reconstructions_block2_conv2',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)

    # block3_conv2
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': 'block3_conv2',
        'content_weight': 1,
        'style_layer_names': None,
        'style_weight': None,
        'total_variation_weight': 0,
        'result_prefix': 'content_reconstructions_block3_conv2',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)

    # block4_conv2
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': 'block4_conv2',
        'content_weight': 1,
        'style_layer_names': None,
        'style_weight': None,
        'total_variation_weight': 0,
        'result_prefix': 'content_reconstructions_block4_conv2',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)

    # block5_conv2
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': 'block5_conv2',
        'content_weight': 1,
        'style_layer_names': None,
        'style_weight': None,
        'total_variation_weight': 0,
        'result_prefix': 'content_reconstructions_block5_conv2',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)


def style_reconstructions(feature_extractor):

    # block1_conv1
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': None,
        'content_weight': None,
        'style_layer_names': [
            "block1_conv1",
        ],
        'style_weight': 1,
        'total_variation_weight': 0,
        'result_prefix': 'style_reconstructions_block1_conv1',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)

    # block2_conv1
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': None,
        'content_weight': None,
        'style_layer_names': [
            "block1_conv1",
            "block2_conv1",
        ],
        'style_weight': 1,
        'total_variation_weight': 0,
        'result_prefix': 'style_reconstructions_block2_conv1',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)

    # block3_conv1
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': None,
        'content_weight': None,
        'style_layer_names': [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
        ],
        'style_weight': 1,
        'total_variation_weight': 0,
        'result_prefix': 'style_reconstructions_block3_conv1',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)

    # block4_conv1
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': None,
        'content_weight': None,
        'style_layer_names': [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
        ],
        'style_weight': 1,
        'total_variation_weight': 0,
        'result_prefix': 'style_reconstructions_block4_conv1',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)

    # block5_conv1
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': None,
        'content_weight': None,
        'style_layer_names': [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ],
        'style_weight': 1,
        'total_variation_weight': 0,
        'result_prefix': 'style_reconstructions_block5_conv1',
        'init_random': True
    }

    neural_style_transfer(feature_extractor, **params)


def nst(feature_extractor):
    # nst
    params = {
        'content_image_path': 'Green_Sea_Turtle_grazing_seagrass.jpg',
        'style_image_path': 'The_Great_Wave_off_Kanagawa.jpg',
        'content_layer_name': 'block4_conv2',
        'content_weight': 1e-3,
        'style_layer_names': [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ],
        'style_weight': 1e0,
        'total_variation_weight': 0,
        'result_prefix': 'nst',
        'init_random': False
    }

    neural_style_transfer(feature_extractor, **params)


def run():

    # vis/content/style/nst
    mode = 'vis/content/style/nst'

    vgg_model = load_model()

    if 'vis' in mode:
        logging.info("visual model input filters")
        vis_model_input_filters(vgg_model)

    feature_extractor = create_feature_extract(vgg_model)

    if 'vis' in mode:
        logging.info("visual model feature maps")
        vis_model_feature_maps(feature_extractor)

    if 'content' in mode:
        logging.info("content reconstruction")
        content_reconstructions(feature_extractor)

    if 'style' in mode:
        logging.info("style reconstruction")
        style_reconstructions(feature_extractor)

    if 'nst' in mode:
        logging.info("neural style transfer")
        nst(feature_extractor)


def main(_):
    run()


if __name__ == '__main__':
    logging.set_verbosity(logging.INFO)
    app.run(main)
