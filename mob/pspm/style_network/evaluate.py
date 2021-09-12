import os

import tensorflow as tf
from tensorflow.keras import models

from utils import tensor_to_image, load_img, clip, resolve_video

IMAGE_TYPE = ('jpg', 'jpeg', 'png', 'bmp')


def transfer(content, saved_model_path, tflite_model_path, max_dim, result):

    if content[-3:] in IMAGE_TYPE:

        dirname = os.path.dirname(result)
        basename = os.path.basename(result)

        # Build the feed-forward network and load the weights.
        transformation_model = models.load_model(saved_model_path)

        # Load content image.
        content_image = load_img(path_to_img=content, max_dim=max_dim, resize=False)
        
        print('Transfering image...')
        # Geneerate the style imagee
        image = transformation_model(content_image)

        # Clip pixel values to 0-255
        image = clip(image)

        # Save the style image
        result_path = os.path.join(dirname, 'saved_model', basename)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        tensor_to_image(image).save(result_path)

        ##################################################

        # Load the TFLite model in TFLite Interpreter
        interpreter = tf.lite.Interpreter(tflite_model_path)
        # There is only 1 signature defined in the model,
        # so it will return it by default.
        # If there are multiple signatures then we can pass the name.
        signature = interpreter.get_signature_runner()

        # signature is callable with input as arguments.
        image = signature(content_image)

        # Clip pixel values to 0-255
        image = clip(image)

        # Save the style image
        result_path = os.path.join(dirname, 'tflite_model', basename)
        if not os.path.exists(dirname):
            os.mkdir(dirname)
        tensor_to_image(image).save(result_path)

    else:
        # Build the feed-forward network and load the weights.
        transformation_model = models.load_model(saved_model_path)
        resolve_video(transformation_model, path_to_video=content, result=result)