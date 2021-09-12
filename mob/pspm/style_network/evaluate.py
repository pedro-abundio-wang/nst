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
        result_dir = os.path.join(dirname, 'saved_model')
        result_path = os.path.join(result_dir, basename)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        tensor_to_image(image).save(result_path)

        ##################################################

        # Load TFLite model
        interpreter = tf.lite.Interpreter(tflite_model_path)

        # Get input and output tensors.
        input_index = interpreter.get_input_details()[0]["index"]
        output_index = interpreter.get_output_details()[0]["index"]

        # resize input shape and allocate tensors
        interpreter.resize_tensor_input(input_index, tf.shape(content_image).numpy(), strict=True)
        interpreter.allocate_tensors()

        interpreter.set_tensor(input_index, content_image)
        interpreter.invoke()
        image = interpreter.get_tensor(output_index)

        # Clip pixel values to 0-255
        image = clip(image)

        # Save the style image
        result_dir = os.path.join(dirname, 'tflite_model')
        result_path = os.path.join(result_dir, basename)
        if not os.path.exists(result_dir):
            os.mkdir(result_dir)
        tensor_to_image(image).save(result_path)

    else:
        # Build the feed-forward network and load the weights.
        transformation_model = models.load_model(saved_model_path)
        resolve_video(transformation_model, path_to_video=content, result=result)