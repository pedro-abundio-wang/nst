import os
import cv2
import PIL.Image

import numpy as np
import tensorflow as tf

from mob.pspm.style_network.style_network import transformation_network
from mob.pspm.style_network import utils

def transfer(content,
             checkpoint,
             result):

    image_type = ('jpg', 'jpeg', 'png', 'bmp')

    if content[-3:] in image_type:
        # Build the feed-forward network and load the weights.
        transformation_model = transformation_network()
        checkpoint_prefix = os.path.join(checkpoint, "ckpt")
        transformation_model.load_weights(checkpoint_prefix)

        # Load content image.
        image = utils.load_image(path_to_image=content, max_dim=None)

        print('Transfering image...')
        # Geneerate the style imagee
        image = transformation_model(image)

        # Clip pixel values to 0-255
        image = clip_image(image)

        # Save the style image
        tensor_to_image(image).save(result)

    else:
        transformation_model = transformation_network()
        checkpoint_prefix = os.path.join(checkpoint, "ckpt")
        transformation_model.load_weights(checkpoint_prefix)
        resolve_video(transformation_model, path_to_video=content, result=result)


def resolve_video(network, path_to_video, result):
    cap = cv2.VideoCapture(path_to_video)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(result, fourcc, 30.0, (640,640))

    while cap.isOpened():
        ret, frame = cap.read()
        #frame = cv2.resize(frame, (256, 256), interpolation = cv2.INTER_LINEAR)

        print('Transfering video...')
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = tf.cast(frame[tf.newaxis, ...], tf.float32) / 255.0

        prediction = network(frame)

        prediction = clip_image(prediction)
        prediction = np.array(prediction).astype(np.uint8).squeeze()
        prediction = cv2.cvtColor(prediction, cv2.COLOR_RGB2BGR)

        out.write(prediction)
        cv2.imshow('prediction', prediction)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    out.release()
    cv2.destroyALLWindow()


def tensor_to_image(tensor):
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return PIL.Image.fromarray(tensor)


def clip_image(image):
    return tf.clip_by_value(image, clip_value_min=0.0, clip_value_max=255.0)