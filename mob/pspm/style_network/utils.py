from PIL import Image
from tensorflow.keras import preprocessing
import numpy as np


def load_image(path_to_image, max_dim = 512):
    image = Image.open(path_to_image)
    if max_dim:
        long = max(image.size)
        scale = max_dim / long
        image = image.resize((round(image.size[0] * scale), round(image.size[1] * scale)), Image.ANTIALIAS)
    image = preprocessing.image.img_to_array(image)
    # We need to broadcast the image array such that it has a batch dimension
    image = np.expand_dims(image, axis=0)
    return image