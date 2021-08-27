from tensorflow.keras import models

from utils import tensor_to_image, load_img, clip, resolve_video

IMAGE_TYPE = ('jpg', 'jpeg', 'png', 'bmp')


def transfer(content, saved_model_path, max_dim, result):

    # Build the feed-forward network and load the weights.
    transformation_model = models.load_model(saved_model_path)

    if content[-3:] in IMAGE_TYPE:

        # Load content image.
        image = load_img(path_to_img=content, max_dim=max_dim, resize=False)
        
        print('Transfering image...')
        # Geneerate the style imagee
        image = transformation_model(image)

        # Clip pixel values to 0-255
        image = clip(image)

        # Save the style image
        tensor_to_image(image).save(result)

    else:
        resolve_video(transformation_model, path_to_video=content, result=result)