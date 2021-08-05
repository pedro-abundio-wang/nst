from utils import tensor_to_image, load_img, clip, resolve_video
from mob.pspm.style_network.models import transformation_network

image_type = ('jpg', 'jpeg', 'png', 'bmp')


def transfer(content, weights, max_dim, result):

    if content[-3:] in image_type:
        # Build the feed-forward network and load the weights.
        transformation_model = transformation_network()
        transformation_model.load_weights(weights).expect_partial()

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
        transformation_model = transformation_network()
        transformation_model.load_weights(weights)

        resolve_video(transformation_model, path_to_video=content, result=result)