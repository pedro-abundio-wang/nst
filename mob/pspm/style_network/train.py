import numpy as np
import tensorflow as tf

from tensorflow.keras import applications

from utils import load_img
from .models import transformation_network
from .models import StyleContentModel
from .models import style_loss
from .models import content_loss
from .models import total_variation_loss


def trainer(style_file, dataset_path, weights_path, content_weight, style_weight,
            tv_weight, learning_rate, batch_size, epochs, debug):
    # Setup the given layers
    content_layers = ['block4_conv2']

    style_layers = ['block1_conv1',
                    'block2_conv1',
                    'block3_conv1',
                    'block4_conv1',
                    'block5_conv1']

    # Build style-network transformer
    transformation_model = transformation_network()

    # Build VGG-19 Loss network
    loss_model = StyleContentModel(style_layers, content_layers)

    # Load style target image
    style_image = load_img(style_file, resize=False)

    # Initialize content target images
    batch_shape = (batch_size, 256, 256, 3)
    input_batch = np.zeros(batch_shape, dtype=np.float32)

    # Extract style target 
    style_target = loss_model(style_image * 255.0)['style']

    # Build optimizer
    opt = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    loss_metric = tf.keras.metrics.Mean()
    sloss_metric = tf.keras.metrics.Mean()
    closs_metric = tf.keras.metrics.Mean()
    tloss_metric = tf.keras.metrics.Mean()

    @tf.function()
    def train_step(content_images):
        with tf.GradientTape() as tape:
            content_target = loss_model(content_images * 255.0)['content']
            combination_images = transformation_model(content_images)
            outputs = loss_model(combination_images)

            s_loss = style_weight * style_loss(outputs['style'], style_target)
            c_loss = content_weight * content_loss(outputs['content'], content_target)
            t_loss = tv_weight * total_variation_loss(combination_images)
            loss = s_loss + c_loss + t_loss

        grad = tape.gradient(loss, transformation_model.trainable_variables)
        opt.apply_gradients(zip(grad, transformation_model.trainable_variables))

        loss_metric(loss)
        sloss_metric(s_loss)
        closs_metric(c_loss)
        tloss_metric(t_loss)

    train_dataset = tf.data.Dataset.list_files(dataset_path + '/*.jpg')
    train_dataset = train_dataset.map(load_img,
                                      num_parallel_calls=tf.data.experimental.AUTOTUNE)
    train_dataset = train_dataset.shuffle(1024)
    train_dataset = train_dataset.batch(batch_size, drop_remainder=True)
    train_dataset = train_dataset.prefetch(tf.data.experimental.AUTOTUNE)

    import time
    start = time.time()

    for e in range(epochs):
        print('Epoch {}'.format(e))
        iteration = 0

        for image_batch in train_dataset:

            for i, image in enumerate(image_batch):
                input_batch[i] = image

            iteration += 1

            train_step(input_batch)

            if iteration % 3000 == 0:
                # Save checkpoints
                transformation_model.save_weights(weights_path, save_format='tf')
                print('=====================================')
                print('            Weights saved!           ')
                print('=====================================\n')

                if debug:
                    print('step %s: loss = %s' % (iteration, loss_metric.result()))
                    print('s_loss={}, c_loss={}, t_loss={}'.format(sloss_metric.result(), closs_metric.result(),
                                                                   tloss_metric.result()))

    end = time.time()
    print("Total time: {:.1f}".format(end - start))

    # Training is done !
    transformation_model.save_weights(weights_path, save_format='tf')
    print('=====================================')
    print('             All saved!              ')
    print('=====================================\n')
