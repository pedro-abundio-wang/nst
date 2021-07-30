import tensorflow as tf


def transfer(content,
             checkpoint,
             result):
    tf.train.latest_checkpoint(checkpoint)
