import os
from typing import Tuple
from absl import logging

import tensorflow as tf

from mob.pspm.style_network import preprocessing


def build(data_dir, split, batch_size) -> tf.data.Dataset:
    dataset = load_records(data_dir, split)
    dataset = pipeline(dataset, batch_size)
    return dataset


def load_records(data_dir, split) -> tf.data.Dataset:
    """Return a dataset loading files with TFRecords."""
    logging.info('Using TFRecords to load data.')
    if data_dir is None:
        raise ValueError('Dataset must specify a path for the data files.')
    file_pattern = os.path.join(data_dir, '{}-?????-of-?????.tfrecord'.format(split))
    dataset = tf.data.Dataset.list_files(file_pattern, shuffle=True)
    return dataset


def pipeline(dataset: tf.data.Dataset, batch_size) -> tf.data.Dataset:
    """Build a pipeline fetching, shuffling, and preprocessing the dataset.

    Args:
      dataset: A `tf.data.Dataset` that loads raw files.
      batch_size: batch size
    Returns:
      A TensorFlow dataset outputting batched images and labels.
    """

    shuffle_buffer_size = 10000
    num_devices = 1

    # Read the data from disk in parallel
    buffer_size = 8 * 1024 * 1024  # Use 8 MiB per file
    dataset = dataset.interleave(
        lambda name: tf.data.TFRecordDataset(name, buffer_size=buffer_size),
        cycle_length=16,
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    dataset = dataset.prefetch(batch_size)

    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.repeat()

    # Parse, pre-process, and batch the data in parallel
    dataset = dataset.map(parse_record, num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.batch(batch_size, drop_remainder=True)

    # Note: we could do image normalization here, but we defer it to the model
    # which can perform it much faster on a GPU/TPU
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_slack = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_vectorization.enabled = True
    options.experimental_optimization.map_parallelization = True
    dataset = dataset.with_options(options)


    # Prefetch overlaps in-feed with training
    # Note: autotune here is not recommended, as this can lead to memory leaks.
    # Instead, use a constant prefetch size like the the number of devices.
    dataset = dataset.prefetch(num_devices)

    return dataset


def parse_record(record: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
    """Parse an MSCCO record from a serialized string Tensor."""
    keys_to_features = {
        'image/encoded':
            tf.io.FixedLenFeature((), tf.string, ''),
        'image/format':
            tf.io.FixedLenFeature((), tf.string, 'jpeg'),
        'image/class/label':
            tf.io.FixedLenFeature([], tf.int64, -1),
        'image/class/text':
            tf.io.FixedLenFeature([], tf.string, ''),
        'image/object/bbox/xmin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymin':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/xmax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/bbox/ymax':
            tf.io.VarLenFeature(dtype=tf.float32),
        'image/object/class/label':
            tf.io.VarLenFeature(dtype=tf.int64),
    }

    parsed = tf.io.parse_single_example(record, keys_to_features)

    label = tf.reshape(parsed['image/class/label'], shape=[1])
    label = tf.cast(label, dtype=tf.int32)

    # Subtract one so that labels are in [0, num_classes)
    label -= 1

    image_bytes = tf.reshape(parsed['image/encoded'], shape=[])
    image, label = preprocess(image_bytes, label)

    return image, label


def preprocess(image: tf.Tensor, label: tf.Tensor
               ) -> Tuple[tf.Tensor, tf.Tensor]:
    """Apply image preprocessing and augmentation to the image and label."""
    image = preprocessing.preprocess_for_train(
        image,
        image_size=256,
        mean_subtract=False,
        standardize=False,
        dtype=tf.float32)

    label = tf.cast(label, tf.int32)

    return image, label