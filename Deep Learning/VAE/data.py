from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds


def preprocessing(data, im_shape, labels=False):
    image = tf.cast(data["image"], tf.float32)
    # Normalize
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, im_shape)
    image = tf.reshape(image, [-1])

    if labels:
        return image, data["label"]
    else:
        return image, image


def get_data(batch_size, im_shape, labels=False, dataset='stanford_dogs'):
    """Load the Stanford Dogs dataset from TensorFlow and return it."""

    preprocess = partial(preprocessing, im_shape=im_shape, labels=labels)

    train_data, info = tfds.load(
        name=dataset,
        split="train",
        with_info=True,
    )

    test_data = tfds.load(
        name=dataset,
        split="test",
    )

    train = (
        train_data.cache()
        .shuffle(10 * batch_size)
        .repeat()
        .map(preprocess)
        .batch(batch_size)
        .prefetch(5)
    )

    test = (
        test_data.cache()
        .repeat()
        .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .batch(batch_size)
        .prefetch(5)
    )

    return train, test, info
