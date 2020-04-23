"""Serves to load the tensorflow dataset and prepare it."""
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm


def preprocessing(data, im_shape, labels=False):
    """Cast images to tensorflow format."""
    image = tf.cast(data["image"], tf.float32)
    # Normalize
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, im_shape)
    # image = tf.reshape(image, [-1])

    if labels:
        return image, data["label"]
    else:
        return image, image


def crop_dogs(data):
    """Use the bounding box annotations to crop the dogs from the images."""
    print("Cropping dogs...")
    for elem in tqdm(data):
        image = elem["image"]
        image_height, image_width = image.shape[0], image.shape[1]

        # Only use the first dog (some images have multiple)
        bbox = elem["objects"]["bbox"][0]

        # Calculate the offsets in pixels
        offset_h = int((bbox[0] * image_height).numpy())
        offset_w = int((bbox[1] * image_width).numpy())

        # Calculate the target dimensions in pixels
        # Note that the annotations are of the form [xmin, ymin, xmax, ymax]
        # crop_to_bounding_box expects the form [xmin, ymin, width, height]
        # Therefore we must subtract the offset.
        target_h = int((bbox[2] * image_height).numpy() - offset_h)
        target_w = int((bbox[3] * image_width).numpy() - offset_w)

        # Crop
        elem["image"] = tf.image.crop_to_bounding_box(
            image, offset_h, offset_w, target_h, target_w
        )

    print("Done.")
    return data


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

    if dataset == 'stanford_dogs':
        crop_dogs(train_data)
        crop_dogs(test_data)

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
