"""Load and preprocess the tensorflow datasets and prepare it."""
from functools import partial

import tensorflow as tf
import tensorflow_datasets as tfds
from tqdm import tqdm

tf.keras.backend.set_floatx('float64')


def preprocessing(data, im_shape, labels=False):
    """Cast images to tensorflow format."""
    image = tf.cast(data["image"], tf.float32)
    # Normalize
    image = image / 255.0
    # Resize the image
    image = tf.image.resize(image, im_shape)
    # Explicitly cast back to float64 to avoid warnings when
    # it is fed to the network
    image = tf.cast(image, tf.float64)

    if labels:
        return image, data["label"]
    else:
        return image, image


def read_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)

    data = {
        'image': image
    }

    return data


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

    if (dataset == "celeb_a"):
        # celeba is in TFDS, but frequently this throws errors due to it being
        # hosted on Google Drive
        # download:  https://www.kaggle.com/jessicali9530/celeba-dataset/data
        # change the path to your local copy of celeba
        celeb_a = tf.data.Dataset.list_files(
            str('/data/s2759799/img_align_celeba/img_align_celeba/*')
        )
        celeb_a = celeb_a.map(
            read_image,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        test_data = celeb_a.take(8000)  # 8000 test samples
        train_data = celeb_a.skip(8000).take(40000)  # 40000 training samples
        steps_per_epoch = int(tf.math.ceil(40000 / batch_size))
        validation_steps = int(tf.math.ceil(8000 / batch_size))

    elif (dataset == "chairs"):
        # the chairs dataset is not in TFDS
        # download: https://www.di.ens.fr/willow/research/seeing3Dchairs/
        # change the path to your local copy of the chairs dataset
        # make sure to move all the images to the root folder
        chairs = tf.data.Dataset.list_files(
            str('/data/s2759799/rendered_chairs/*.png*')
        )
        chairs = chairs.map(
            read_image,
            num_parallel_calls=tf.data.experimental.AUTOTUNE
        )
        test_data = chairs.take(8000)  # 8000 test samples
        train_data = chairs.skip(8000).take(40000)  # 40000 training samples
        steps_per_epoch = int(tf.math.ceil(40000 / batch_size))
        validation_steps = int(tf.math.ceil(8000 / batch_size))

    else:
        train_data, info = tfds.load(
            name=dataset,
            split="train",
            with_info=True,
        )

        test_data = tfds.load(
            name=dataset,
            split="test",
        )

        steps_per_epoch = int(tf.math.ceil(
                info.splits["train"].num_examples / batch_size
        ))
        validation_steps = int(tf.math.ceil(
                info.splits["test"].num_examples / batch_size
        ))

    if dataset == 'stanford_dogs':
        crop_dogs(train_data)
        crop_dogs(test_data)

    train = (
        train_data
        .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .cache()
        .shuffle(10 * batch_size)
        .repeat()
        .batch(batch_size)
        .prefetch(5)
    )

    test = (
        test_data
        .map(preprocess, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        .cache()
        .repeat()
        .batch(batch_size)
        .prefetch(5)
    )

    return train, test, steps_per_epoch, validation_steps
