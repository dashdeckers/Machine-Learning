import glob
import json
from os import makedirs

# Experiment dictionary must contain these entries
assert [key in exp for key in [
    'dataset', 'im_shape', 'channels', 'latent_dim', 'batch_size',
    'epochs', 'beta', 'model_path', 'checkpoint_format',
]], exp.items()


layers.Conv2D(
    input_shape=input_shape,
    data_format='channels_last',
    filters=32,
    kernel_size=2,
    padding='same',
    activation='relu',
    name='Conv-1',
),
layers.Conv2D(
    filters=64,
    kernel_size=2,
    padding='same',
    activation='relu',
    strides=(2, 2),
    name='Conv-2',
),

layers.Conv2DTranspose(
    filters=64,
    kernel_size=3,
    strides=(1, 1),
    padding='same',
    activation='relu',
    name='DeConv-1',
),
layers.Conv2DTranspose(
    filters=exp['channels'],
    kernel_size=3,
    strides=(1, 1),
    padding='same',
    activation='relu',
    name='DeConv-2',
),


def get_model(exp):
    """Load a model from checkpoint or create a new one.

    Manages loading from most recent checkpoint, creating the
    directories, returning the model.
    """
    model_path = exp['model_path']
    input_shape = (1,) + exp['im_shape'] + (exp['channels'], )

    vae = VariationalAutoEncoder(exp)
    vae.compile(optimizer=tf.keras.optimizers.Adam())
    vae(tf.zeros(input_shape))  # create the weights

    if path.exists(model_path):
        # Grab the newest .h5 model checkpoint file, if it exists
        list_of_saved_models = glob.glob(path.join(model_path, '*.h5'))

        if list_of_saved_models:
            latest_model = max(list_of_saved_models, key=path.getctime)

            # Load the latest checkpoint model
            print('Loading the model from checkpoint', latest_model, '\n')
            vae.load_weights(latest_model)

    else:
        # Create a new model folder and write the experiment dict to file
        print('Creating a new model at', model_path, '\n')
        makedirs(model_path)
        with open(path.join(model_path, 'experiment.json'), 'w') as file:
            file.write(json.dumps(exp, indent=4))

    return vae



# Load or create the model
vae = get_model(exp['model_path'], input_shape=(1, 28, 28, 1))


# Basic visualization of images to track progress
def show_digit(output):
    plt.imshow(tf.reshape(output, (28, 28)).numpy())
    plt.show()


x_test = list(test.take(1).as_numpy_iterator())[0][0]
test_input = tf.reshape(x_test[0], (1, 28, 28, 1))

show_digit(test_input)
show_digit(vae(test_input))

"""
>>> Alternative MNIST source:

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32) / 255.0
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32) / 255.0

x_train = tf.reshape(x_train, (60000, 28, 28, 1))
x_test = tf.reshape(x_test, (10000, 28, 28, 1))


vae.fit(
    x_train,
    x_train,
    epochs=10,
    batch_size=64,
)
print('Validation loss:', vae.evaluate(
    x_test,
    x_test,
    batch_size=64,
))

"""
