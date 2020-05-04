"""Main file."""
import argparse
import os

import tensorflow as tf
from data import get_data
from model import get_model, gpu_configuration

tf.keras.backend.set_floatx('float64')
checkpoint_format = 'checkpoint.h5'

# Parse arguments. Require the user to provide a project/run name
parser = argparse.ArgumentParser(description='VAE')
parser.add_argument('--resume', default=False, action='store_true')
parser.add_argument('--checkpoint', default='newest')
parser.add_argument('name', type=str)
args = parser.parse_args()

# Setup GPU options
gpu_configuration()

# Create the model and maybe resume from checkpoint
vae, exp = get_model(
    project_name=args.name,
    resume=args.resume,
    checkpoint=args.checkpoint,
)

# Load and preprocess the data
train, test, steps_per_epoch, validation_steps = get_data(
    batch_size=exp['batch_size'],
    im_shape=exp['im_shape'],
    dataset=exp['dataset'],
)

# Train and evaluate the model
vae.fit(
    train,
    epochs=exp['epochs'],
    steps_per_epoch=steps_per_epoch,
    validation_data=test,
    validation_steps=validation_steps,
    callbacks=[
        # Log validation losses
        tf.keras.callbacks.CSVLogger(
            filename=os.path.join(args.name, 'losses.csv'),
            append=True
        ),
        # Create model checkpoints
        tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(args.name, checkpoint_format),
            save_weights_only=True,
            save_best_only=True,
            monitor='val_loss',
            save_freq='epoch',
            verbose=1,
        ),
    ],
)
