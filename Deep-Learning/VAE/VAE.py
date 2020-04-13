import math
import os
import tensorflow as tf
import numpy as np
import json

from data import get_data
from model import CustomCallback, gpu_configuration, make_model, load_model, save_model

# Define experiments
stanford_dogs = {
    'dataset': 'stanford_dogs',
    'im_shape': (32, 32),  # (32, 32)
    'channels': 3,
    'interm_dim': 256,
    'latent_dim': 15,
    'batch_size': 512,
    'epochs': 10,
    'epsilon_std': 1.0,
    'model_path': 'models_dogs',
    'checkpoint': 100,
    'resume' : False,
}

mnist = {
    'dataset': 'mnist',
    'im_shape': (28, 28),
    'channels': 1,
    'interm_dim': 256,
    'latent_dim': 2,
    'batch_size': 512,
    'epochs': 1,
    'epsilon_std': 1.0,
    'model_path': 'models_mnist',
    'checkpoint': 0,
    'resume' : False,
}


def main(
            dataset,
            im_shape,
            channels,
            interm_dim,
            latent_dim,
            batch_size,
            epochs,
            epsilon_std,
            model_path,
            checkpoint,
            resume
        ):
    """Run the experiment."""
    original_dim = im_shape[0] * im_shape[1] * channels

    # Get the data
    train, test, info = get_data(
        batch_size=batch_size,
        im_shape=im_shape,
        dataset=dataset,
    )

    # Define more globals
    steps_per_epoch = math.ceil(info.splits["train"].num_examples / batch_size)
    val_steps = math.ceil(info.splits["test"].num_examples / batch_size)

    # Setup GPU options
    gpu_configuration()

    if resume:
        vae, encoder, decoder, resume = load_model(
                                            original_dim=original_dim,
                                            interm_dim=interm_dim,
                                            latent_dim=latent_dim,
                                            epochs=epochs,
                                            epsilon_std=epsilon_std,
                                            model_path=model_path,
                                            resume=resume,
                                            train=train,
                                        )
    if not resume:
        # Make the model
        print("********************************")
        vae, encoder, decoder = make_model(
            original_dim=original_dim,
            interm_dim=interm_dim,
            latent_dim=latent_dim,
            epochs=epochs,
            epsilon_std=epsilon_std,
        )

    # Train the model
    vae.fit(
        train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test,
        validation_steps=val_steps,
        callbacks=[CustomCallback(
                path=model_path, 
                checkpoint=checkpoint, 
                encoder=encoder, 
                decoder=decoder,
            )],
    )

    # Save the model
    save_model(
        vae=vae,
        encoder=encoder,
        decoder=decoder,
        model_path=model_path,
    )
    # Save the experiment details
    dict = mnist if dataset=="mnist" else stanford_dogs

    with open(os.path.join(model_path,'experiment.json'), 'w') as outfile:
        json.dump(dict, outfile)

if __name__ == '__main__':
    # main(**stanford_dogs)
    main(**mnist)
