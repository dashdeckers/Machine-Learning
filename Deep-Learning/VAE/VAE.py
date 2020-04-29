"""Main file."""
import json
import math
import os
from os import path

from data import get_data
from model import (CustomCallback, getTensorboardCallback, gpu_configuration,
                   load_model, make_model, save_model)
from tensorflow.keras.callbacks import CSVLogger

# Define experiments
stanford_dogs = {
    'dataset': 'stanford_dogs',
    'im_shape': (32, 32),
    'channels': 3,
    'interm_dim': 512,
    'latent_dim': 30,
    'batch_size': 512,
    'epochs': 3600,
    'epsilon_std': 1.0,
    'beta': 1.0,
    'model_path': 'models_dogs',
    'checkpoint': 100,
}

mnist = {
    'dataset': 'mnist',
    'im_shape': (28, 28),
    'channels': 1,
    'interm_dim': 256,
    'latent_dim': 2,
    'batch_size': 512,
    'epochs': 10,
    'epsilon_std': 1.0,
    'beta': 1.0,
    'model_path': 'models_mnist',
    'checkpoint': 0,
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
            beta,
            model_path,
            checkpoint,
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

    # Load the model
    vae, encoder, decoder, = load_model(
        original_dim=original_dim,
        interm_dim=interm_dim,
        latent_dim=latent_dim,
        epochs=epochs,
        epsilon_std=epsilon_std,
        beta=beta,
        model_path=model_path,
        train=train,
    )

    # Or create the model
    if any([item is None for item in [vae, encoder, decoder]]):
        print('*' * 35)
        vae, encoder, decoder = make_model(
            original_dim=original_dim,
            interm_dim=interm_dim,
            latent_dim=latent_dim,
            epochs=epochs,
            epsilon_std=epsilon_std,
            beta=beta,
        )

    # Train the model
    vae.fit(
        train,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_data=test,
        validation_steps=val_steps,
        callbacks=[
            CustomCallback(
                path=model_path,
                checkpoint=checkpoint,
                encoder=encoder,
                decoder=decoder,
            ),
            getTensorboardCallback(
                path=model_path,
            ),
            CSVLogger(
                filename=path.join(model_path, 'losses.csv'),
                append=True
            ),
        ],
        # Verbose=2 shows a lot more warnings,
        # but it comes out nicer with the additional metrics
        verbose=2
    )

    # Save the model
    save_model(
        vae=vae,
        encoder=encoder,
        decoder=decoder,
        model_path=model_path,
    )

    # Save the experiment details
    with open(os.path.join(model_path, 'experiment.json'), 'w') as outfile:
        json.dump(globals()[dataset], outfile)


if __name__ == '__main__':
    main(**stanford_dogs)
    # main(**mnist)
