# Variational Autoencoders
Here is the implementation of the Beta-VAE and Beta-TCVAE for the Deep Learning course at the University of Groningen.

## Running the VAE
`python VAE.py <experiment name>`. A folder called `experiment_name` will be created.
Optional arguments:
* `--beta <beta value (float)>`. Default: `1.0`.
* `--tc <True|False>`. Default: `False`
* `--dataset <mnist|stanford_dogs|celeb_a|chairs>`. Default: `stanford_dogs`.

## Plotting results
Results can be visualized using `plot.py <experiment name>`. The images can be found in the `experiment_name` folder.

## Peregrine
If you have access to the Peregrine cluster of the RUG, you can run one of the provided jobscripts using e.g. `sbatch jobscript-celeba` to run one of the experiments.
