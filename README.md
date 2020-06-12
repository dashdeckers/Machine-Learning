# Character_Denoising
We compare the performance difference between a Multi-Task-Learning setting and a traditional Single-Task-Learning setting using two BLSTM models trained to denoise corrupted character sequences.

The code requires python3.6 or higher to run.

To run, first install the required packages using pip in a virtual environment:
`pip install -r requirements.txt`

Then, run the script:
`python main.py`

When done, the script will have written the data to a folder named "run". Open Tensorboard to visualize the results by executing the following code in the terminal:
`tensorboard --logdir=runs`

To plot the results using the `plot.py` script, you need to download the JSON data for each of the logs via Tensorboard. Then, place them in a folder called `json` and run `plot.py` in the same directory.
