# Usage
The code is meant to be used interactively via the python IDLE, an easy way to drop into a file interactively is to use the `-i` flag. For example, to do a single Rosenblatt training run just type `python -i rosenblatt.py`, and then in the interpreter you can execute the function `run_rosenblatt()` with any optional parameters you want.
To reproduce any of the full experiments, open the file `experiments.py` and execute the `collect_data()` with one of the pre-set experiment dictionaries. For example to reproduce the results from assignment 1, execute `collect_data(experiment1)`.
