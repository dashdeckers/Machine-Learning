import os
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from tensorflow.python.summary.summary_iterator import summary_iterator

mpl.style.use('seaborn')

# Type annotations, because this is cool now
Scalars = List[float]
Data = List[Tuple[str, str, Scalars, Scalars]]

print('========================SHIT STARTS HERE========================')

# Iterate recursively over every file in the logs/ directory
data: Data = list()
for dirpath, dirnames, filenames in os.walk('logs'):
    for file in filenames:
        # Instantiate the summary_iterator that can read the file
        full_path = os.path.abspath(os.path.join(dirpath, file))
        iterator = summary_iterator(full_path)

        # Get the name of the run without the time+date
        folders = os.path.normpath(full_path).split(os.path.sep)
        name = ' '.join(folders[6].split()[0:-2])
        metric_type = folders[7]

        # Get the metrics we want from the file
        accs: Scalars = list()
        cat_accs: Scalars = list()
        for event in iterator:
            for value in event.summary.value:
                print(value.tag)
                # 'epoch_loss' is also an available tag name
                if value.tag == 'epoch_accuracy':
                    accs.append(value.simple_value)
                if value.tag == 'epoch_categorical_accuracy':
                    cat_accs.append(value.simple_value)

        data.append((name, metric_type, accs, cat_accs))

# List the possible runs
activations = {'baseline', 'elu', 'linear', 'sigmoid', 'softplus'}
optimizers = {'baseline', 'sgd', 'rmsprop', 'adadelta'}
architectures = {'baseline', 'short_cnn', 'long_cnn', 'small_model'}
dropouts = {'baseline', 'none', 'high'}

# Plot the data
n_epochs = 20
for tup in data:
    if tup[0] == 'baseline':
        name = 'baseline'
    else:
        name = ''.join(tup[0].split('_')[1:])

    if name in activations:
        plt.plot(range(n_epochs), tup[2], label=f'{tup[0]} {tup[1]}')

plt.legend()
plt.show()
