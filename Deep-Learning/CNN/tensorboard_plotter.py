import os
from itertools import groupby
from typing import List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib import cm
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
                # print(value.tag)
                # 'epoch_loss' is also an available tag name
                if value.tag == 'epoch_accuracy':
                    accs.append(value.simple_value)
                if value.tag == 'epoch_categorical_accuracy':
                    cat_accs.append(value.simple_value)

        data.append((name, metric_type, accs, cat_accs))

# Define the comparison sets
comparison_sets = {
    'activations': {'baseline', 'elu', 'linear', 'sigmoid', 'softplus'},
    'optimizers': {'baseline', 'sgd', 'rmsprop', 'adadelta'},
    'architectures': {'baseline', 'short_cnn', 'long_cnn', 'small_model'},
    'dropouts': {'baseline', 'none', 'high'},
}

# Define color maps
colors = [
    cm.get_cmap('Reds'),
    cm.get_cmap('Blues'),
    cm.get_cmap('Greens'),
    cm.get_cmap('Purples'),
    cm.get_cmap('Greys'),
    cm.get_cmap('Oranges'),
]

# Plot the data
n_epochs = 20
comparison = 'activations'
metric = 'accuracy'

# Remove any incomplete runs (runs must have exactly n_epochs datapoints)
data = [tup for tup in data if all(len(run) == n_epochs for run in tup[2:])]

# Group runs by name to group train and validation data together
for name, (valid, train) in groupby(data, lambda g: g[0]):
    # Make sure train and validation data are not mixed up
    assert valid[1] == 'validation'
    assert train[1] == 'train'
    train_data = train[2:]
    valid_data = valid[2:]

    # Format the name into a nice label
    if name == 'baseline':
        label = 'baseline'
    else:
        label = '_'.join(name.split('_')[1:])

    # Convert metric to metric_idx
    metric_idx = {'accuracy': 0, 'categorical_accuracy': 1}[metric]

    # Filter out runs that we are not comparing
    if label in comparison_sets[comparison]:
        cmap = colors.pop()

        # Plot the train and validation data of a run with similar colors
        plt.plot(
            range(n_epochs),
            train_data[metric_idx],
            label=f'{label} train',
            color=cmap(0.5)
        )
        plt.plot(
            range(n_epochs),
            valid_data[metric_idx],
            label=f'{label} validation',
            color=cmap(0.9)
        )

    t, v = train_data[metric_idx], valid_data[metric_idx]
    print(f'Run Name: {label}, metric: {metric}')
    print(f'Best Train: {round(max(t), 2)}')
    print(f'Best Validation: {round(max(v), 2)}')
    print(f'Average Train: {round(sum(t) / len(t), 2)}')
    print(f'Average Validation: {round(sum(v) / len(v), 2)}\n')

plt.title(f'Training and validation errors for different {comparison}')
plt.ylabel(f"{' '.join(metric.split('_')).title()}")
plt.xlabel('Epoch')
plt.xticks(list(range(n_epochs)))
plt.yticks([x/10 for x in range(10)])
plt.legend()
plt.show()
