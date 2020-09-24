import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess

# Define strings for TB logging lines
runs = "run-"
metrics = ["loss", "True_corrects", "Pred_corrects",
           "Class_corrects", "Repaired_corrects", "Change_rate"]
datas = ["train", "val", "test"]
joins = ["Conjoined", "Disjoined"]
append = "-tag-dat.json"

# Concatenate combinations of these strings
cores = list(itertools.product(metrics, datas, joins))
core_strings = ['_'.join(line)for line in cores]
path_strings = [runs + core_string + append for core_string in core_strings]

# Load the data from the strings
data_dict = dict()
for path in path_strings:
    try:
        json_file = open('json/' + path)
        data = json.load(json_file)
        data_dict[path] = np.array(data)
    except FileNotFoundError:
        print("No file " + 'json/'+path)


# TRAIN/VAL Losses


fig, axs = plt.subplots(nrows=1, ncols=2)

mtl_loss_val = "run-loss_val_Conjoined-tag-dat.json"
mtl_loss_train = "run-loss_train_Conjoined-tag-dat.json"

stl_loss_val = "run-loss_val_Disjoined-tag-dat.json"
stl_loss_train = "run-loss_train_Disjoined-tag-dat.json"

losses = [
    (mtl_loss_train, mtl_loss_val, 'MTL'),
    (stl_loss_train, stl_loss_val, 'STL')
]

for plt_idx, (train_key, val_key, model) in enumerate(losses):
    train = data_dict[train_key]
    val = data_dict[val_key]

    for c_idx, (dataset, label) in enumerate([(train, 'train'), (val, 'val')]):
        color = 'C' + str(c_idx)
        label = model + '_' + label

        # The datapoints
        axs[plt_idx].scatter(
            dataset[:, 1], dataset[:, 2],
            label=label, alpha=0.2, c=color
        )

        # The line
        smoothed = lowess(dataset[:, 2], dataset[:, 1], frac=0.5)
        axs[plt_idx].plot(dataset[:, 1], smoothed, c=color)

        # The error lines + shading
        abs_diff = abs(dataset[:, 2] - smoothed[:, 1])
        smoothed_diff = lowess(abs_diff, dataset[:, 1], frac=0.5)

        axs[plt_idx].fill_between(
            dataset[:, 1],
            (smoothed - smoothed_diff)[:, 1],
            (smoothed + smoothed_diff)[:, 1],
            alpha=0.5,
            edgecolor=color,
            facecolor=color
        )

    # The rest
    axs[plt_idx].legend()
    axs[plt_idx].set_ylim(0.02, 0.15)  # 0.05, 0.065 || 0.045, 0.09
    axs[plt_idx].set_xlabel("Batch")

axs[0].set_ylabel("BCE loss")
axs[0].set_title("MTL model")
axs[1].set_title("STL model")

axs[1].tick_params(
    axis='y',
    which='both',
    left=False,
    labelleft=False
)

assert axs[0].get_ylim() == axs[1].get_ylim(), axs[0].get_ylim()

fig.suptitle('Train and validation loss for the MTL and STL models')
fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # Make the title show right
plt.show()


# TRAIN/VAL Accuracies


pred_mtl = "run-Pred_corrects_val_Conjoined-tag-dat.json"
pred_stl = "run-Pred_corrects_val_Disjoined-tag-dat.json"

clas_mtl = "run-Class_corrects_val_Conjoined-tag-dat.json"
clas_stl = "run-Class_corrects_val_Disjoined-tag-dat.json"

comb_mtl = "run-True_corrects_val_Conjoined-tag-dat.json"
comb_stl = "run-True_corrects_val_Disjoined-tag-dat.json"

data_labels = [
    (pred_mtl, 'MTL prediction', 'darkblue'),
    (pred_stl, 'STL prediction', 'lightblue'),

    (clas_mtl, 'MTL classification', 'darkorange'),
    (clas_stl, 'STL classification', 'gold'),

    (comb_mtl, 'MTL combined', 'darkviolet'),
    (comb_stl, 'STL combined', 'fuchsia'),
]

for key, label, color in data_labels:
    plt.plot(data_dict[key][:, 1], data_dict[key][:, 2],
             label=label, alpha=1, c=color)

plt.legend()
plt.ylim(0.5, 0.875)
plt.xlabel("Batch")
plt.ylabel("Accuracy")
plt.title("Validation accuracies")

plt.show()
