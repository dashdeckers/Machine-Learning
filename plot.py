import itertools
import json

import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import splev, splrep
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

data_dict = dict()
for path in path_strings:
    try:
        json_file = open('json/' + path)
        data = json.load(json_file)
        data_dict[path] = np.array(data)
    except FileNotFoundError:
        print("No file " + 'json/'+path)

### TRAIN/VAL Con
loss_val = "run-loss_val_Conjoined-tag-dat.json"
loss_train = "run-loss_train_Conjoined-tag-dat.json"

plt.scatter(data_dict[loss_train][:, 1], data_dict[loss_train][:, 2], label="Train", alpha=0.2)
plt.scatter(data_dict[loss_val][:, 1], data_dict[loss_val][:, 2], label="Validation", alpha=0.2)

# poly = splrep(data_dict[loss_train][:, 1], data_dict[loss_train][:, 2], s=1)
# poly_y = splev(data_dict[loss_train][:, 1], poly)
ys = lowess(data_dict[loss_train][:, 2], data_dict[loss_train][:, 1], frac=0.5)

plt.plot(data_dict[loss_train][:, 1], ys, c='C0')

ys = lowess(data_dict[loss_val][:, 2], data_dict[loss_val][:, 1], frac=0.5)

plt.plot(data_dict[loss_val][:, 1], ys, c='C1')

plt.legend()
plt.ylim(0.05, 0.065)
plt.xlabel("Batch")
plt.ylabel("BCE loss")

plt.title("Train and validation loss for Conjoined network")

plt.show()


### TRAIN/VAL Dis
loss_val = "run-loss_val_Disjoined-tag-dat.json"
loss_train = "run-loss_train_Disjoined-tag-dat.json"

plt.scatter(data_dict[loss_train][:, 1], data_dict[loss_train][:, 2], label="Train", alpha=0.2)
plt.scatter(data_dict[loss_val][:, 1], data_dict[loss_val][:, 2], label="Validation", alpha=0.2)

ys = lowess(data_dict[loss_train][:, 2], data_dict[loss_train][:, 1], frac=0.5)

plt.plot(data_dict[loss_train][:, 1], ys,  c='C0')

ys = lowess(data_dict[loss_val][:, 2], data_dict[loss_val][:, 1], frac=0.5)

plt.plot(data_dict[loss_val][:, 1], ys, c='C1')

plt.legend()
plt.ylim(0.05, 0.065)
plt.xlabel("Batch")
plt.ylabel("BCE loss")

plt.title("Train and validation loss for Disjoined network")

plt.show()

### LOSS

loss_con = "run-loss_val_Conjoined-tag-dat.json"
loss_dis = "run-loss_val_Disjoined-tag-dat.json"
plt.plot(data_dict[loss_con][:, 1], data_dict[loss_con][:, 2], label="Conjoined")
plt.plot(data_dict[loss_dis][:, 1], data_dict[loss_dis][:, 2], label="Disjoined")

# poly = splrep(data_dict[loss_con][:, 1], data_dict[loss_con][:, 2], s=100000)
# poly_y = splev(data_dict[loss_con][:, 1], poly)

# # plt.plot(data_dict[loss_con][:, 1], poly_y, label="Conjoined")

# poly = splrep(data_dict[loss_dis][:, 1], data_dict[loss_dis][:, 2], s=100000)
# poly_y = splev(data_dict[loss_dis][:, 1], poly)

# plt.plot(data_dict[loss_dis][:, 1], poly_y, label="Disjoined")

plt.legend()
plt.ylim(0.05, 0.065)
plt.xlabel("Batch")
plt.ylabel("BCE loss")

plt.title("Validation loss")

plt.show()


### PREDICTION CORRECT

metric_con = "run-Pred_corrects_val_Conjoined-tag-dat.json"
metric_dis = "run-Pred_corrects_val_Disjoined-tag-dat.json"
plt.plot(data_dict[metric_con][:, 1], data_dict[metric_con][:, 2],
         label="Conjoined", alpha=1)
plt.plot(data_dict[metric_dis][:, 1], data_dict[metric_dis][:, 2],
         label="Disjoined", alpha=1)

plt.legend()
plt.ylim(0.76, 0.8)
plt.xlabel("Batch")
plt.ylabel("Prediction accuracy")

plt.title("Validation prediction accuracy")

plt.show()


### CLASS CORRECT

metric_con = "run-Class_corrects_val_Conjoined-tag-dat.json"
metric_dis = "run-Class_corrects_val_Disjoined-tag-dat.json"
plt.plot(data_dict[metric_con][:, 1], data_dict[metric_con][:, 2],
         label="Conjoined", alpha=1)
plt.plot(data_dict[metric_dis][:, 1], data_dict[metric_dis][:, 2],
         label="Disjoined", alpha=1)

plt.legend()
plt.ylim(0.83, 0.88)
plt.xlabel("Batch")
plt.ylabel("Recognition accuracy")

plt.title("Validation recognition accuracy")

plt.show()


### CLASS CORRECT

metric_con = "run-True_corrects_val_Conjoined-tag-dat.json"
metric_dis = "run-True_corrects_val_Disjoined-tag-dat.json"
plt.plot(data_dict[metric_con][:, 1], data_dict[metric_con][:, 2],
         label="Conjoined", alpha=1)
plt.plot(data_dict[metric_dis][:, 1], data_dict[metric_dis][:, 2],
         label="Disjoined", alpha=1)
plt.legend()
plt.ylim(0.76, 0.8)
plt.xlabel("Batch")
plt.ylabel("Resulting accuracy")

plt.title("Validation accuracy of final output")

plt.show()
