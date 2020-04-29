import csv
from os import path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

dogs_path = 'models_dogs'
mnist_path = 'models_mnist'

model_path = mnist_path

df = pd.read_csv(path.join(model_path, 'losses.csv'), sep=',')
df['ID'] = df.index

ax1 = df.plot(kind='scatter', x='ID', y='KLDiv', label='KLDiv', c='r')
df.plot(kind='scatter', x='ID', y='val_KLDiv',
        label='val_KLDiv', c='salmon',  ax=ax1)
df.plot(kind='scatter', x='ID', y='nll', label='nll',
        c='b', ax=ax1)
df.plot(kind='scatter', x='ID', y='val_nll', label='val_nll',
        c='skyblue', ax=ax1)

plt.xlabel('epoch')
plt.ylabel('losses')
plt.title('Loss components over epochs')
plt.show()
