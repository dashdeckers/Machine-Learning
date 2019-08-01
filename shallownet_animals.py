from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from MachineLearning.preprocessors import ImageToArrayPreprocessor, SimplePreprocessor
from MachineLearning.datasets.dataset_loaders import SimpleDatasetLoader
from MachineLearning.conv_nets import ShallowNet
from keras.optimizers import SGD
from imutils import paths
import matplotlib.pyplot as plt
import numpy as np
import argparse

# Require the user to provide a path to the dataset
ap = argparse.ArgumentParser()
ap.add_argument('-d', '--dataset', required=True, help='path to input dataset')
args = vars(ap.parse_args())

# Get the list of images from the dataset location
print('[INFO] Loading images')
image_paths = list(paths.list_images(args['dataset']))

# Initialize the preprocessors
sp = SimplePreprocessor(32, 32) # This resizes the images
iap = ImageToArrayPreprocessor() # This converts them to arrays

# Load the dataset from disk, apply preprocessing in sequential order
# and then scale the pixel input to [0, 1]
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255.0

# Split the data set (75% training data) and convert labels to vectors
(trainX, testX, trainY, testY) = train_test_split(data, labels,
                                                  test_size=0.25,
                                                  random_state=42)
trainY = LabelBinarizer().fit_transform(trainY)
testY = LabelBinarizer().fit_transform(testY)

# Initialize the optimizer and the model
print('[INFO] Compiling model')
opt = SGD(lr=0.005)
model = ShallowNet.build(width=32, height=32, depth=3, classes=3)
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

# Train the network
print('[INFO] Training network')
H = model.fit(trainX, trainY,
              validation_data=(testX, testY),
              batch_size=32,
              epochs=100,
              verbose=True)

# Save the model
model.save('saved_model.h5')

# Evaluate the network
print('[INFO] Evaluating the network')
predictions = model.predict(testX, batch_size=32)
print(classification_report(testY.argmax(axis=1),
                            predictions.argmax(axis=1),
                            target_names=['cat', 'dog', 'panda']))

# Plot the training loss and accuracy
plt.style.use('ggplot')
plt.figure()
plt.plot(np.arange(0, 100), H.history['loss'], label='train_loss')
plt.plot(np.arange(0, 100), H.history['val_loss'], label='val_loss')
plt.plot(np.arange(0, 100), H.history['acc'], label='train_acc')
plt.plot(np.arange(0, 100), H.history['val_acc'], label='val_acc')
plt.title('Training Loss and Accuracy')
plt.xlabel('Epoch #')
plt.ylabel('Loss/Accuracy')
plt.legend()
plt.show()

