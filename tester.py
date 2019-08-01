from MachineLearning.preprocessors import ImageToArrayPreprocessor, SimplePreprocessor
from MachineLearning.datasets.dataset_loaders import SimpleDatasetLoader
from keras.models import load_model
from imutils import paths
import numpy as np
import argparse

# Get the path to the model from the user
ap = argparse.ArgumentParser()
ap.add_argument('-m', '--model', required=True, help='path to the model to load')
args = vars(ap.parse_args())

# Load the model
model = load_model(args['model'])

# Initialize the preprocessors and load the test images
sp = SimplePreprocessor(32, 32)
iap = ImageToArrayPreprocessor()
sdl = SimpleDatasetLoader(preprocessors=[sp, iap])
image_paths = paths.list_images('Test_Images')
(data, labels) = sdl.load(image_paths, verbose=500)
data = data.astype('float') / 255.0

# Make the predictions
for data, label in zip(data, labels):
	data = np.reshape(data, [1] + list(data.shape))
	prediction = model.predict(data)

	string = ''
	labels = ['Cat', 'Dog', 'Panda']
	for idx, p in enumerate(prediction[0]):
		string += f'{labels[idx]}: {round(p * 100, 0)}% '

	print(f'{string} --- {label}')