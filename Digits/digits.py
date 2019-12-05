import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import pandas as pd
import time
from sklearn.model_selection import train_test_split

def show_digit(digit):
	''' Show the digit as an image
	'''
	assert digit.ndim == 2, (digit, digit.shape)
	assert digit.shape[0] == digit.shape[1], (digit, digit.shape)
	plt.imshow(digit, cmap='Greys')
	plt.show()

def one_hot_encode_labels(y, n_possible_vals=10):
	''' Returns a matrix of one-hot encoded vectors for the given array
	of values. Assumes that the labels in y correspond to the indices in
	[0, n_possible_vals) that should be set to 1 in the resulting matrix.

		label_matrix.shape == (n_possible_vals, n_labels)

	'''
	assert y.ndim == 1, (y, y.shape)

	N = len(y)
	# Create a matrix of zeros
	label_matrix = np.zeros(shape=(n_possible_vals, N))
	# Entries with (row, col) == (label, c_index) should equal 1
	label_matrix[y, range(N)] = 1

	return label_matrix

def reshape_image_data(x, shape='channels_first'):
	''' Turns a 3D image matrix into a 2D matrix in which the images have
	been converted to column vectors. The resulting shape will be: 

		x.shape == (img_dim1 * img_dim2, n_imgs).

	'''
	assert x.ndim == 3, (x, x.shape)
	assert shape in ['channels_first', 'channels_last'], shape

	if shape == 'channels_first':
		n, _, _ = x.shape

	if shape == 'channels_last':
		_, _, n = x.shape

	return x.reshape(n, -1, ).T

def center_data(x):
	''' Compute the centered dataset by subtracting the column-wise mean
	from each columm vector in the matrix. 

	Assuming each column represents an image.
	'''
	assert x.ndim == 2, (x, x.shape)

	# Assuming the format (img_dim1 * img_dim2, n_imgs)
	return x - x.mean(axis=1).reshape(-1, 1)

def compute_first_m_PCs_of_x(x, m):
	''' Compute the first m principle components of the dataset matrix x.

		U.shape == (img_dim1 * img_dim2, m)

	'''
	# Compute covariance matrix
	C = np.cov(x)
	# Get the SVD of C
	U, S, V = np.linalg.svd(C)
	# Return the first m columns of U
	return U[:, :m]

def compute_MSE(V, F, W):
	''' Compute the MSE

	Still working on this.

	'''
	diff = (V - np.dot(W, F)) # (10, 60000)
	normed = np.sqrt(np.square(diff).sum(axis=0)) # (60000, )
	return np.square(normed).sum() / len(normed)

if __name__ == '__main__':
	t0 = time.time()

	# Step 0: Load and preprocess Data
	#(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

	f = open("mfeat-pix.txt", 'r')
	X = f.readlines()
	Y = []

	#Probably horribly inefficient, perhaps Travis knows how to do this with Python magic
	#Create an array as [0,0,0, ... ,1,1,1, ..., 2,2,2,2..., 9,9,9]
	for i in range(0,10):
	    for j in range(0,200):
	        Y.append(i)
	df = pd.DataFrame(X)
	x_train, x_test, y_train, y_test = train_test_split(df, Y, test_size=0.1)
	x_centered = center_data(reshape_image_data(x_train))
	print(f'Loaded and preprocessed data ({time.time() - t0})')

	# Step 1: PCA
	m = 30
	Um = compute_first_m_PCs_of_x(x_centered, m)
	F = np.dot(Um.T, x_centered)
	print(f'Computed PCA feature vectors ({time.time() - t0})')

	# Step 2: One-Hot Encode Labels
	V = one_hot_encode_labels(y_train)
	print(f'One-Hot encoded labels ({time.time() - t0})')

	# Step 3: Compute LG Classifier
	W = np.dot(np.dot(np.linalg.inv(np.dot(F, F.T)), F), V.T).T
	print(f'Computed linear regression weight matrix ({time.time() - t0})')

	# Step 4: Compute the Error
	MSE_train = compute_MSE(V, F, W)
	print(f'Computed the Error ({time.time() - t0})')
	print(MSE_train)
