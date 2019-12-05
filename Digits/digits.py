import matplotlib.pyplot as plt
import numpy as np
import time

def show_digit(digit):
	''' Show the digit as an image
	'''
	assert digit.ndim == 2, (digit, digit.shape)
	assert digit.shape == (16, 15), (digit, digit.shape)

	plt.imshow(digit, cmap='Greys')
	plt.show()

def one_hot_encode_labels(labels, n_possible_vals=10):
	''' Returns a matrix of one-hot encoded vectors for the given array
	of values. Assumes that the labels in y correspond to the indices in
	[0, n_possible_vals) that should be set to 1 in the resulting matrix.

		label_matrix.shape == (n_possible_vals, n_labels)

	'''
	assert labels.ndim == 1, (labels, labels.shape)

	N = len(labels)
	# Create a matrix of zeros
	label_matrix = np.zeros(shape=(n_possible_vals, N))
	# Entries with (row, col) == (label, c_index) should equal 1
	label_matrix[labels, range(N)] = 1

	return label_matrix

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

	# summer = 0
	# for i in range(2000):
	# 	diff = V[:,i] - np.dot(W, F[:,i])
	# 	norm = np.sqrt( (diff**2).sum() )
	# 	summer += norm
	# summer /= 2000
	# return summer

	# return np.mean( (V - np.dot(W, F))**2, axis=0 )

	normed = np.sqrt(np.square(diff).sum(axis=0)) # (60000, )
	return np.square(normed).sum() / len(normed)

if __name__ == '__main__':
	t0 = time.time()

	# Step 0: Load and preprocess Data
	with open('mfeat-pix.txt', 'r') as datafile:
		# Load the datafile into numpy matrix
		data = np.loadtxt(datafile).T
		# Create an array of labels (each 200 elements is a digit)
		labels = np.zeros(data.shape[1], dtype=np.int)
		for digit in range(10):
			labels[digit * 200: (digit+1) * 200] = digit
		# Center the data
		centered = center_data(data)

	print(f'Loaded and preprocessed data ({time.time() - t0})')

	for m in [2, 20, 30, 40, 50]:
		print(f'Setting m={m}:')

		# Step 1: PCA
		Um = compute_first_m_PCs_of_x(centered, m)
		F = np.dot(Um.T, centered)
		print(f'Computed PCA feature vectors ({time.time() - t0})')

		# Step 2: One-Hot Encode Labels
		V = one_hot_encode_labels(labels)
		print(f'One-Hot encoded labels ({time.time() - t0})')

		# Step 3: Compute LG Classifier
		W = np.dot(np.dot(np.linalg.inv(np.dot(F, F.T)), F), V.T).T
		print(f'Computed linear regression weight matrix ({time.time() - t0})')

		# Step 4: Compute the Error
		MSE_train = compute_MSE(V, F, W)
		print(f'Computed the Error ({time.time() - t0})')
		print(f'\tError (for m={m}): {np.log10(MSE_train)}\n')
