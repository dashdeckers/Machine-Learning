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

def load_data(filename='mfeat-pix.txt', preprocess=True):

	with open(filename, 'r') as datafile:
		# Load the transposed datafile to get each image in a col vector
		data = np.loadtxt(datafile).T # shape == (240, 2000)

		# Create an array of labels (each 200 elements is a digit)
		labels = np.zeros(data.shape[1], dtype=np.int)
		for digit in range(10):
			labels[digit * 200: (digit+1) * 200] = digit # shape == (2000,)

		if preprocess:
			# Normalize the data to range [0,1] (assuming range [0,6])
			data /= 6
			# Center the data by subtracting the mean (not really helping)
			# data -= data.mean(axis=1).reshape(-1, 1)

		# Split the data into train and test by first determining the indices
		even = np.array([np.arange(i*100, (i+1)*100) for i in range(0,20,2)])
		odd  = np.array([np.arange(i*100, (i+1)*100) for i in range(1,20,2)])
		even = even.reshape(-1) # reshape to get a single,
		odd  = odd.reshape(-1)  # long array of indices

		# And then selecting via the array of indices
		x_train, y_train = data[:, even], labels[even]
		x_test,  y_test  = data[:, odd],  labels[odd]

		return (x_train, y_train), (x_test, y_test)

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
	(x_train, y_train), (x_test, y_test) = load_data()
	print(f'Loaded and preprocessed data ({time.time() - t0})')

	for m in [2, 20, 30, 40, 50, 100, 200]:
		print(f'Setting m={m}:')

		# Step 1: PCA
		Um = compute_first_m_PCs_of_x(x_train, m)
		F = np.dot(Um.T, x_train)
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
		print(f'\tError (for m={m}): {np.log10(MSE_train)}\n')
