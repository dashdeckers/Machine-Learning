import matplotlib.pyplot as plt
import numpy as np
import time

def show_digit(digit, col_vector=False):
	''' Show the digit as an image.

	Usage:
		digit = data[:,0] # get column vector
		digit = digit.reshape(16, 15) # resize into 2D
		show_digit(digit)

		digit = data[:,0]
		show_digit(digit, col_vector=True)
	'''
	if col_vector:
		assert digit.ndim == 1, (digit, digit.shape)
		assert digit.shape == (240,), (digit, digit.shape)
		digit = digit.reshape(16, 15)
	else:
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

def compute_LR_classifier(F, V):
    ''' Compute the Linear Regression classifier according to:

        W' = (F F')^-1 F V'

    '''
    F_Fprime = np.dot(F, F.T)           # F*F'
    F_Fp_inv = np.linalg.inv(F_Fprime)  # (F*F')^-1

    Triple_F = np.dot(F_Fp_inv, F)      # (F*F')^-1 * F
    W_transp = np.dot(Triple_F, V.T)    # (F*F')^-1 * F * V'

    return W_transp.T                   # W'
	# Compute covariance matrix
	C = np.cov(x)
	# Get the SVD of C
	U, S, V = np.linalg.svd(C)
	# Return the first m columns of U
	return U[:, :m]

def compute_MSE(V, F, W):
    ''' Compute the MSE.

    Across all vectors, subtract W*F from V, then find the euclidean norm
    and finally sum the squares and divide by the number of elements.
    (I hope I understood the equation correctly)

    '''
    diff = (V - np.dot(W, F)) # (10, 1000)
    normed = np.sqrt(np.square(diff).sum(axis=0)) # (1000, )
    return np.square(normed).sum() / len(normed) # (1, )

if __name__ == '__main__':
    t0 = time.time()

    # Step 0: Load and preprocess Data
    (x_train, y_train), (x_test, y_test) = preprocess_data(load_data())
    print(f'Loaded and preprocessed data ({time.time() - t0})')

    MSE_train_vals = list()
    MSE_test_vals  = list()
    m_vals = list(range(1, 240)) # [2, 20, 30, 40, 50, 100, 200]

    for m in m_vals:
        print(f'Setting m={m}:')

        # Step 1: PCA
        Um = compute_first_m_PCs_of_x(x_train, m)
        F_train = np.dot(Um.T, x_train)
        F_test  = np.dot(Um.T, x_test)
        print(f'Computed PCA feature vectors ({time.time() - t0})')

        # Step 2: One-Hot Encode Labels
        V_train = one_hot_encode_labels(y_train)
        V_test  = one_hot_encode_labels(y_test)
        print(f'One-Hot encoded labels ({time.time() - t0})')

		# Step 3: Compute LG Classifier
		W = np.dot(np.dot(np.linalg.inv(np.dot(F, F.T)), F), V.T).T
		print(f'Computed linear regression weight matrix ({time.time() - t0})')
        # Step 3: Compute LR Classifier
        W = compute_LR_classifier(F_train, V_train)
        print(f'Computed linear regression weight matrix ({time.time() - t0})')

        # Step 4: Compute the Error
        MSE_train = compute_MSE(V_train, F_train, W)
        MSE_test  = compute_MSE(V_test, F_test, W)
        MSE_train_vals.append(np.log10(MSE_train))
        MSE_test_vals.append(np.log10(MSE_test))
        print(f'Computed the Error ({time.time() - t0})')
        print(f'\tError (for m={m}): {MSE_train_vals[-1]}\n')

    # Step 5: Plot the results
    plt.plot(m_vals, MSE_train_vals, c='blue', label='MSE_train')
    plt.plot(m_vals, MSE_test_vals, c='red', label='MSE_test')
    plt.xlabel('m')
    plt.ylabel('MSE (log10)')
    plt.title('MSE vs chosen m')
    plt.legend()
    plt.show()

    '''
    Results show the correct behaviour of MSE(train) but with worse total
    performance. The error reaches just under -0.6, which is nowhere near
    the -1.5 from the lecture notes. Also no jitter?
    '''
