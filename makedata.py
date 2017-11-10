import numpy as np
from math import floor

# This ensures that we get the same random dataset everytime
np.random.seed(10)

"""
	Returns the following data structure:

		Features: 
			R			G			B
			193		129		27
			13		79		208
	
		Each row is a random color
		
		Labels:
			1
			0
		
		Should the text be white or black

"""
def make_data(n_examples):
	X = np.random.randint(0, 255, (n_examples, 3))
	Y = []
	for i in range(X.shape[0]):
		Y.append(yiq(X[i, :]))
	return (X, np.asarray(Y).reshape(n_examples, 1))

def make_split_data(n_examples = 1000, ratio = .9):
	X, Y = make_data(n_examples)
	n_train = floor(n_examples * ratio)
	return (X[:n_train,:], X[n_train:,:], Y[:n_train], Y[n_train:])


"""
	The function we are trying to replicate
	Returns 0 if the text should be black and 1 if the text should be white
"""
def yiq(color):
	yiq = (color[0] * 299 + color[1] * 587 + color[2] * 114) / 1000
	return (0 if yiq >= 128 else 1)