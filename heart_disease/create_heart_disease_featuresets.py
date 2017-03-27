'''
Create X_train, y_train, X_test, y_test dataset for the neural network

'''

import sys
reload(sys)
sys.setdefaultencoding('utf8')

import io

def create_vectors():
	featureset = []
	with io.open('processed_cleveland_data.txt') as f:
		contents = f.readlines()
		for l in contents[:]:
			features = np.zeros(len())