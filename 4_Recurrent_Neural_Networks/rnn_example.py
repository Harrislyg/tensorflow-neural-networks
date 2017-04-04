'''

RNN Example with MINST dataset

'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib import rnn

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

hm_epochs = 3
n_classes = 10
batch_size = 128

# 28 chunks of 28 pixels
chunk_size = 28
n_chunks = 28
rnn_size = 128

# place holder function defines the structure of the x varaible. Will throw an error if we try to ft someonething that is not of dimension - 784 flatten
x = tf.placeholder('float', [None, n_chunks, chunk_size])
y = tf.placeholder('float')

def recurrent_neural_network_model(x):
	
	layer = {'weights': tf.Variable(tf.random_normal([rnn_size, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

	# reshape the matrix from (3,1,2) to (2,1,3). Converts each row into an array by swapping the 0 dimension with the first dimension. (0 dimension refers to the row, 1 dimension refers to the column)
	x = tf.transpose(x, [1,0,2])

	# reshape the matrix to a matrice with column length = chunk size. This falttens the array by one dimension
	x = tf.reshape(x, [-1, chunk_size])

	# split the matrix along dimension 0 into n_chunks and return x
	x = tf.split(x, n_chunks, 0)

	lstm_cell = rnn.BasicLSTMCell(rnn_size)
	outputs, states = rnn.static_rnn(lstm_cell, x, dtype=tf.float32)

	output = tf.add(tf.matmul(outputs[-1], layer['weights']), layer['biases'])

	return output

def train_neural_network(x):
	prediction = recurrent_neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# Default learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)



	with tf.Session() as sess:
		# Iniitaliize all variables (weights, biases) to be tensors full of zeros
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			# _ is a variable that we don't care about in the for loop
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				epoch_x = epoch_x.reshape((batch_size, n_chunks, chunk_size))
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

				epoch_loss += c

			print 'Epoch', epoch, ' completed out of', hm_epochs, 'loss:', epoch_loss

		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))

		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print 'Accuracy: ', accuracy.eval({x:mnist.test.images.reshape((-1, n_chunks, chunk_size)), y:mnist.test.labels})

train_neural_network(x)
