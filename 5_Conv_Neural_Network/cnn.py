'''

cnn example

'''
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 10 classes, 0-9
# 0 = [1,0,0,0,0,0,0,0,0,0] => one hot encoding

n_classes = 10
batch_size = 128

# place holder function defines the structure of the x varaible. Will throw an error if we try to fit a matrice that is not of dimension - 784 flatten
# The None dimension is a placeholder for the batch size. At runtime, TensorFlow will accept any batch size greater than 0.
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')

def maxpool2d(x):
	return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')

def convolutional_neural_network(x):
	
	weights = {'W_conv1': tf.Variable(tf.random_normal([5,5,1,32])),
			   'W_conv2': tf.Variable(tf.random_normal([5,5,32,64])),
			   'W_fc': tf.Variable(tf.random_normal([7*7*64,1024])),
			   'out': tf.Variable(tf.random_normal([1024, n_classes]))
			  }

	biases = {'b_conv1': tf.Variable(tf.random_normal([32])),
			   'b_conv2': tf.Variable(tf.random_normal([64])),
			   'b_fc': tf.Variable(tf.random_normal([1024])),
			   'out': tf.Variable(tf.random_normal([n_classes]))
			  }

	x = tf.reshape(x, reshape=[-1, 28, 28, 1])

	conv1 =  tf.nn.relu( conv2d(x,weights['W_conv1']) + b_conv1 )
	conv1 =  maxpool2d(conv1)

	conv2 =  tf.nn.relu( conv2d(conv1,weights['W_conv2']) + b_conv2 )
	conv2 = maxpool2d(conv2)

	fc = tf.reshape(conv2, [-1, 7*7*64])
	fc = tf.nn.relu(tf.matmul(fc, weights['W_fc']) + biases['b_fc'])

	fc = tf.nn.dropout(fc, keep_rate)

	output = tf.matmul(fc, weights['out']) + biases['out']

	return output

def train_neural_network(x):
	prediction = convolutional_neural_network(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))

	# Default learning_rate = 0.001
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	# cycles of feed forward + backprop
	hm_epochs = 10

	with tf.Session() as sess:
		# Iniitaliize all variables (weights, biases) to be tensors full of zeros
		sess.run(tf.global_variables_initializer())

		for epoch in range(hm_epochs):
			epoch_loss = 0
			# _ is a variable that we don't care about in the for loop
			for _ in range(int(mnist.train.num_examples/batch_size)):
				epoch_x, epoch_y = mnist.train.next_batch(batch_size)
				_, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})

				epoch_loss += c

			print 'Epoch', epoch, ' completed out of', hm_epochs, 'loss:', epoch_loss

		# tf.argmax gives us the index of the highest entry in a tensor along axis=1
		# tf.equal will give us a list of booleans [True, True, False, False]
		correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y,1))

		# tf.cast will convert the True to 1 and False to 0 [1,1,0,0]
		# reduce_mean takes the average over these sums and returns 0.5 for [1,1,0,0]
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

		print 'Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels})

train_neural_network(x)


