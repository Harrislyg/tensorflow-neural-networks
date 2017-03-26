'''

input > weight > hidden layer 1 (activation function) > weights > hidden layer 2 (activation function) > weights > output layer

compare output to intended output > cost function (cross entropy)

optimization function (optimizer) > minimize cost (AdamOptimizer....SGD, AdaGrad)

backpropogation 

feed forward + back propagation = epoch

'''
import tensorflow as tf

from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

# 10 classes, 0-9
# 0 = [1,0,0,0,0,0,0,0,0,0] => one hot encoding

n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

n_classes = 10
batch_size = 100

# place holder function defines the structure of the x varaible. Will throw an error if we try to ft someonething that is not of dimension - 784 flatten
x = tf.placeholder('float', [None, 784])
y = tf.placeholder('float')

def neural_network_model(data):
	
	hidden_1_layer = {'weights': tf.Variable(tf.random_normal([784, n_nodes_hl1])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])), 'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])), 'biases': tf.Variable(tf.random_normal([n_classes]))}

	# Model: (input_data * weight) + biases => neurons aer fired even though input_data = 0

	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1) # Activation function

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.add(tf.matmul(l3, output_layer['weights']), output_layer['biases'])

	print 'output: ', output

	print output.shape

	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
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











