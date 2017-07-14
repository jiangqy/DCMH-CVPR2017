import tensorflow as tf
import scipy.misc
import numpy as np
import scipy.io
LAYER1_NODE = 8192

def txt_net_strucuture(text_input, dimy, bit):

	W_fc8 = tf.random_normal([1, dimy, 1, LAYER1_NODE], stddev=1.0) * 0.01
	b_fc8 = tf.random_normal([1, LAYER1_NODE], stddev=1.0) * 0.01
	fc1W = tf.Variable(W_fc8)
	fc1b = tf.Variable(b_fc8)

	# ### debugging .......................
	# wb = scipy.io.loadmat('data/wb-text.mat')
	# fc1W = tf.Variable(wb['w1'] * 0.01)
	# fc1b = tf.Variable(wb['b1'] * 0.01)

	conv1 = tf.nn.conv2d(text_input, fc1W, strides=[1, 1, 1, 1], padding='VALID')

	layer1 = tf.nn.relu(tf.nn.bias_add(conv1, tf.squeeze(fc1b)))

	W_fc2 = tf.random_normal([1, 1, LAYER1_NODE, bit], stddev=1.0) * 0.01
	b_fc2 = tf.random_normal([1, bit], stddev=1.0) * 0.01
	fc2W = tf.Variable(W_fc2)
	fc2b = tf.Variable(b_fc2)

	# ### debugging .......................
	# fc2W = tf.Variable(wb['w2'] * 0.01)
	# fc2b = tf.Variable(wb['b2'] * 0.01)
	conv2 = tf.nn.conv2d(layer1, fc2W, strides=[1, 1, 1, 1], padding='VALID')
	output_g = tf.transpose(tf.squeeze(tf.nn.bias_add(conv2, tf.squeeze(fc2b))))
	return output_g
