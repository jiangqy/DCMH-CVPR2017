import tensorflow as tf
import scipy.misc
import numpy as np
import scipy.io

MODEL_DIR = './data/imagenet-vgg-f.mat'
def img_net_strucuture(input_image, bit):
	data = scipy.io.loadmat(MODEL_DIR)
	layers = (
		'conv1', 'relu1', 'norm1', 'pool1','conv2', 'relu2', 'norm2', 'pool2','conv3', 'relu3', 'conv4', 'relu4', 'conv5',
		'relu5', 'pool5','fc6', 'relu6', 'fc7', 'relu7','fc8')
	weights = data['layers'][0]
	mean = data['normalization'][0][0][0]
	net = {}
	ops = []
	current = tf.convert_to_tensor(input_image,dtype='float32')
	for i, name in enumerate(layers[:-1]):
		if name.startswith('conv'):
			kernels, bias = weights[i][0][0][0][0]
			# matconvnet: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			#kernels = np.transpose(kernels, (1, 0, 2, 3))

			bias = bias.reshape(-1)
			pad = weights[i][0][0][1]
			stride = weights[i][0][0][4]
			current = _conv_layer(current,kernels,bias,pad,stride,i,ops,net)
		elif name.startswith('relu'):
			current = tf.nn.relu(current)
		elif name.startswith('pool'):
			stride = weights[i][0][0][1]
			pad = weights[i][0][0][2]
			area = weights[i][0][0][5]
			current = _pool_layer(current,stride,pad,area)
		elif name.startswith('fc'):
			kernels, bias = weights[i][0][0][0][0]
			# matconvnet: weights are [width, height, in_channels, out_channels]
			# tensorflow: weights are [height, width, in_channels, out_channels]
			#kernels = np.transpose(kernels, (1, 0, 2, 3))

			bias = bias.reshape(-1)
			current = _full_conv(current,kernels,bias,i,ops,net)
		elif name.startswith('norm'):
			current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
		net[name] = current
	W_fc8 = tf.random_normal([4096, bit], stddev=1.0) * 0.01

	b_fc8 = tf.random_normal([bit],stddev = 1.0) * 0.01
	w = tf.Variable(W_fc8, name='w' + str(20))
	b = tf.Variable(b_fc8, name='bias' + str(20))

	### debugging...................
	# layer8 = scipy.io.loadmat('data/wb-image.mat')
	#
	# w = tf.Variable(np.squeeze(layer8['w8']) * 0.01, name='w' + str(20))
	# b = tf.Variable(np.squeeze(layer8['b8']) * 0.01, name='bias' + str(20))

	ops.append(w)
	ops.append(b)
    
	fc8 = tf.matmul(tf.squeeze(current),w) + b
	net['weigh21'] = w
	net['b21'] = b
	net[layers[-1]] = fc8
	return net, mean

def _conv_layer(input, weights, bias,pad,stride,i,ops,net):
	pad = pad[0]
	stride= stride[0]
	input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
	w = tf.Variable(weights,name='w'+str(i),dtype='float32')
	b = tf.Variable(bias,name='bias'+str(i),dtype='float32')
	ops.append(w)
	ops.append(b)
	net['weights' + str(i)] = w
	net['b' + str(i)] = b
	conv = tf.nn.conv2d(input, w, strides=[1,stride[0],stride[1],1],padding='VALID',name='conv'+str(i))
	return tf.nn.bias_add(conv, b,name='add'+str(i))

def _full_conv(input, weights, bias,i,ops,net):
	w = tf.Variable(weights, name='w' + str(i),dtype='float32')
	b = tf.Variable(bias, name='bias' + str(i),dtype='float32')
	ops.append(w)
	ops.append(b)
	net['weights' + str(i)] = w
	net['b' + str(i)] = b
	conv = tf.nn.conv2d(input, w,strides=[1,1,1,1],padding='VALID',name='fc'+str(i))
	return tf.nn.bias_add(conv, b,name='add'+str(i))

def _pool_layer(input,stride,pad,area):
	pad = pad[0]
	area = area[0]
	stride = stride[0]
	input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
	return tf.nn.max_pool(input, ksize=[1, area[0], area[1], 1], strides=[1,stride[0],stride[1],1],padding='VALID')

def preprocess(image, mean_pixel):
	return image - mean_pixel


def unprocess(image, mean_pixel):
	return image + mean_pixel

def get_meanpix(data_path):
	data = scipy.io.loadmat(data_path)
	mean = data['normalization'][0][0][0]
	return mean
