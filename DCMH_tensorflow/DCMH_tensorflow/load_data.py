import h5py
import numpy as np

def loading_data(path):
	file = h5py.File(path)
	images = file['images'][:].transpose(0,3,2,1)
	labels = file['LAll'][:].transpose(1,0)
	tags = file['YAll'][:].transpose(1,0)

	file.close()
	return images, tags, labels

if __name__=='__main__':
	path = 'data/FLICKR-25K.mat'
	images, tags, labels = loading_data(path)
	print images.shape
	print tags.shape
	print labels.shape