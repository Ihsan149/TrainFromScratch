#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 12:51:46 2017

@author: 09959295800
"""

CAFFE_ROOT = '/home/09959295800/git/caffe/'
IMAGE_ROOT = '/home/09959295800/SERPRO/ProjetoEstrategico/ReconhecimentoImagem/CompCars/'

# set up Python environment: numpy for numerical routines, and matplotlib for plotting
import numpy as np
import matplotlib.pyplot as plt
import caffe
from pyutils.timer import Timer

# set display defaults
plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

#Set Caffe to CPU mode and load the net from disk.
caffe.set_mode_cpu()
model_def = CAFFE_ROOT + 'models/bvlc_googlenet_cars/deploy.prototxt'
model_weights = CAFFE_ROOT + 'models/bvlc_googlenet_cars/googlenet_finetune_web_car_iter_10000.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

# load the mean ImageNet image (as distributed with Caffe) for subtraction
mu = np.load(CAFFE_ROOT + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

# set the size of the input (we can skip this if we're happy
#  with the default; we can also change it later, e.g., for different batch sizes)
net.blobs['data'].reshape(50,        # batch size
                          3,         # 3-channel (BGR) images
                          224, 224)  # image size is 227x227

image_path = 'data/image/67/1698/2005/'
image_name = '0a884f5a90267d.jpg'
image = caffe.io.load_image(IMAGE_ROOT + image_path + image_name)
transformed_image = transformer.preprocess('data', image)
#plt.imshow(image)

# copy the image data into the memory allocated for the net
net.blobs['data'].data[...] = transformed_image
timer = Timer()
timer.tic()
### perform classification
output = net.forward()
timer.toc()
print ('Elapsed time {:.3f} s.').format(timer.total_time)
output_prob = output['prob'][0]  # the output probability vector for the first image in the batch

print 'predicted class is:', output_prob.argmax()

top_inds = output_prob.argsort()[::-1][:5]  # reverse sort and take five largest items

print '5 top probabilities and labels:'
print zip(output_prob[top_inds], top_inds)
