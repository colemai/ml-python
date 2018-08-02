#!/usr/bin/env python3
"""
Author: Ian Coleman
Purpose: Linear regression implementation with tensorflow
"""

import pdb
from sys import argv
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#for ipython:
# %matplotlib inline

from numpy import genfromtxt
from sklearn.datasets import load_boston

def read_dataset(filePath, delimiter=','):
	"""
	Using Numpy, read text from csv/tsv into a Numpy array
	"""
	return genfromtxt(filePath, delimiter=delimiter)

def read_boston_data():
	""" specifically grab data from the boston db imported above"""
	boston = load_boston()
	features = np.array(boston.data)
	labels = np.array(boston.target)
	return features, labels

def feature_normalise(dataset):
	"""regularise by mean and standard deviation"""
	mu = np.mean(dataset, axis=0)
	sigma = np.std(dataset, axis=0)
	return (dataset - mu)/sigma

def append_bias_reshape(features, labels):
	n_training_samples = features.shape[0]
	n_dim = features.shape[1]
	f = np.reshape(np.c_[np.ones(n_training_samples),features],[n_training_samples,n_dim + 1])
	l = np.reshape(labels,[n_training_samples,1])
	return f,l

if __name__ == "__main__":
	features,labels = read_boston_data()
	normalised_features = feature_normalise(features)
	f,l = append_bias_reshape(normalised_features, labels)
	n_dim = f.shape[1]
	pdb.set_trace()

	rnd_indices = np.random.rand(len(f)) < 0.80

	train_x = f[rnd_indices]
	train_y = l[rnd_indices]
	test_x = f[~rnd_indices]
	test_y = l[~rnd_indices]

	#Add the data to TF's own data structures
	learning_rate = 0.01
	training_epochs = 1000
	cost_history = np.empty(shape=[1],dtype=float)

	X = tf.placeholder(tf.float32,[None,n_dim])
	Y = tf.placeholder(tf.float32,[None,1])
	W = tf.Variable(tf.ones([n_dim,1]))

	init = tf.initialize_all_variables()

	y_ = tf.matmul(X, W) #multiply features matrix to weights matrix
	cost = tf.reduce_mean(tf.square(y_ - Y)) #mse
	training_step = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) #grad desc optimisation

	#Run training
	sess = tf.Session()
	sess.run(init)

	for epoch in range(training_epochs):
	    sess.run(training_step,feed_dict={X:train_x,Y:train_y})
	    cost_history = np.append(cost_history,sess.run(cost,feed_dict={X: train_x,Y: train_y}))


	#Plot Cost

	plt.plot(range(len(cost_history)),cost_history)
	plt.axis([0,training_epochs,0,np.max(cost_history)])
	plt.show()


	# preds and mse
	pred_y = sess.run(y_, feed_dict={X: test_x})
	mse = tf.reduce_mean(tf.square(pred_y - test_y))
	print("MSE: %.4f" % sess.run(mse)) 

	fig, ax = plt.subplots()
	ax.scatter(test_y, pred_y)
	ax.plot([test_y.min(), test_y.max()], [test_y.min(), test_y.max()], 'k--', lw=3)
	ax.set_xlabel('Measured')
	ax.set_ylabel('Predicted')
	plt.show()