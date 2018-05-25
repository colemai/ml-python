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
