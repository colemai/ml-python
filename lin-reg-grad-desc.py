#!/usr/bin/env python3
"""
Author: Ian Coleman
Purpose: Create a template/example of linear regression optimised
by gradient descent
Input: 
Output:
"""

from sys import argv
import pandas as pd
import pylab
import pdb

def optimise_cost_func(x, y, current_B0 = 0, current_B1 = 0, iters=1000, alpha=0.001):
	"""
	Purpose: Optimise cost function of lin reg (one input var) using gradient descent
	Input: x: array of floats, y: array of floats same length as x
	Output: Optimised B0, B1, last cost
	"""
	assert len(x) == len(y), "Optimising error: x and y have different lengths"
	
	m = len(y)
	for i in range(iters):
		pred_y = current_B0 + (current_B1 * x)
		cost = sum([data **2 for data in (y - pred_y)])/m
		B0_gradient = -(2/m) * sum(x * (y - pred_y)) 
		B1_gradient = -(2/m) * sum(y - pred_y)
		current_B0 = current_B0 - (alpha * B0_gradient) 
		current_B1 = current_B1 - (alpha * B1_gradient)
	return current_B0, current_B1, cost


if __name__ == "__main__":
	#Read in csv
	df = pd.read_csv(argv[1])

	x = df['perCapIncome']
	y = df['deathRate']
	
	#plot x vs y
	# pylab.plot(x, y, 'o')
	# pylab.show()

	outputs = optimise_cost_func(x, y)
	print(outputs)

#TODO 1. Neaten 2. Expand plotting 3. Regularisation 4. Verify Results 5. Adapt for multiple X 
	



