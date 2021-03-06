#!/usr/bin/python


import numpy as np

def outlierCleaner(predictions, ages, net_worths):
	cleaned_data = []

	# calclate errors in predictions
	errors = predictions - net_worths

	# create tuple
	cleaned_data = zip(ages, net_worths, errors)
	
	# sort from largest to smalles based on erorrs
	cleaned_data = sorted(cleaned_data, key=lambda x: x[2], reverse=True)

	# set limit up to the first 10% of the sorted data
	limit = int(len(net_worths) * 0.1)

	# return the rest 90%
	return list(cleaned_data[limit:])

