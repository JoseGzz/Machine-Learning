#!/usr/bin/python

import pickle
import sys
import matplotlib.pyplot
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit


### read in data dictionary, convert to numpy array
data_dict = pickle.load( open("../final_project/final_project_dataset.pkl", "r") )

data_dict.pop('TOTAL', 0)

features = ["salary", "bonus"]
data = featureFormat(data_dict, features)

value_at_index = data_dict.values()[69]

for field, possible_values in data_dict.iteritems():
	# check salaries
	if possible_values.values()[0] == 26704229.0:
    		print field


for field, possible_values in data_dict.iteritems():
	# check salaries
	if possible_values.values()[0] >= 1000000 and possible_values.values()[5] >= 5000000:
    		#if possible_values.values()[0] != 'NaN':
		print field  


if 'NaN' > 1000000:
	print('yes')

### your code below

count = 1

for point in data:
	salary = point[0]
	bonus  = point[1]
	if bonus == 97343619.0:
		print count
	matplotlib.pyplot.scatter(salary, bonus)
	count += 1

matplotlib.pyplot.xlabel("salary")
matplotlib.pyplot.ylabel("bonus")
# matplotlib.pyplot.show() 
# 26704229.0 97343619.0

