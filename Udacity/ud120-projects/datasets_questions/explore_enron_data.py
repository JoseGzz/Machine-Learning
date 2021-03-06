#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle

enron_data = pickle.load(open("../final_project/final_project_dataset.pkl", "r"))

# no. of data points
data_points = len(enron_data)
print data_points

# no. of features per person
print len(enron_data.values()[0])

# features
print enron_data.values()[0].keys()

# no. of persons of interest
print len([name for name in enron_data if enron_data[name]['poi'] == True])

# stock of James Prentice
print enron_data['PRENTICE JAMES']['total_stock_value']

# no. of emails from this person to POI
print enron_data['COLWELL WESLEY']['from_this_person_to_poi']

# value of stock options exercised by Jeffrey K Skilling
print enron_data['SKILLING JEFFREY K']['exercised_stock_options']

print enron_data.values()[0]

# How many folks in this dataset have a quantified salary?
print len([name for name in enron_data if enron_data[name]['salary'] != 'NaN'])

# NaN email addresses
print len([name for name in enron_data if enron_data[name]['email_address'] != 'NaN'])

# NaN in total payments
nan_payments = len([name for name in enron_data if enron_data[name]['total_payments'] == 'NaN'])

print nan_payments

print (nan_payments * 100) / data_points





















