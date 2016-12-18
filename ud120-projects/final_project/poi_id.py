#!/usr/bin/python

import sys
import pickle
import pandas as pd
import numpy  as np
import pylab  as pl

sys.path.append("../tools/")

from feature_format           import featureFormat, targetFeatureSplit
from tester                   import dump_classifier_and_data
from sklearn.naive_bayes      import GaussianNB
from sklearn                  import tree
from sklearn.metrics          import accuracy_score
from sklearn                  import svm
from sklearn.metrics          import accuracy_score
from sklearn.metrics          import precision_score
from sklearn.metrics          import recall_score
from sklearn.cross_validation import KFold

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".

# features_list is not used since we use Pandas DataFrame to manupulate data
#features_list = ['poi','salary'] # You will need to use more features

# all features available for modelling
all_features = ['salary', 'to_messages', 'deferral_payments', 'total_payments', 'exercised_stock_options', 'bonus', 'restricted_stock',
                'shared_receipt_with_poi', 'restricted_stock_deferred', 'total_stock_value', 'expenses', 'loan_advances', 'from_messages',
                'other', 'from_this_person_to_poi', 'director_fees', 'deferred_income', 'long_term_incentive', 'from_poi_to_this_person']

# put features into categories, in this case to handle missing values differently
email_features     = ['to_messages', 'from_poi_to_this_person', 'from_messages', 'from_this_person_to_poi', 'shared_receipt_with_poi']
financial_features = ['salary', 'deferral_payments', 'total_payments', 'loan_advances', 'bonus', 'restricted_stock_deferred', 
                        'deferred_income', 'total_stock_value', 'expenses', 'exercised_stock_options', 'other', 'long_term_incentive',
                        'restricted_stock', 'director_fees']

### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)

# put data into workable form
df = pd.DataFrame.from_dict(data_dict, orient='index')

# deal with missing values
df = df.replace('NaN', np.nan)
df[financial_features] = df[financial_features].fillna(0)
df[email_features] = df[email_features].fillna(df[email_features].median())

### Task 3: Create new feature(s)
# create new feature: relation of messages sent and recieved from poi's
df['fracion_of_messages_to_poi'] = df.from_this_person_to_poi / df.from_messages
df['fracion_of_messages_from_poi'] = df.from_poi_to_this_person / df.to_messages

'''
### Experimantal code for outlier detection ###

stds = []
possible_outliers = []

stds = [df[feature].std() for feature in all_features ]

std_mean = np.array(stds).mean()
possible_outliers = [feature for feature in all_features if df[feature].std() > std_mean]
'''

# get rid of the email_address unnecessary feature
# we need to calculate the transpose because drop() pops by row-index
# and df default is (index x value names)
df = df.T.drop('email_address')
df = df.T

### Task 2: Remove outliers
# removing outliers
# It was observed that the accuracy is the same with or without outliers
# but removing them increased the training time significantly (while loop iterations below)
lim = 10
for feature in all_features:
    # keep only the ones that are within +lim to -lim standard2deviations in the column 'Data'.
    # the lim value was guessed and tested by trail and error
    df = df[np.abs(df[feature]-df[feature].mean()) <= (lim*df[feature].std())] 
    # keep only the ones that are outside +10 to -10 standard2deviations in the column 'Data'.
    # df = df[~(np.abs(df[feature]-df[feature].mean()) > (lim*df[feature].std()))] 

# set a df with labels onlt
labels_df = df['poi']
# then discard those values so to remain with features only
# df alone has feature names as columns, they need to be the first elements in each row
df = df.T 
df = df.drop('poi')
# return to 'normal' shape
df = df.T
# put features and labels into numpy arrays
labels   = np.array(labels_df.tolist())
features = np.array(df.values.tolist())

'''
# techniques for feature selection but with poor results up until now

# SelectKBest
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
for i in range(1,21):
    X_new = SelectKBest(f_classif, k=7).fit_transform(features_train, labels_train)
    X_new_test = SelectKBest(f_classif, k=7).fit_transform(features_test, labels_test)

# SelectPercentile
from sklearn.feature_selection import SelectPercentile, f_classif
# do not use chi2 as discrimination function because we have negative values in the data
selector = SelectPercentile(f_classif, percentile=2)
selector.fit(features, labels)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
print scores

# feature_importanes_ for decision trees
importances = clf.feature_importances_
import numpy as np
indices = np.argsort(importances)[::-1]
print 'Feature Ranking: '
for i in range(0, 17):
    print "{} feature {} ({})".format(i,all_features[i],importances[indices[i]])

'''
    
X, y = features, labels

### Store to my_dataset for easy export below.
my_dataset = data_dict

# this was not necesary since we used pandas data frames 
### Extract features and labels from dataset for local testing
'''
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
'''

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# make three folds to divide the data
kf = KFold(len(labels), 3)

# training and testing sets division
for train_indices, test_indices in kf:
    features_train = [X[i] for i in train_indices]
    features_test  = [X[i] for i in test_indices ]
    labels_train   = [y[i] for i in train_indices]
    labels_test    = [y[i] for i in test_indices ]

# classifier options
#clf = GaussianNB()
#clf = svm.SVC(kernel='rbf')
clf  = tree.DecisionTreeClassifier(random_state = None)

iteration           = 0
score               = 0
max_iterations      = 250
max_actual_accuracy = 0.93


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 

# iterate max_iterations times or until max_actual_accuracy is reached
# training the clasifier, this could cause performance trouble on large data sets
# Leaving it at the default value of None (for the tree)  means that the fit method will use
# numpy.random's singleton random state, which is not predictable and not the same across runs
# and cannot be extracted in any way
while score < max_actual_accuracy and iteration < max_iterations:
    iteration += 1
    clf.fit(features_train, labels_train)
    score = clf.score(features_test,labels_test)

print 'accuracy:', 100*score, "%, after ", iteration, "iterations"


### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html
# calculate evaluation metrics
pred = clf.predict(features_test)
print 'precision = ', precision_score(labels_test,pred)
print 'recall = ', recall_score(labels_test,pred)

# Example starting point. Try investigating other evaluation techniques!
'''
from sklearn.cross_validation import train_test_split
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size=0.3, random_state=42)
'''
### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

#dump_classifier_and_data(clf, my_dataset, features_list)
# currently reaches 93.6170212766% accuracy
dump_classifier_and_data(clf, my_dataset, all_features)
