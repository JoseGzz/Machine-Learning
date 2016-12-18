#!/usr/bin/python


"""
    Starter code for the evaluation mini-project.
    Start by copying your trained/tested POI identifier from
    that which you built in the validation mini-project.

    This is the second step toward building your POI identifier!

    Start by loading/formatting the data...
"""

import pickle
import sys
sys.path.append("../tools/")
from feature_format import featureFormat, targetFeatureSplit

data_dict = pickle.load(open("../final_project/final_project_dataset.pkl", "r") )

### add more features to features_list!
features_list = ["poi", "salary"]

data = featureFormat(data_dict, features_list)
labels, features = targetFeatureSplit(data)


### your code goes here 

from sklearn import tree
from sklearn import cross_validation
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

# random_state especifica la posicion en la que se divide el set
features_train, features_test, labels_train, labels_test = cross_validation.train_test_split(features, labels, test_size=0.3, random_state=42)



clf = tree.DecisionTreeClassifier()
clf = clf.fit(features_train, labels_train)
pois = clf.predict(features_test)
print "labels= "
print labels_test
print "# Predictions= "
print len([i for i in pois if i == 1])
print "predictions= "
print pois
print "TOTAL="
print len(pois)
print "precision="
print precision_score(labels_test, pois)
print clf.score(features_test, labels_test)
print "recall"
print recall_score(features_test, labels_test, average=None)
