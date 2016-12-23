from pandas.tools.plotting import scatter_matrix
from sklearn.decomposition import PCA
from sklearn import tree
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn import svm, grid_search
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
# for learning curve
import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_digits
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
'''
crim
per capita crime rate by town.

zn
proportion of residential land zoned for lots over 25,000 sq.ft.

indus
proportion of non-retail business acres per town.

chas
Charles River dummy variable (= 1 if tract bounds river; 0 otherwise).

nox
nitrogen oxides concentration (parts per 10 million).

rm
average number of rooms per dwelling.

age
proportion of owner-occupied units built prior to 1940.

dis
weighted mean of distances to five Boston employment centres.

rad
index of accessibility to radial highways.

tax
full-value property-tax rate per \$10,000.

ptratio
pupil-teacher ratio by town.

black
1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town.

lstat
lower status of the population (percent).

medv
median value of owner-occupied homes in \$1000s.
'''

features = ['crim', 'zn', 'indus', 'chas', 'nox', 'rm', 'age', 'dis', 'rad', 'tax', 'ptration', 'black', 'lstat', 'medv']



def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):

    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt

# function for correlation matrix (colors)
def plot_corr(df,size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()


# put data into workable form
print "Loading data..."
train = pd.read_csv('train.csv')
testFinal = pd.read_csv('test.csv')


ids = testFinal['ID']
testFinal = testFinal.drop('ID', axis=1)


print "Finished loading data!"
X = train[train.columns[1:train.shape[1] - 1]]
y = train['medv']


# display each column's data type
# so we can detect inconsistencies
print "---- Data types by feature ----"
columns =  X.columns
for column in columns:
	vals = [type(val) for val in X[column].unique()]
	#print 'feature: {}, types: {}'.format(column, np.unique(vals))

# turn all values into int's
'''
for column in columns:
	X[column] = X[column].astype(np.float64)
'''
# remove null values
X = X.replace('', np.nan)
X = X.replace('NaN', np.nan)
X = X.replace('nan', np.nan)
X = X.fillna(0)
y = [int(val) for val in y]

#X['chas'] = X['chas'].astype(np.int64)
#testFinal['chas'] = testFinal['chas'].astype(np.int64)



# plot relations between features to detect patters or
# anomalies
#X.hist()
#scatter_matrix(X)

# correlation matrix (colors), > correlation is red/brown
#plot_corr(X)


#half_cols = columns[0:len(columns)/2 + 1]

# create a list of tuples with 2 features each and 
# the correlation between them
corr_tups = []

for column in columns:
	corr = X.corr()[column]
	corr_dict = corr.to_dict()
	del corr_dict[column]
	max_key = max(corr_dict, key=lambda k: corr_dict[k])
	pair = (column, max_key, corr_dict[max_key])
	corr_tups.append(pair)
	#print '{} with {}, value: {}'.format(
		#column, max_key, corr_dict[max_key])

# get only features that have more than 0.69 correlation
min_corr = 0.69
corr_tups = filter(lambda tup: tup[2] > min_corr, corr_tups)

# remove symetric duplicates
corr_vals = []
_ = [
	corr_vals.append((a,b,c)) for a,b,c in corr_tups
		if (b,a,c) not in corr_vals
]

# manual removal of outliers

train = train[train['tax'] < 600]
train = train[train['indus'] < 16]
train = train[train['nox'] < 0.6]

lim = 10
for column in columns:
    # keep only the ones that are within +lim to -lim standard2deviations in the columns.
    # the lim value was guessed and tested by trail and error
	X = X[np.abs(X[column]-X[column].mean()) > (lim*X[column].std())] 

# reasign values to fit X and y dimensions 
y = train['medv']
X = train[train.columns[1:train.shape[1] - 1]]

# plot correlated featues (used to deal with outliers)
for tup in corr_vals:
	f1, f2 = X[tup[0]], X[tup[1]]
	plt.figure()
	plt.scatter(f1, f2, marker='x')
	plt.xlabel(tup[0])
	plt.ylabel(tup[1])
#plt.show()

# perform dimensionality reduction with PCA
del_cols = set()

for tup in corr_vals:
	print "still inside"
	X_aux = pd.DataFrame()
	X_test = pd.DataFrame()

	X_aux[str(tup[0])] = X[str(tup[0])]
	X_aux[str(tup[1])] = X[str(tup[1])]
	
	X_test[str(tup[0])] = testFinal[str(tup[0])]
	X_test[str(tup[1])] = testFinal[str(tup[1])]

	pca = PCA(n_components=1, svd_solver='arpack')
	pca.fit(X_aux)

	X_new = pca.transform(X_aux)
	new_name = str(tup[0]) + '-' + str(tup[1])
	X[new_name] = X_new

	pca.fit(X_test)
	X_new_test = pca.transform(X_test)

	testFinal[new_name] = X_new_test
	del_cols.add(tup[0])
	del_cols.add(tup[1])
X = X.drop(list(del_cols), axis=1)
testFinal = testFinal.drop(list(del_cols), axis=1)

# SelectPercentile
from sklearn.feature_selection import SelectPercentile, f_classif
selector = SelectPercentile(f_classif, percentile=5)
selector.fit(X, y)
scores = -np.log10(selector.pvalues_)
scores /= scores.max()
#print 'SCORES'
#print scores

#X = X.drop('dis', axis=1)
#print X.head()

# linear regression score increased 16% after PCA
clf = linear_model.LinearRegression(
	copy_X=True, normalize=False, fit_intercept=True
)
#clf = linear_model.LinearRegression()
#clf = linear_model.SGDClassifier()
#clf = linear_model.Lasso(alpha=0.1)
#clf = linear_model.ElasticNet(alpha=0.1, l1_ratio=0.7)
#clf = svm.SVC(kernel='linear')
#y = y.astype(np.int64)
#clf = svm.SVR()

#print clf.get_params().keys()
'''
parameters = {
				'fit_intercept':(True, False),
				'normalize':(True, False),
				'copy_X': (True, False),
			}

clfgs = grid_search.GridSearchCV(clf, parameters)
clfgs.fit(X, y)
print clfgs.best_params_
'''

X_v = X.values
y_v = y.values

# do cross validation with K folds
kf = KFold(n_splits=2)

print "Training and testing..."
scores = [
	clf.fit(X_v[train], y_v[train]).score(X_v[test],y_v[test]) 
	for train, test in kf.split(X_v)
]
print 'Score:', np.array(scores).mean()

#------------ evaluate variance-bias with learning curve
# results show some bias, more trainig examples will not help much
title = "Learning Curves (Linear Regression)"
# Cross validation with 100 iterations to get smoother mean test and train
# score curves, each time with 20% data randomly selected as a validation set.
cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=0)

estimator = clf
#plot_learning_curve(estimator, title, X, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
#plt.show()


predictions = clf.predict(X)
print 'MSE error:', mean_squared_error(y, predictions)

predictions = clf.predict(testFinal)

print "Generating CSV"
submission = pd.DataFrame({
        "ID": ids,
        "medv": predictions
    })
 
submission.to_csv('bostonHouseSubmission1.csv', index=False)
