import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.cross_validation import KFold
import numpy as np
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import re
import operator
import matplotlib.pyplot as plt
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.ensemble import GradientBoostingClassifier
 
 
def get_title(name):
    regex = re.search(' ([A-Za-z]+)\.', name)
    if regex:
        return regex.group(1)
    return ""
 
 
def get_family_id(row):
    last_name = row["Name"].split(",")[0]
    family_id = "{0}{1}".format(last_name, row['FamilySize'])
    if family_id not in family_id_mapping:
        if len(family_id_mapping) == 0:
            current_id = 1
        else:
            current_id = (max(family_id_mapping.items(), key=operator.itemgetter(1))[1] + 1)
        family_id_mapping[family_id] = current_id
    return family_id_mapping[family_id]
 
 
titanic = pd.read_csv("train2.csv")
test = pd.read_csv("test.csv")
 
print('---Has NaN values---')
for attr in titanic:
    print(str(titanic[attr].name) + ': ' + str(titanic[attr].isnull().values.any()))
 
train = titanic[['PassengerId', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 'Ticket', 'Fare', 'Cabin', 'Embarked']]
target = titanic['Survived']
 
print('Most common value in Embarked: ' + str(train.Embarked.value_counts().idxmax()))
 
train.Embarked = train.Embarked.fillna('S')
train.Age = train.Age.fillna(train.Age.median())
 
print('---Has NaN values---')
for attr in train:
    print(str(train[attr].name) + ': ' + str(train[attr].isnull().values.any()))
 
print(train.columns.to_series().groupby(train.dtypes).groups)
 
print(train.Embarked.unique())
print(train.Sex.unique())
 
train.Embarked[train['Embarked'] == 'S'] = 0
train.Embarked[train['Embarked'] == 'C'] = 1
train.Embarked[train['Embarked'] == 'Q'] = 2
 
train.Sex[train.Sex == 'male'] = 0
train.Sex[train.Sex == 'female'] = 1
 
print(train.Sex)
 
'''
print(train.head(0))
print('-----------')
print(target.head())
print(train.describe())
'''
 
print(train.columns.to_series().groupby(train.dtypes).groups)
 
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]
 
alg = LinearRegression()
 
kf = KFold(train.shape[0], n_folds=3, random_state=1)
 
predictions = []
 
for training, testing in kf:
    train_samples = (train[predictors].iloc[training, :])
    train_target = target.iloc[training]
    alg.fit(train_samples, train_target)
    test_predictions = alg.predict(train[predictors].iloc[testing, :])
    predictions.append(test_predictions)
    print('----')
    print(test_predictions)
 
print(predictions)
 
predictions = np.concatenate(predictions, axis=0)
 
predictions[predictions > 0.5] = 1
predictions[predictions <= 0.5] = 0
 
match = target == predictions
print('Accuracy with Linear R ' + str(100 * sum(match) / predictions.size) + " %")
 
alg = LogisticRegression(random_state=1)
scores = cross_validation.cross_val_score(alg, train[predictors], target, cv=3)
 
print("Accuracy with Logistic R: " + str(100 * scores.mean()))
 
print("Without cv: " + str(alg.fit(train[predictors], target)))
 
print(alg.score(train[predictors], target))
 
alg = RandomForestClassifier(random_state=1, n_estimators=10, min_samples_split=2, min_samples_leaf=1)
kf = cross_validation.KFold(train.shape[0], n_folds=3, random_state=1)
scores = cross_validation.cross_val_score(alg, train[predictors], target, cv=kf)
print("Score with random forest: " + str(scores.mean()))
 
# increase parameters to reduce overfitting
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=4, min_samples_leaf=2)
scores = cross_validation.cross_val_score(alg, train[predictors], target, cv=kf)
print('Random forest socres with increased parameters: ' + str(scores.mean()))
 
# create new features
train["FamilySize"] = train["SibSp"] + train["Parch"]
train["NameLength"] = train["Name"].apply(lambda x: len(x))
 
 
titles = train.Name.apply(get_title)
print(pd.value_counts(titles))
 
title_map = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2}
 
for t, i in title_map.items():
    titles[titles == t] = i
 
print(pd.value_counts(titles))
 
train["Title"] = titles
 
family_id_mapping = {}
 
 
# Get the family ids with the apply method
family_ids = train.apply(get_family_id, axis=1)
 
family_ids[train.FamilySize < 3] = -1
 
print(pd.value_counts(family_ids))
 
train["FamilyId"] = family_ids
 
 
# visualize data
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
 
selector = SelectKBest(f_classif, k=5)
selector.fit(train[predictors], target)
 
scores = -np.log10(selector.pvalues_)
 
plt.bar(range(len(predictors)), scores)
plt.xticks(range(len(predictors)), predictors, rotation='vertical')
plt.show()
 
predictors = ["Pclass", "Sex", "Fare", "Title"]
 
alg = RandomForestClassifier(random_state=1, n_estimators=50, min_samples_split=8, min_samples_leaf=4)
 
kf = cross_validation.KFold(train.shape[0], n_folds=3, random_state=1)
 
scores = cross_validation.cross_val_score(alg, train[predictors], target, cv=kf)
 
print("Score with the new Title feature and random forest: " + str( 100 * scores.mean()))
 
# use gradient boosting
 
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]
 
# Initialize the cross validation folds
kf = KFold(train.shape[0], n_folds=3, random_state=1)
 
predictions = []
 
for training, testing in kf:
    train_target = target.iloc[training]
    full_test_predictions = []
    for alg, predictors in algorithms:
        alg.fit(train[predictors].iloc[training, :], train_target)
        test_predictions = alg.predict_proba(train[predictors].iloc[testing, :].astype(float))[:, 1]
        full_test_predictions.append(test_predictions)
    test_predictions = (full_test_predictions[0] + full_test_predictions[1]) / 2
    test_predictions[test_predictions <= 0.5] = 0
    test_predictions[test_predictions > 0.5] = 1
    predictions.append(test_predictions)
 
predictions = np.concatenate(predictions, axis=0)
 
accuracy = sum(predictions[predictions == target]) / len(predictions)
print('Accuracy with Gradient boosting and RF clf average: ' + str(100 * accuracy))
 
# construct the test set
 
test["Embarked"] = test["Embarked"].fillna('S')
test.Age = test.Age.fillna(test.Age.median())
 
print("Embarked unique: " + str(test.Embarked.unique()))
 
 
test.Embarked[test.Embarked == 'S'] = 0
test.Embarked[test.Embarked == 'C'] = 1
test.Embarked[test.Embarked == 'Q'] = 2
 
test.Sex[test.Sex == 'male'] = 0
test.Sex[test.Sex == 'female'] = 1
 
test.Fare = test.Fare.fillna(test.Fare.median())
 
titles = test["Name"].apply(get_title)
 
title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6, "Major": 7, "Col": 7, "Mlle": 8, "Mme": 8, "Don": 9, "Lady": 10, "Countess": 10, "Jonkheer": 10, "Sir": 9, "Capt": 7, "Ms": 2, "Dona": 10}
 
for k, v in title_mapping.items():
    titles[titles == k] = v
 
test["Title"] = titles
 
print(pd.value_counts(test["Title"]))
 
# Now, we add the family size column.
test["FamilySize"] = test["SibSp"] + test["Parch"]
 
print(family_id_mapping)
 
family_ids = test.apply(get_family_id, axis=1)
family_ids[test["FamilySize"] < 3] = -1
test["FamilyId"] = family_ids
test["NameLength"] = test["Name"].apply(lambda x: len(x))
 
predictors = ["Pclass", "Sex", "Age", "Fare", "Embarked", "FamilySize", "Title", "FamilyId"]
 
algorithms = [
    [GradientBoostingClassifier(random_state=1, n_estimators=25, max_depth=3), predictors],
    [LogisticRegression(random_state=1), ["Pclass", "Sex", "Fare", "FamilySize", "Title", "Age", "Embarked"]]
]
 
# generate predictions and submission
full_predictions = []
 
for attr in test:
    print(str(test[attr].name) + ': ' + str(test[attr].isnull().values.any()))
 
 
for alg, predictors in algorithms:
    alg.fit(train[predictors], target)
    predictions = alg.predict_proba(test[predictors].astype(float))[:, 1]
    full_predictions.append(predictions)
 
# The gradient boosting classifier generates better predictions, weight it higher
predictions = (full_predictions[0] * 3 + full_predictions[1]) / 4
 
predictions[predictions <= .5] = 0
predictions[predictions > .5] = 1
 
predictions = predictions.astype(int)
 
submission = pd.DataFrame({
        "PassengerId": test["PassengerId"],
        "Survived": predictions
    })
 
submission.to_csv('titanicSubmission3.csv', index=False)
