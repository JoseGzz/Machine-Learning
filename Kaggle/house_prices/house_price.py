import pandas as pd
from sklearn.linear_model import LinearRegression


train = pd.read_csv('train.csv')
ids   = train[train.columns[1]]

X = train[train.columns[1:train.shape[1]-1]]
y = train['SalePrice']

clf = LinearRegression()

print X.head()


#clf.fit(X, y)

#print clf.score(X, y)

