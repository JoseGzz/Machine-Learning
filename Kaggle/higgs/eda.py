# load training data
import pandas as pd
print 'Loading trainig data'
train_data = pd.read_csv('l_train.csv')
# asign sets and make splits
y       = pd.DataFrame(train_data['Label'])
X       = train_data.iloc[:,:32]


# first I want to see what correlations exist within features
import matplotlib.pyplot as plt

def plot_corr(df,size=10):
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(size, size))
    ax.matshow(corr)
    plt.xticks(range(len(corr.columns)), corr.columns)
    plt.yticks(range(len(corr.columns)), corr.columns)
    plt.show()

plot_corr(X)

