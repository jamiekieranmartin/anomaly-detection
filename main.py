# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%

from IPython import get_ipython

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import pandas as pd
from matplotlib import pyplot as plt

from sklearn.decomposition import PCA

from sklearn import metrics
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import IsolationForest

import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')
# %%
df = pd.read_csv("./creditcard.csv")
# %%
cols = ['Time', 'Amount']

pca = PCA()
pca.fit(df[cols])
X_pca = pca.transform(df[cols])

df['V29'] = X_pca[:, 0]
df['V30'] = X_pca[:, 1]

df.drop(cols, axis=1, inplace=True)
# %%
columns = df.drop('Class', axis=1).columns
grid = plt.GridSpec(6, 5)

plt.figure(figsize=(20, 10*2))

for n, col in enumerate(df[columns]):
    ax = plt.subplot(grid[n])
    sns.distplot(df[df.Class == 1][col], color='g')
    sns.distplot(df[df.Class == 0][col], color='r')
    ax.axes.xaxis.set_visible(False)
    ax.axes.yaxis.set_visible(False)
    ax.set_frame_on(False)
    ax.set_title(str(col))
    ax.set_xlabel('')

plt.show()
# %%

# https://www.kaggle.com/sabanasimbutt/anomaly-detection-using-unsupervised-techniques

columns = df.drop('Class', axis=1).columns
non = df[df.Class == 0]
fraud = df[df.Class == 1]
size = len(fraud)
significant = ['Class']
critical_value = 2.5758

for i in columns:
    mean = non[i].mean()
    f_mean = fraud[i].mean()
    f_std = fraud[i].std()
    z = (f_mean - mean) / (f_std/np.sqrt(size))

    if(abs(z) >= critical_value):
        print(i, "significant")
        significant.append(i)
    else:
        print(i, "insignificant!")

df = df[significant]
df.shape
# %%

# https://towardsdatascience.com/train-test-split-and-cross-validation-in-python-80b61beca4b6

cols = ['V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V9', 'V10',
        'V11', 'V12', 'V14', 'V16', 'V17', 'V18', 'V19', 'V20',
        'V21', 'V24', 'V27', 'V28', 'V29', 'V30']

X = np.array(df[cols])
y = np.array(df['Class'])

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1)
# %%

# https://scikit-learn.org/stable/modules/generated/sklearn.metrics.classification_report.html


def classify(name, clf):
    '''function to test all models'''
    # fit the classifier with train data
    m = clf.fit(X_train, y_train)

    # run model prediction with training data
    print(name, "TRAINING")
    print(metrics.classification_report(y_train, m.predict(X_train)))

    # run model prediction with test data
    print(name, "TEST")
    print(metrics.classification_report(y_test, m.predict(X_test)))


# %%

# https://realpython.com/logistic-regression-python/
# https://datascience.foundation/sciencewhitepaper/understanding-logistic-regression-with-python-practical-guide-1

lr = LogisticRegression()
classify('Logistic Regression', lr)

# %%

# https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html
# https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html

# possible values for the parameter max_depth
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, None]

# parameters that will be used in cross validation
params = {"max_depth": max_depths}

# setup cross validation using a Decision Tree classifer,
# random_state set to 1 for repeatability and cv set to 10 folds
classify('DecisionTree', GridSearchCV(
    DecisionTreeClassifier(random_state=1), params, cv=10))

# %%

# https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

# possible values for the parameter n_neighbors
n_neighbors = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# parameters that will be used in cross validation
params = {"n_neighbors": n_neighbors}

# setup cross validation using a Nearest Neighbors classifer,
# random_state set to 1 for repeatability and cv set to 10 folds
classify('Nearest Neighbors', GridSearchCV(
    KNeighborsClassifier(), params, cv=10))

# %%

# https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html

# possible values for the parameter C
Cs = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# parameters that will be used in cross validation
params = {"C": Cs}

# setup cross validation using a Support Vector Machine classifer,
# random_state set to 1 for repeatability and cv set to 10 folds
classify('Support Vector Machine classifier',
         GridSearchCV(SVC(random_state=1), params, cv=10))

# %%

# https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPClassifier.html

# possible values for the parameter hidden_layer_sizes
hidden_layer_sizes = [50, 75, 100, 125, 150, 175,
                      200, 225, 250, 275, 300, 325, 350, 375, 400]

# parameters that will be used in cross validation
params = {"hidden_layer_sizes": hidden_layer_sizes}

# setup cross validation using a Neural Network classifer,
# random_state set to 1 for repeatability
# tol set to 1e-2 so that the maximum iteration is never reached
# cv set to 10 folds
classify('Neural Network', GridSearchCV(
    MLPClassifier(random_state=1), params, cv=10))

# %%

# https://towardsdatascience.com/outlier-detection-with-isolation-forest-3d190448d45e

# separate inliers and outliers
non = df[df.Class == 0].drop(['Class'], axis=1)
fraud = df[df.Class == 1].drop(['Class'], axis=1)

# setup and fit IsolationForest with non-fraudulent transactions
clf = IsolationForest(random_state=1)
clf.fit(non)


def accuracy(values, type):
    '''tests accuracy of given values'''
    accuracy = list(values).count(type)/values.shape[0]

    return '{accuracy:.2f}%'.format(accuracy=accuracy * 100)


# test accuracy of non-fraudulent transactions
print("Non-Fraudulent Transaction Accuracy:",
      accuracy(clf.predict(non), 1))
# test accuracy of fraudulent transactions
print("Fraudulent Transaction Accuracy:",
      accuracy(clf.predict(fraud), -1))
