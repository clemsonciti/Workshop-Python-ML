---
title: "Data Partition with Scikit-Learn"
teaching: 20
exercises: 0
questions:
- "What is Data Partition"
objectives:
- "Learn how to split data using sklearn"
keypoints:
- "sklearn, data partition"
---

# Data partition: training and testing

![image](https://user-images.githubusercontent.com/43855029/120378647-b1716080-c2ec-11eb-8693-60defbbad7e2.png)


- In Machine Learning, it is mandatory to have training and testing set. Some
 time a verification set is also recommended. Here are some functions 
 for splitting training/testing set in `sklearn`:

- `train_test_split`: create series of test/training partitions
- `Kfold` splits the data into k groups
- `StratifiedKFold` splits the data into k groups based on a grouping factor.
- `RepeatKfold`
- `ShuffleSplit`
- `LeaveOneOut`
- `LeavePOut`

Due to time constraint, we only focus on `train_test_split`, `KFolds` and  `StratifiedKFold`.

## 3.1 Scikit-Learn data

The `sklearn.datasets` package embeds some small sample [datasets](https://scikit-learn.org/stable/datasets.html)

For each dataset, there are 4 varibles:

- **data**: numpy array of predictors/`X`
- **target**: numpy array of predictant/target/`y`
- **feature_names**: names of all predictors in `X`
- **target_names**: names of all predictand in `y`

For example:

```python
from sklearn.datasets import load_iris
data = load_iris()
print(data.data)
print(data.target)
print(data.feature_names)
print(data.target_names)
```


In this example we gonna use the renowned iris flower data

```python
from sklearn.datasets import load_iris
iris = load_iris()
X = iris.data
y = iris.target
```

## 3.2 Data splitting using `train_test_split`: **Single fold**
Here we use `train_test_split` to randomly split 60% data for training and the rest for testing:

```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.6,random_state=123)
#random_state: int, similar to R set_seed function
```

## 3.3 Data splitting using `K-fold`
- This is the Cross-validation approach.
- This is a resampling process used to evaluate ML model on limited data sample.
- The general procedure:
    - Shuffle data randomly
    - Split the data into **k** groups
    For each group:
        - Split into training & testing set
        - Fit a model on each group's training & testing set
        - Retain the evaluation score and summarize the skill of model


![image](https://user-images.githubusercontent.com/43855029/114211785-103edd00-992f-11eb-89d0-bbd7bd0c0178.png)

[Documentation on split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html#sklearn.model_selection.KFold.split)

```python
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

kf10 = KFold(n_splits=10,shuffle=True,random_state=20)

# initialize the model
model = LogisticRegression(solver="liblinear", multi_class="auto")
i = 1

for train_index, test_index in kf10.split(X):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    model.fit(X_train, y_train) #Training the model, not running now
    y_pred = model.predict(X_test)
    print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test, y_pred)}")
    i += 1
```



## 3.4 Data splitting using `Stratified K-fold`
- StratifiedKFold takes the cross validation one step further: it ensures that the target has balance class distribution.

Running Logistic Regression for Stratified KFold

```python
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
import pandas as pd

kf = StratifiedKFold(n_splits=3, shuffle=True, random_state=123)
model = LogisticRegression(solver="liblinear", multi_class="auto")

i = 1

for train_index, test_index in kf.split(X, y):
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    model.fit(X_train, y_train) #Training the model
    y_pred = model.predict(X_test)
    print(f"Accuracy for the fold no. {i} on the test set: {accuracy_score(y_test, y_pred)}") 
    i += 1  
```
- Look at the sample below:
  - The target has imbalanced class distribution with 12 values of `1.0` and 4 
  values of `0.0`. KFold will not take that into consideration when splitting the Fold

```python
import pandas as pd
data = [[0.43547, 1.0], [0.871825, 1.0], [0.835452, 1.0], [0.555067, 1.0],
        [0.598458, 1.0], [0.297142, 1.0], [0.336659, 1.0], [0.397795, 1.0],
        [0.206699, 1.0], [0.025118, 1.0], [0.816815, 1.0], [0.101904, 1.0],
        [0.722744, 0.0], [0.049825, 0.0], [0.965084, 0.0], [0.928273, 0.0]]
 
# Create the pandas DataFrame
df = pd.DataFrame(data, columns=['col_a', 'target'])
X = df.col_a
y = df.target

# print dataframe.
df
```

Here is the result if using K-Fold:

```python
from sklearn.model_selection import KFold
kf10 = KFold(n_splits=4,shuffle=True,random_state=20)
for train_index, test_index in kf10.split(X):
    print(train_index,test_index,sep="--")

```
Here is the result of using Stratified K-Fold:

```python
from sklearn.model_selection import StratifiedKFold
kf = StratifiedKFold(n_splits=4, shuffle=True, random_state=123)

for train_index, test_index in kf.split(X, y):
    print(train_index,test_index,sep="--")
```