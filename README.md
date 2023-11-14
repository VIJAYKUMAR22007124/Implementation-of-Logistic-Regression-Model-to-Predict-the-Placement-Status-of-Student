# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the necessary libraries.
2. Drop the unnecessary columns.
3. Generate the cat codes.
4. Identify dependant and independent variables.
5. Split the data into training and testing data.
6. Implement Logistic Regression and train the model.
7. Predict whether the student would get placed or not using the model.

## Program:
```python
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by   :  B VIJAY KUMAR
RegisterNumber :  212222230173
*/
```
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df = pd.read_csv("Placement_Data_Full_Class.csv")

print(df.head())
print(df.info())
print(df.describe())

df = df.drop('sl_no', axis=1)
df = df.drop(['ssc_b','hsc_b','gender'],axis=1)

df['degree_t'] = df['degree_t'].astype('category')
df['workex'] = df['workex'].astype('category')
df['specialisation'] = df['specialisation'].astype('category')
df['status'] = df['status'].astype('category')
df['hsc_s'] = df['hsc_s'].astype('category')

df['degree_t'] = df['degree_t'].cat.codes
df['workex'] = df['workex'].cat.codes
df['specialisation'] = df['specialisation'].cat.codes
df['status'] = df['status'].cat.codes
df['hsc_s'] = df['hsc_s'].cat.codes

X = df.iloc[:,:-1].values
Y = df.iloc[:,-1].values

print(X.shape,'\n',Y.shape)

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X,Y,test_size=0.2)


df.head()

print(df['status'])


df.duplicated()

from sklearn.linear_model import LogisticRegression

clf = LogisticRegression(solver="liblinear")
clf.fit(X_train,Y_train)
y_pred = clf.predict(X_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion = (y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)
```
#### Prediction
```python
if(list(clf.predict([[0,87,0,95,0,2,0,0,2]]))[0] == 1):
    print("Will get placed")
else:
    print("Won't get placed")
```

## Output:

#### Placement data
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/0ee86b3d-a530-4c64-8896-752e2854c2ce)
#### Salary data
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/edd3f56a-dfd5-4739-b227-f781e26dd148)
#### Checking the null function()
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/f97c28ca-9d13-473a-abd6-a60f79624f8a)

#### Data duplicate
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/0bb65667-ff40-4650-859e-f229918fbac0)

#### Print data
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/01afd62b-e92b-4ea4-bc15-fffaa3f495ed)

#### Data status
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/e5828568-f464-4814-be69-47c1eb27ea4b)

#### y_prediction Array
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/b53c6054-da1c-414a-ab79-88e4ea943faf)

#### Accuracy value
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/95fa75b5-62f6-484b-8cee-78f3b57ae98e)

#### Confusion matrix
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/8bb8d630-1027-4cf6-84ef-88259355bf3a)

#### Classification report
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/35a214ce-7f29-45cf-9967-d316b8311f71)

#### Prediction of LR
![image](https://github.com/VIJAYKUMAR22007124/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/119657657/8bbd57f7-b7a8-49ab-9283-455268ae4c70)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
