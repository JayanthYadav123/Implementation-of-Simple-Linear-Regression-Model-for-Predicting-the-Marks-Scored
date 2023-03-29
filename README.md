# Implementation-of-Simple-Linear-Regression-Model-for-Predicting-the-Marks-Scored

## AIM:
To write a program to predict the marks scored by a student using the simple linear regression model.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the standard Libraries.
2. Set variables for assigning dataset values.
3. Import linear regression from sklearn.
4. Assign the points for representing in the graph
5. Predict the regression for marks by using the representation of the graph.
6. Compare the graphs and hence we obtained the linear regression for the given datas.
## Program:
```
Program to implement the simple linear regression model for predicting the marks scored.
Developed by: G.Jayanth
RegisterNumber: 212221230030
```
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error,mean_squared_error
df=pd.read_csv('student_scores.csv')
#displaying the content in datafile
df.head()
df.tail()
x=df.iloc[:,:-1].values
x
y=df.iloc[:,1].values
y
#splitting train and test data set
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=1/3,random_state=0)
from sklearn.linear_model import LinearRegression
regressor=LinearRegression()
regressor.fit(x_train,y_train)
y_pred=regressor.predict(x_test)
#displaying the predicted values
y_pred
y_test
#graph plot for traing data
plt.scatter(x_train,y_train,color='green')
plt.plot(x_train,regressor.predict(x_train),color='orange')
plt.title("Hours vs Scores(training set)")
plt.xlabel("Hours")
plt.ylabel("Scores")
plt.show()
mse=mean_squared_error(y_test,y_pred)
print('MSE = ',mse)
mae=mean_absolute_error(y_test,y_pred)
print('MAE = ',mae)
rmse=np.sqrt(mse)
print('RMSE = ',rmse)
```
## Output:
<img width="390" alt="image" src="https://user-images.githubusercontent.com/94836154/228607005-08e711f1-1e88-453c-a0bf-f51821a0a50b.png">
<img width="242" alt="image" src="https://user-images.githubusercontent.com/94836154/228607467-a75fb527-5567-4d21-abcc-ba5bee867722.png">
<img width="341" alt="image" src="https://user-images.githubusercontent.com/94836154/228607825-bd4c32d0-2709-4274-9c0f-1e6fb5f47312.png">
<img width="469" alt="image" src="https://user-images.githubusercontent.com/94836154/228608214-e2375fa2-9e91-4d97-9569-db5b0ca18ce7.png">
<img width="373" alt="image" src="https://user-images.githubusercontent.com/94836154/228608461-6aee2bbf-6f1e-41c2-82c9-87a23353a568.png">
<img width="380" alt="image" src="https://user-images.githubusercontent.com/94836154/228608597-6e5496ce-4ae7-4156-8c95-db3b3a841cf3.png">

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
