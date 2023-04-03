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
1.df.head()


![image](https://user-images.githubusercontent.com/94836154/229478850-203156f5-c3e0-4862-9608-ac78042f90bd.png)

2.df.tail()


![image](https://user-images.githubusercontent.com/94836154/229478896-6025a80f-dbec-47ba-bb13-f35ee270b885.png)

3.Array value of X


![image](https://user-images.githubusercontent.com/94836154/229479052-5c933ff4-e752-45c2-a3de-b3fe3774fb09.png)

4.Array value of Y


![image](https://user-images.githubusercontent.com/94836154/229479108-36d609ff-7b70-4500-9fe0-0290c91a4541.png)

5.Values of Y prediction


![image](https://user-images.githubusercontent.com/94836154/229479152-ed314fd5-fd2c-41d9-988f-802dd61eaa47.png)

6.Array values of Y test


![image](https://user-images.githubusercontent.com/94836154/229479219-a432f6d4-59c7-4c90-a1cb-e3891f7f1111.png)

7.Training Set Graph


![image](https://user-images.githubusercontent.com/94836154/229479266-efc4648d-1c26-400f-b6e7-875ce066e7f0.png)

8.Values of MSE, MAE and RMSE


![image](https://user-images.githubusercontent.com/94836154/229479303-bc3d6fbd-4d2e-40c4-b5db-f24cd85672a9.png)

## Result:

Thus the program to implement the simple linear regression model for predicting the marks scored is written and verified using python programming.
