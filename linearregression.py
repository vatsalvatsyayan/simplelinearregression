# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn

dataset = pd.read_csv('Salary_Data.csv')

# We will now split the dataset into an independent variable X and dependent variable y

X = dataset.iloc[:,:-1].values
y = dataset.iloc[:,1].values

# We now split data based on training and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=1/3,random_state=0)

# We now implement classifier using Simple Linear Regression

from sklearn.linear_model import LinearRegression
simpleLinearRegression = LinearRegression()
simpleLinearRegression.fit(X_train,y_train)

y_predict=simpleLinearRegression.predict(X_test)

# We now implement the graph

plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,simpleLinearRegression.predict(X_train))
plt.show()