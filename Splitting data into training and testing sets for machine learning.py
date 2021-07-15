# -*- coding: utf-8 -*-
"""
Created on Thu May  6 01:23:43 2021

@author: abc
"""

#Splitting the dataset into training and testing

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model
df = pd.read_csv('cells.csv')
print(df)

x_df = df.drop('cells', axis='columns')
y_df = df.cells
#Step : 1 Creating an instance for model

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_df, y_df, test_size=0.4, random_state=10)

#print(X_train)

#Step : 2 fit the dataset
reg = linear_model.LinearRegression()
reg.fit(X_train,y_train)

#Step : 3 predict the dataset

prediction_test = reg.predict(X_test)
print(y_test,prediction_test)
print("Mean sq. error between y_test and predicted =",np.mean(prediction_test-y_test)**2)


#Step : 4 Residual plot
plt.scatter(prediction_test, prediction_test-y_test)
plt.hlines(y=0, xmin = 200, xmax=310)

                    

                       #THANK YOU 














