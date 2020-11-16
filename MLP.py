# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 13:31:02 2019

@author: Emine
"""

import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score


#MLP Neural Network Model Creation
df = pd.read_csv('data.txt')
df.head()

X=df.drop('Class',axis=1)
y= df['Class']


X_train,X_test,y_train,y_test = train_test_split(X,y)

scaler = StandardScaler()
# Fit the Training Data 
scaler.fit(X_train)
X_train = scaler.transform(X_train)
X_test = scaler.transform(X_test)

mlp = MLPClassifier(hidden_layer_sizes=(4,7,2),max_iter=10)
mlp.fit(X_train,y_train)
predictions =mlp.predict(X_test)


print("MLP ACCURACY:",accuracy_score(y_test,predictions))
