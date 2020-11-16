# -*- coding: utf-8 -*-
"""
Created on Sat Dec 14 11:51:58 2019

@author: Emine
"""
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score



bankdata = pd.read_csv("data.txt")


X = bankdata.drop('Class', axis=1)
y = bankdata['Class']



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20)


svclassifier = SVC(kernel='linear')
svclassifier.fit(X_train, y_train)

y_pred = svclassifier.predict(X_test)


print("SVM ACCURACY:",accuracy_score(y_test,y_pred))

