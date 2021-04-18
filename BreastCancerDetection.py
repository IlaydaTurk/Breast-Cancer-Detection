#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


#Breast Cancer Dataset  "https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+%28Diagnostic%29"
df_BreastCancer=pd.read_csv('/Users/ilayda/Desktop/data.csv')
df_BreastCancer.head()


df_BreastCancer.shape

#Features of dataset
df_BreastCancer.columns

#Removing unnecessary columns that Unnamed:32 and id from dataset.
df_BreastCancer.drop(['id', 'Unnamed: 32'], axis=1, inplace=True)


df_BreastCancer.head()


df_BreastCancer.shape


df_BreastCancer.columns


#Visualizing Breast Cancer Data with first 5 features.
sns.pairplot(df_BreastCancer, hue = 'diagnosis', vars = ['radius_mean', 'texture_mean', 'perimeter_mean','area_mean','smoothness_mean'] )


#M = Malignant, B = Benign
df_BreastCancer['diagnosis'].value_counts()


sns.countplot(df_BreastCancer['diagnosis'], label = "Count")


#X is the features dataframe without the diagnosis column that we used for the training to model.
X = df_BreastCancer.drop(['diagnosis'], axis = 1) 
X.head()


#y is the target feature that we are trying to predict.
y = df_BreastCancer['diagnosis']
y.head()


#Splitting dataset as training data and testing data. We used 80%-20% ratio. 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)



#Normalizing our training data to improve the model's accuracy.
X_train_min = X_train.min()
X_train_max = X_train.max()
X_train_range = (X_train_max- X_train_min)
X_train_normalized = (X_train- X_train_min)/(X_train_range)


X_train_normalized.head()


#Normalize our test data.
X_test_min = X_test.min()
X_test_max = X_test.max()
X_test_range = (X_test_max - X_test_min)
X_test_normalized = (X_test - X_test_min)/X_test_range


X_test_normalized.head()


#SVM Model
svc_model = SVC()
svc_model.fit(X_train_normalized, y_train)


y_predict = svc_model.predict(X_test_normalized)
cm = confusion_matrix(y_test, y_predict)


#Using the confusion matrix to show the count of predicted true and false.
cm = np.array(confusion_matrix(y_test, y_predict, labels=['B','M']))
confusion = pd.DataFrame(cm, index=['Cancer', 'Healthy'],
                         columns=['predicted_Cancer','predicted_Healthy'])
print(confusion)


#Classification Report
print(classification_report(y_test,y_predict))






