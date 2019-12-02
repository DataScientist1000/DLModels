# -*- coding: utf-8 -*-
"""
Created on Mon Jul  8 15:03:05 2019
@author: BBerry
Model used : ANN
"""

#Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

path = os.path.abspath("ANN/Data/data.csv")
#Import the data
dataset = pd.read_csv(path)
del dataset['Unnamed: 32']

X = dataset.iloc[:, 2:].values
y = dataset.iloc[:, 1].values

# Encoding categorical data
from sklearn.preprocessing import LabelEncoder
le_x1 = LabelEncoder()
y = le_x1.fit_transform(y) 

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(X ,y,test_size = 0.1 , random_state = 0)
 
#Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.fit_transform(X_test)

#Import keras packages
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout
 
#Initialising the ANN
classifier = Sequential()
 
#Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 16 ,init = 'uniform',activation = 'relu', input_dim = 30 ))

#Adding dropout to prevent overfitting
classifier.add(Dropout(p = 0.1))
 
"""
input_dim - number of columns of the dataset

output_dim - number of outputs to be fed to the next layer, if any

activation - activation function which is ReLU in this case

init - the way in which weights should be provided to an ANN

The ReLU function is f(x)=max(0,x). Usually this is applied element-wise to the output of some other function, such as a matrix-vector product. In MLP usages, rectifier units replace all other activation functions except perhaps the readout layer. But I suppose you could mix-and-match them if you'd like. One way ReLUs improve neural networks is by speeding up training. The gradient computation is very simple (either 0 or 1 depending on the sign of x). Also, the computational step of a ReLU is easy: any negative elements are set to 0.0 -- no exponentials, no multiplication or division operations. Gradients of logistic and hyperbolic tangent networks are smaller than the positive portion of the ReLU. This means that the positive portion is updated more rapidly as training progresses. However, this comes at a cost. The 0 gradient on the left-hand side is has its own problem, called "dead neurons," in which a gradient update sets the incoming values to a ReLU such that the output is always zero; modified ReLU units such as ELU (or Leaky ReLU etc.) can minimize this.
 """
 # Adding the second hidden layer
classifier.add(Dense(output_dim=16, init='uniform', activation='relu'))
# Adding dropout to prevent overfitting
classifier.add(Dropout(p=0.1))

# Adding the output layer
classifier.add(Dense(output_dim=1, init='uniform', activation='sigmoid'))

#compiling the ANN
classifier.compile(optimizer ='adam',loss = 'binary_crossentropy', metrics = ['accuracy'])

#Fitting ANN to the training Set 
classifier.fit(X_train,y_train,batch_size= 100, epochs = 150)

#predicting the test results
y_pred = classifier.predict(X_test)
y_pred = (y_pred > 0.5) 

#Confusion matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)

print("Our accuracy is {}%".format(((cm[0][0] + cm[1][1])/57)*100))

sns.heatmap(cm,annot=True)
plt.savefig(r'C:\Users\bberry\Documents\MLTraining\Kaggle\ANNCancer\h.png')
