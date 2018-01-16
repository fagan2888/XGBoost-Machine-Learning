import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import StandardScaler

train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

X_train = train.drop('target',axis=1)
y_train = train['target']


x_train,x_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.3,random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_validation = sc.transform(x_validation)
test_norm = sc.transform(test)

import keras
from keras.models import Sequential
from keras.layers import Dense ,Dropout

classifier = Sequential()

classifier.add(Dense(output_dim = 30, init = 'uniform',activation = 'relu', input_dim = 58 ))

classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim = 30, init = 'uniform',activation = 'relu'))

classifier.add(Dropout(0.5))

classifier.add(Dense(output_dim = 1, init = 'uniform',activation = 'sigmoid'))


classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

classifier.fit(x_train,y_train, batch_size = 10, nb_epoch = 2)

y_pred = classifier.predict(test_norm)

# For Model Performance
confusion_matrix(yvl,y_pred)
