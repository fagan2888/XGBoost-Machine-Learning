
from xgboost import XGBClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import StandardScaler
#from sklearn.model_selection import GridSearchCV 
#from sklearn.decomposition import PCA

train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

X_train = train_df.drop('target',axis=1)
y_train = train_df['target']

#pca = PCA(n_components = 2)
#X_train = pca.fit_transform(X_train)

x_train,x_validation,y_train,y_validation = train_test_split(X_train,y_train,test_size=0.2,random_state=0)

sc = StandardScaler()
x_train = sc.fit_transform(x_train)
x_validation = sc.transform(x_validation)
test_norm = sc.transform(test_df)

classifier = XGBClassifier(max_depth = 10, subsample = 1, min_child_weight = 1.5, eta = 0.1, colsample_bytree = 0.5, seed = 42, num_rounds = 1000, silent = 1)
classifier.fit(x_train,y_train)

#y_pred = classifier.predict(x_validation)
y_pred = classifier.predict_proba(test_norm)
y_pred = y_pred[:,1]
y_pred = (y_pred > 0.5)

sub=pd.DataFrame()
test_id = test_df.id.values
sub['id'] = test_id
sub['target'] = y_pred
sub.to_csv('Xgboost4.csv', index=False)

#y_pred = classifier.predict(x_validation)

cm = confusion_matrix(y_validation,y_pred)

acc = accuracy_score(y_validation, y_pred)


accuracies = cross_val_score(estimator = classifier , X = x_train, y = y_train, cv = 10)
accuracies.mean()
accuracies.std()

parameters = [{'learning_rate':[0.05,0.08,0.1,]}]

grid_search = GridSearchCV(estimator= classifier, param_grid = parameters, scoring = 'accuracy', cv = 10, n_jobs = -1)

grid_search = grid_search.fit(x_train,y_train)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_
