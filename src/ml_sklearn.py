import pandas as pd 
import os
import sklearn as sk
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm

os.chdir("/home/quangio/JuliaProject/test-project/data/processed_working")
data = pd.read_csv("out/data.csv")
data.head()
n = len(data.iloc[0]) - 1
Y = data.iloc[:, n]
X = data.iloc[:, :n]

RF = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
RF.fit(X, Y)
round(RF.score(X,Y), 4)

RF.predict(X.iloc[460:,:])

SVM = svm.LinearSVC()
SVM.fit(X, Y)
round(SVM.score(X,Y), 4)