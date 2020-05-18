# -*- coding: utf-8 -*-
"""Kaggletitanic.py

titani survival project.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

full_train_data = pd.read_csv("train.csv")
full_test_data = pd.read_csv("test.csv")
submission_data = pd.read_csv("gender_submission.csv")

display(full_train_data.head())

print(full_train_data["Embarked"].unique())

def clean_data(data):
  data["Fare"] = data["Fare"].fillna(data["Fare"].dropna().median())
  data["Age"] = data["Age"].fillna(data["Age"].dropna().median())

  data.loc[data["Sex"] == "male" ,  "Sex"] = 0
  data.loc[data["Sex"] == "female", "Sex"] = 1

  data["Embarked"] = data["Embarked"].fillna("S")
  data.loc[data["Embarked"] == "S", "Embarked"] = 0
  data.loc[data["Embarked"] == "C", "Embarked"] = 1
  data.loc[data["Embarked"] == "Q", "Embarked"] = 2

clean_data(full_train_data)
clean_data(full_test_data)

feature = full_train_data.drop(['PassengerId','Survived','Name', "Ticket", "Cabin"], axis=1)
target = full_train_data["Survived"]

x_train, x_test, y_train, y_test = train_test_split(feature, target, test_size = 0.25, random_state=40)

clf = DecisionTreeClassifier()
clf.fit(x_train, y_train)
y = clf.predict(x_test)
print(accuracy_score(y_test, y))

y1 = clf.predict(full_test_data.drop(['PassengerId','Name', "Ticket", "Cabin"], axis=1))

display(y1)


