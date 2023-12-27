# Import necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('Churn_Modelling.csv')
dataset

# Break the Dataset into Dependent and Independent Variables
X=dataset.iloc[:, 3:13].values
y=dataset.iloc[:, 13].values

# Label Encode and On Hot Encode
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

labelencoder_X_1 = LabelEncoder()
X[:, 1] = labelencoder_X_1.fit_transform(X[:, 1])

labelencoder_X_2 = LabelEncoder()
X[:, 2] = labelencoder_X_1.fit_transform(X[:, 2])

onehotencoder = OneHotEncoder()
X = onehotencoder.fit_transform(X).toarray()
X = X[:, 1:]

#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.25, random_state=0)

# Training our XGBoost model linear kernal
from xgboost import XGBClassifier

classifier = XGBClassifier()
classifier.fit(X_train, y_train)

# Predict our testing data

y_pred= classifier.predict(X_test)

# Create Confusion Matrix to see the accuracy
# note: Right digonal shows correct predictions while left digonal shows wrong predictions
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
cm

# To find accuracy of our model
from sklearn.metrics import accuracy_score

accuracy_score(y_test,y_pred)

