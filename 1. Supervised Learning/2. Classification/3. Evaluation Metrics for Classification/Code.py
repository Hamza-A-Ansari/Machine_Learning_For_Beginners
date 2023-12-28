# Import necessary Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Import the dataset
dataset = pd.read_csv('LR.csv')
dataset

# Break the Dataset into Dependent and Independent Variables
X=dataset.iloc[:,[0,1]].values
y=dataset.iloc[:,2].values

#Training and Testing Data (divide the data into two part)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =train_test_split(X,y,test_size=0.25, random_state=0)

# Standardize our Data for better prediction
from sklearn.preprocessing import StandardScaler

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training our DTC model with entropy function
from sklearn.tree import DecisionTreeClassifier

# Here we use Entropy and Information Gain method for classification
classifer = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifer.fit(X_train,y_train)

# Predict our testing data

y_pred= classifer.predict(X_test)

# Create Confusion Matrix to see the accuracy
# note: Right digonal shows correct predictions while left digonal shows wrong predictions
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test,y_pred)
cm

# To find precision, recall and f1 score in short way
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

# To show ROC curve (receiver operating characteristic curve)
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, marker='.')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve')
plt.xlabel('Flase Positive Rate (1- Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)
plt.show()

print(fpr)
print(tpr)
print(thresholds)

# To find ROC score (Area Under Curve)
from sklearn.metrics import roc_auc_score

roc_auc_score(y_test, y_pred)

# To find the values of:
from sklearn import metrics

confusion = metrics.confusion_matrix(y_test, y_pred)
TP = confusion[1, 1] # True Positive
TN = confusion[0, 0] # True Negative
FP = confusion[0, 1] # False Positive
FN = confusion[1, 0] # False Negative

# To find Accuracy of our model through formula
print((TP + TN) / float(TP + TN + FP + FN))


# To find accuracy of our model through library
from sklearn.metrics import accuracy_score
accuracy_score(y_test,y_pred)

# To find Recall of our model through formula
print(TP / float(TP + FN))


# To find accuracy of our model through library
from sklearn import metrics
metrics.recall_score(y_test,y_pred)

# To find Precision of our model through formula
print(TP / float(TP + FP))


# To find accuracy of our model through library
from sklearn import metrics
metrics.precision_score(y_test,y_pred)

