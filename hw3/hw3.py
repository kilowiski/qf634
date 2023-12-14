"""
The data is related with direct marketing campaigns of a Portuguese banking institution. The
marketing campaigns were based on phone calls. Often, more than one contact to the same client was
required, in order to access if the product (bank term deposit) would be (or not) subscribed. Data set is
public – acknowledgement: S. Moro, R. Laureano and P. Cortez. Using Data Mining for Bank Direct
Marketing: An Application of the CRISP-DM Methodology. In P. Novais et al. (Eds.), Proceedings of
the European Simulation and Modelling Conference - ESM'2011, pp. 117-121, Guimarães, Portugal,
October, 2011. EUROSIS.
Download from ‘banking3.csv’. This is a simplified version of the original data set. There are 18
features and one label variable y. y = 1 denotes success in getting client to put in a term deposit. y=0
denotes failure to do so.
The features are self-explanatory. Many variables are dummy variables, e.g., job_blue-collar = 1 if the
potential client is a blue-collar worker, 0 otherwise. Default_no = 1 means client has defaulted in some
loans before, 0 otherwise. Contact_cellular = 1 if it is contact by handphone (not land line telephone)
and 0 otherwise. Month_apr = 1 means last contact timing is previous April, 0 otherwise, and so o
"""
import pandas as pd
import numpy as np
from sklearn import preprocessing
import matplotlib.pyplot as plt

plt.rc("font", size=14)
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import sklearn.metrics as metrics
from sklearn.ensemble import RandomForestClassifier
# read data frame
df = pd.read_csv("/home/kilo/Desktop/qf634/qf634file/banking3.csv")
df.dropna(inplace=True)
# print(df.info())
y = df["y"]
X = df.drop(columns=["Unnamed: 0", "y"])
#print(y.info())
#print(X.info())

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
RF_model = RandomForestClassifier(n_estimators=1000, random_state=1, max_features=None, max_depth=None)
RF_model.fit(X_train,y_train)
y_pred_RF = RF_model.predict(X_test)

Accuracy_RF = metrics.accuracy_score(y_test, y_pred_RF)
print("RF Accuracy:",Accuracy_RF)

confusion_matrix = confusion_matrix(y_test, y_pred_RF)
print('------- confusion matrix -------')
print(confusion_matrix)
print('\n------- classification report -------')
print(classification_report(y_test, y_pred_RF))

# calculate the fpr and tpr for all thresholds of the classification
preds_RF = RF_model.predict_proba(X_test)[:,1]
fpr, tpr, thresholds = metrics.roc_curve(y_test, preds_RF)  
# matches y_test of 1's and 0's versus pred prob of 1's for each of the 10,523 test cases
# sklearn.metrics.roc_curve(y_true, y_score,...) requires y_true as 0,1 input and y_score as prob inputs
# this metrics.roc_curve returns fpr, tpr, thresholds (Decreasing thresholds used to compute fpr and tpr)
# above can also be done using: fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
roc_auc_logreg = metrics.auc(fpr, tpr)
# sklearn.metrics.auc(fpr,tpr) returns AUC using trapezoidal rule
# Compute Area Under the Curve (AUC) using the trapezoidal rule.
# This is a general function, given points on a curve. For computing the area under the ROC-curve, see roc_auc_score.

#plot stuff
plt.rc("font", size=14)
plt.title("Receiver Operating Characteristic")
plt.plot(fpr, tpr, "b", label="AUC = %0.5f" % roc_auc_logreg)
plt.legend(loc="lower right")
plt.plot([0, 1], [0, 1], "r--")
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()
