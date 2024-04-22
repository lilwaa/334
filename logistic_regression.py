import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from templatesplit import Splitter

splitter= Splitter()
xtrain, xtest, ytrain, ytest= splitter.get_all_subsets()
#out put: coefficients + score dictionary
def logregression(key_name):
    lr= LogisticRegression(penalty="l1", solver= "liblinear")
    xTrain= xtrain[key_name]
    yTrain= ytrain[key_name]
    xTest= xtest[key_name]
    yTest= ytest[key_name]
    lr.fit(xTrain,yTrain)
    AUC= metrics.roc_auc_score(y_true= yTest, y_score= lr.predict_proba(xTest)[:, 1])
    precision, recall, thresholds = metrics.precision_recall_curve(yTest, lr.predict_proba(xTest)[:, 1])
    AUPRC= metrics.auc(recall, precision)
    F1= metrics.f1_score(yTest, lr.predict(xTest))
    fpr, tpr, thr = metrics.roc_curve(yTest, y_score=lr.predict_proba(xTest)[:, 1])
    return lr.coef_, {"AUC": AUC, "AUPRC": AUPRC, "F1": F1}, {"fpr": fpr, "tpr":tpr}
def main(): 
    coef, score, fpr_tpr= logregression("Occupation_Student_M")
    print("coef:\n", coef)
    print("score:\n", score)
    #print("fpr & tpr:\n", fpr_tpr)

if __name__ == "__main__":
    main()