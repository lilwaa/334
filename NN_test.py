import pandas as pd
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from templatesplit import Splitter

splitter= Splitter()
xtrain, xtest, ytrain, ytest= splitter.get_all_subsets()
#out put: coefficients + score dictionary
def logregression(key_name):
    #lr= LogisticRegression()
    lr= MLPClassifier()
    xTrain= xtrain[key_name].to_numpy()
    yTrain= ytrain[key_name].to_numpy()
    xTest= xtest[key_name].to_numpy()
    yTest= ytest[key_name].to_numpy()
    lr.fit(xTrain,yTrain)
    AUC= metrics.roc_auc_score(y_true= yTest, y_score= lr.predict_proba(xTest)[:, 1])
    precision, recall, thresholds = metrics.precision_recall_curve(yTest, lr.predict_proba(xTest)[:, 1])
    AUPRC= metrics.auc(recall, precision)
    F1= metrics.f1_score(yTest, lr.predict(xTest))
    #return lr.coef_, {"AUC": AUC, "AUPRC": AUPRC, "F1": F1}
    return {"AUC": AUC, "AUPRC": AUPRC, "F1": F1}
def main(): 
    #coef, score= logregression("Occupation_Student_M")
    score= logregression("Occupation_Student_M")
    #print("coef:\n", coef)
    print("score:\n", score)

if __name__ == "__main__":
    main()