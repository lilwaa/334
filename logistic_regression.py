import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from templatesplit import Splitter

splitter = Splitter()
xtrain, xtest, ytrain, ytest = splitter.get_all_subsets()

def logregression(key_name):
    lr = LogisticRegression()
    xTrain = xtrain[key_name].to_numpy().reshape(-1, 1)
    yTrain = ytrain[key_name].to_numpy().ravel()
    xTest = xtest[key_name].to_numpy().reshape(-1, 1)
    yTest = ytest[key_name].to_numpy().ravel()
    lr.fit(xTrain, yTrain)
    y_proba = lr.predict_proba(xTest)[:, 1]
    AUC = metrics.roc_auc_score(y_true=yTest, y_score=y_proba)
    precision, recall, thresholds = metrics.precision_recall_curve(yTest, y_proba)
    AUPRC = metrics.auc(recall, precision)
    F1 = metrics.f1_score(yTest, lr.predict(xTest))
    return lr.coef_, {"AUC": AUC, "AUPRC": AUPRC, "F1": F1}

def main(): 
    coef, score = logregression("Occupation_Student_M")
    print("coef:\n", coef)
    print("score:\n", score)

if __name__ == "__main__":
    main()
