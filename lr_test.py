from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from split_test import Splitter
splitter= Splitter()
xtrain, xtest, ytrain, ytest= splitter.get_all_subsets()
coef_dic= {}
def logregression(key_name):
    # Define logistic regression model
    lr = LogisticRegression(solver="liblinear")
    
    # Define hyperparameters to search
    param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]}
    
    # Perform grid search 
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='roc_auc')
    
    # Fit the model
    xTrain = xtrain[key_name]
    yTrain = ytrain[key_name]
    xTest = xtest[key_name]
    yTest = ytest[key_name]
    grid_search.fit(xTrain, yTrain)
    
    # Get the best model from grid search
    best_lr = grid_search.best_estimator_
    
    # Evaluate the model
    AUC = metrics.roc_auc_score(y_true=yTest, y_score=best_lr.predict_proba(xTest)[:, 1])
    precision, recall, thresholds = metrics.precision_recall_curve(yTest, best_lr.predict_proba(xTest)[:, 1])
    AUPRC = metrics.auc(recall, precision)
    F1 = metrics.f1_score(yTest, best_lr.predict(xTest))
    
    return best_lr.coef_, {"AUC": AUC, "AUPRC": AUPRC, "F1": F1, "Best_Params": grid_search.best_params_}

# Example usage
def main(): 
    for subgroup in xtrain.keys():
        print("-----------------------------------------------------")
        print(subgroup)
        coef, score= logregression(subgroup)
        coef_dic[subgroup]=coef[0]
        print("coef:\n", coef)
        print("score:\n", score)

    column_names = pd.read_csv("var/xtrain/Occupation_Others_F.csv").columns
    coef_df = pd.DataFrame(coef_dic, index=column_names)
    coef_df = coef_df.transpose()
    coef_df.to_csv("lr_coeff_var.csv", index=False)
    
    
if __name__ == "__main__":
    main()