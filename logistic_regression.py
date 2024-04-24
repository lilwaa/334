from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from templatesplit import Splitter
import matplotlib.pyplot as plt
splitter= Splitter()
xtrain, xtest, ytrain, ytest= splitter.get_all_subsets()
coef_dic= {}
def logregression(key_name):
    # Define logistic regression model
    lr = LogisticRegression(solver="liblinear", multi_class="ovr")
    
    # Define hyperparameters to search
    param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10]}
    
    # Perform grid search 
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1_macro')
    
    # Fit the model
    xTrain = xtrain[key_name]
    yTrain = ytrain[key_name]
    xTest = xtest[key_name]
    yTest = ytest[key_name]
    grid_search.fit(xTrain, yTrain)
    
    # Get the best model from grid search
    best_lr = grid_search.best_estimator_
    classes = np.unique(yTrain)
    #coef_interpretation = {classes[i]: coef for i, coef in enumerate(best_lr.coef_)}
    y_score= best_lr.predict_proba(xTest)
    '''
    # Evaluate the model
    AUC = metrics.roc_auc_score(y_true=yTest, y_score=y_score, multi_class="ovr", average="macro")
    precision, recall, thresholds = metrics.precision_recall_curve(yTest, y_score)
    AUPRC = metrics.auc(recall, precision)
    F1 = metrics.f1_score(yTest, y_score)
    '''
    '''
    #prc
    '''
    #auc
    AUC = metrics.roc_auc_score(y_true=yTest, y_score=y_score, multi_class="ovr", average="macro")
    #----------------------------------------------------------------------------------------
    #auprc
    #precision = dict()
    #recall = dict()
    auprc_scores = []
    for i, label in enumerate(classes):
        # Treat each class as binary (1 if class i, 0 otherwise)
        y_binary = (yTest == label).astype(int)
        precision, recall, _ = metrics.precision_recall_curve(y_binary, y_score[:, i])
        auprc = metrics.auc(recall, precision)
        auprc_scores.append(auprc)
        #precision[label], recall[label], _ = metrics.precision_recall_curve(y_binary, y_score[:, i])
        #lt.plot(recall[label], precision[label], lw=2, label='class {}'.format(label))
    AUPRC = np.mean(auprc_scores)
    #-----------------------------------------------------------------------------------------
    #f1
    F1= metrics.f1_score(yTest, best_lr.predict(xTest), average= "macro")
    '''
    plt.xlabel("recall")
    plt.ylabel("precision")
    plt.legend(loc="best")
    plt.title("precision vs. recall curve")
    plt.show()
    '''
    
    
    
    
    return  {"AUC": AUC, "AUPRC": AUPRC, "F1": F1, "Best_Params": grid_search.best_params_}
    #return best_lr.coef_

# Example usage
def main(): 
    '''
     for subgroup in xtrain.keys():
        print("-----------------------------------------------------")
        print(subgroup)
        coef, score= logregression(subgroup)
        coef_dic[subgroup]=coef[0]
        print("coef:\n", coef)
        print("score:\n", score)

    column_names = pd.read_csv("xtrain/Occupation_Others_F.csv").columns
    coef_df = pd.DataFrame(coef_dic, index=column_names)
    coef_df = coef_df.transpose()
    coef_df.to_csv("lr_coeff.csv", index=False)
    '''
    print(logregression("Occupation_Business_F"))
      
    
if __name__ == "__main__":
    main()