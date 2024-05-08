from sklearn.model_selection import GridSearchCV
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from templatesplit import Splitter
import time
import seaborn as sns
import matplotlib.pyplot as plt
splitter= Splitter()
xtrain, xtest, ytrain, ytest= splitter.get_all_subsets()
coef_dic= {}
def logregression(key_name):
    # Define logistic regression model
    lr = LogisticRegression(solver="liblinear", multi_class="ovr")
    
    # Define hyperparameters to search 
    param_grid = {'penalty': ['l1', 'l2'], 'C': [0.1, 1, 10], "tol": [1e-3, 1e-4, 1e-5]}
    
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
    start= time.time()
    best_lr.fit(xTrain, yTrain)
    best_lr.predict(xTest)
    end = time.time()
    Time= end -start
    y_score= best_lr.predict_proba(xTest)
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
    AUPRC = np.mean(auprc_scores)
    #-----------------------------------------------------------------------------------------
    #f1
    F1= metrics.f1_score(yTest, best_lr.predict(xTest), average= "macro")
    return  best_lr.coef_, {"AUC": AUC, "AUPRC": AUPRC, "F1": F1, "Best_Params": grid_search.best_params_, "Time": Time}
    #return best_lr.coef_
def draw_barplot(df):
    plt.figure(figsize=(70,10))
    data_melted = df.melt(id_vars='Class', var_name='x', value_name='y')
    sns.barplot(data=data_melted, x='x', y='y', hue='Class')
    plt.show()
def draw_heatmap(df,label):
    pivot_data = df.pivot_table(index='Class')
    # Plot
    plt.figure(figsize=(10, 6))
    sns.heatmap(pivot_data, cmap='coolwarm', annot=True, fmt=".3f", linewidths=.5)
    plt.title(f"Heatmap for Correlations by Logistic Regression with {label}")
    plt.tight_layout()
    plt.show()

# Example usage
def main(): 
    column_names = pd.read_csv("xtrain/Occupation_Others_F.csv").columns
    for subgroup in xtrain.keys():
        print("-----------------------------------------------------")
        print(subgroup)
        coef, score= logregression(subgroup)
        print("score:\n", score)
        coef_dic_no = coef[0]
        coef_dic_maybe = coef[1]
        coef_dic_yes = coef[2]
    
        coef_df_yes = pd.DataFrame(coef_dic_yes, index=column_names).transpose()
        coef_df_yes["Class"] = "Yes"
    
        coef_df_no = pd.DataFrame(coef_dic_no, index=column_names).transpose()
        coef_df_no["Class"] = "No"
    
        coef_df_maybe = pd.DataFrame(coef_dic_maybe, index=column_names).transpose()
        coef_df_maybe["Class"] = "Maybe"
    
        data = pd.concat([coef_df_yes, coef_df_maybe, coef_df_no])
        draw_heatmap(data, subgroup)

    
    #coef_df.to_csv("lr_coeff.csv", index=False)
    
      
    
if __name__ == "__main__":
    main()