import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from sklearn.model_selection import train_test_split


def find_subgroups(dcol, df, gender):
    label = []
    subgroup = []
    for c in dcol:
        d = df[df[c]==1]
        l = gender+"/"+c
        d.drop(columns=dcol, inplace=True)
        subgroup.append(d)
        label.append(l)

        print(l)
        print(d.shape)
    
    return label, subgroup

def split_data(df):
    # target var used
    target_var = "Growing_stress_bin"
    y_values = df[target_var]

    x_values = df.drop(columns=[target_var])

    xTrain, xTest, yTrain, yTest = train_test_split(x_values, y_values, train_size=0.75, shuffle=True, stratify = y_values, random_state=100) #
    print("-------------TRAIN-TEST SPLITS--------------------------")
    print("X Train: ", xTrain.shape)
    print("Y Train: ", yTrain.shape)

    print("X Test: ", xTest.shape)
    print("Y Test: ", yTest.shape)
    return xTrain, xTest, yTrain, yTest

def logregression(xTrain, xTest, yTrain, yTest):
    lr= LogisticRegression(penalty="l1", solver= "liblinear")

    lr.fit(xTrain,yTrain)
    AUC= metrics.roc_auc_score(y_true= yTest, y_score= lr.predict_proba(xTest)[:, 1])
    precision, recall, thresholds = metrics.precision_recall_curve(yTest, lr.predict_proba(xTest)[:, 1])
    AUPRC= metrics.auc(recall, precision)
    F1= metrics.f1_score(yTest, lr.predict(xTest))
    fpr, tpr, thr = metrics.roc_curve(yTest, y_score=lr.predict_proba(xTest)[:, 1])
    return lr.coef_, {"AUC": AUC, "AUPRC": AUPRC, "F1": F1}, {"fpr": fpr, "tpr":tpr}

def main(): 
    df = pd.read_csv("clean-data/4-16-firstclean.csv")

    # Create subgroups in based on following criteria

    print("Creating Subgroups...")
    
    # Split based on gender
    male_df = df[df['Gender_bin'] == 0]
    male_df.drop(columns=['Gender_bin'], inplace=True)
    female_df = df[df['Gender_bin'] == 1]
    female_df.drop(columns=['Gender_bin'], inplace=True)

    # Split based on Occupation
    d_col = ['Occupation_Corporate', 'Occupation_Business', 'Occupation_Housewife', 'Occupation_Others', 'Occupation_Student']
    lab0, arr0_df = find_subgroups(d_col, male_df, "Male")
    lab1, arr1_df = find_subgroups(d_col, female_df, "Female")

    # Run Male subgroup data
    for label, arr in zip(lab0, arr0_df):
        # test train split
        xTrain, xTest, yTrain, yTest = split_data(arr)
        


        coef, score, fpr_tpr= logregression(xTrain, xTest, yTrain, yTest)
        print("coef:\n", coef)
        print("score:\n", score)
        print("fpr & tpr:\n", fpr_tpr)

if __name__ == "__main__":
    main()