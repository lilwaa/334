import pandas as pd 
import numpy as np 
import shap
from templatesplit import Splitter
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
from sklearn.neighbors import KNeighborsClassifier as KNN
def train_KNN(params, xTrain, yTrain):
    """
        Train the NN using the data.

        Parameters
        ----------
        params : array with shape m 

        xFeat : nd-array with shape n x d
            Training data 
        y : 1d array with shape n
            Array of responses associated with training data.

        Returns
        -------
        model : object
            Keys represent the epochs and values the number of mistakes
    """
    model = KNN()
    model = model.fit(X=xTrain.to_numpy(), y=yTrain.to_numpy().flatten())
    return model
def predict_KNN(model, xTest, yTest):
    """
        Predict the data using the given NN model.

        Parameters
        ----------
        model : a trained MLPClassifier 

        xTest : nd-array with shape n x d
            Testing data 
        yTest : 1d array with shape n
            Array of responses associated with testing data.

        Returns
        -------
        predictions : 1d array with shape n
            Array of prediction based on the testing data
        metricsDict : dict
            A Python dictionary with the following 4 keys,
            "Accuracy", "AUC", "AUPRC", "F1" and the values are the floats
            associated with them for the test set.
        roc : dict
            A Python dictionary with 2 keys, fpr, and tpr, where
            each of the values are lists of the fpr and tpr associated
            with different thresholds. You should be able to use this
            to plot the ROC for the model performance on the test curve.
    """
    yHat = model.predict(xTest.to_numpy())
    yTest = yTest.to_numpy().flatten()
    predProbTest= model.predict_proba(xTest)[:,1]
    
    AUC = roc_auc_score(yTest, predProbTest)

    testPrecision, testRecall, testThreshold = precision_recall_curve(yTest, predProbTest)
    AUPRC = auc(testRecall, testPrecision)

    F1 = f1_score(y_pred=yHat, y_true=yTest)

    #fpr, tpr, thresholds = roc_curve(y_true=yTest, y_score=predProbTest)
    acc = accuracy_score(y_true=yTest, y_pred=yHat)
    
    return yHat, {"Accuracy" : acc, "AUC": AUC, "AUPRC": AUPRC, "F1": F1}#, {"fpr": fpr, "tpr": tpr}


def eval_gridsearch(xTrain, yTrain):
    """
    Given a parameter grid to search, choose the optimal parameters 
    from pgrid using Grid Search CV and train the Neural Network 
    using the training dataset and evaluate the performance on 
    the test dataset.

    Parameters
    ----------
    pgrid : dict
        The dictionary of parameters to tune for in the model
    xTrain : nd-array with shape (n, d)
        Training data
    yTrain : 1d array with shape (n, )
        Array of labels associated with training data
    xTest : nd-array with shape (m, d)
        Test data
    yTest : 1d array with shape m
        Array of labels associated with test data.

    Returns
    -------
    bestParams: dict
        A Python dictionary with the best parameters chosen by the
        GridSearch. 
    """
    pgrid = {"n_neighbors": [5, 20, 50], "weights": ['uniform', 'distance'] }
    clf = KNN()
    xTrain = xTrain.to_numpy()
    yTrain = yTrain.to_numpy().flatten()
    clf.fit(X=xTrain, y=yTrain)
    gscv = GridSearchCV(estimator=clf, param_grid=pgrid)
    gscv = gscv.fit(X=xTrain, y=yTrain)
    return gscv.best_params_

def plotting(xTrain, model, xTest):
    xt = shap.sample(xTrain, 100)
    explainer = shap.KernelExplainer(model.predict,xt)
    shap_values = explainer.shap_values(xTest,nsamples=100)
    shap.summary_plot(shap_values,xTest,feature_names=xTest.columns)

def main():
    splitter = Splitter() #Note: you need to go to the templatesplit.py to change the target column if needed
    xtrain, xtest, ytrain, ytest = splitter.get_all_subsets()

    occGen = ['Occupation_Business_F', 'Occupation_Corporate_F', 'Occupation_Housewife_F',
       'Occupation_Others_F', 'Occupation_Student_F','Occupation_Business_M', 'Occupation_Corporate_M', 'Occupation_Housewife_M',
       'Occupation_Others_M', 'Occupation_Student_M']
    
    predictions = [0] * len(occGen)
    metricsDict = [0]*  len(occGen)
    roc = [0]*  len(occGen)
    models = [0]*  len(occGen)

    for i in range( len(occGen)): 
        name = occGen[i]
        print(name)
        xTrain = xtrain[name]
        yTrain = ytrain[name]

        bp = eval_gridsearch(xTrain, yTrain)

        model = KNN()
        model = train_KNN(bp, xTrain, yTrain)
        models[i] = model
    
        xTest = xtest[name]
        yTest = ytest[name]

        print(type(xTrain))
        predictions[i], metricsDict[i] = predict_KNN(model, xTest, yTest)
        print (metricsDict[i])
        
        
if __name__ == "__main__":
    main()

