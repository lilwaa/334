import pandas as pd 
import numpy as np 
import shap
from sklearn import metrics
from templatesplit import Splitter
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_recall_curve, auc, roc_curve
import time

def train_NN(params, xTrain, yTrain):
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
    hdls = int(xTrain.shape[1]*2/3) + 1
    model = MLPClassifier(alpha=params["alpha"], solver=params["solver"], hidden_layer_sizes=(hdls,hdls))
    model = model.fit(X=xTrain.to_numpy(), y=yTrain.to_numpy().flatten())
    return model
def predict_NN(model, xTest, yTest):
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
    
    y_score= model.predict_proba(xTest)
    #auprc
    AUC = metrics.roc_auc_score(y_true=yTest, y_score=y_score, multi_class="ovr", average="macro")
    #auc
    #TODO: figure out where "classes" comes from
    classes = [0, 1, 2]
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
    F1 = metrics.f1_score(yTest, model.predict(xTest), average= "macro")
    """ AUC = roc_auc_score(yTest, predProbTest)

    testPrecision, testRecall, testThreshold = precision_recall_curve(yTest, predProbTest)
    AUPRC = auc(testRecall, testPrecision)

    F1 = f1_score(y_pred=yHat, y_true=yTest)
    """
    #fpr, tpr, thresholds = roc_curve(y_true=yTest, y_score=y_score)
    
    acc = accuracy_score(y_true=yTest, y_pred=yHat)
    
    return yHat, {"Accuracy" : acc, "AUC": AUC, "AUPRC": AUPRC, "F1": F1}


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
    pgrid = {"solver": ['lbfgs', 'adam'], "alpha": [0.00001, 0.0001]}
    clf = MLPClassifier()
    xTrain = xTrain.to_numpy()
    yTrain = yTrain.to_numpy().flatten()
    clf.fit(X=xTrain, y=yTrain)
    gscv = GridSearchCV(estimator=clf, cv=5, param_grid=pgrid, scoring='f1_macro')
    gscv = gscv.fit(X=xTrain, y=yTrain)
    return gscv.best_params_

def plotting(xTrain, model, xTest):
    xt = shap.sample(xTrain, 300)
    explainer = shap.KernelExplainer(model.predict,xt)
    shap_values = explainer.shap_values(xTest,nsamples=500)
    shap.summary_plot(shap_values,xTest,feature_names=xTest.columns)

def main():
    splitter = Splitter() 
    xtrain, xtest, ytrain, ytest = splitter.get_all_subsets()

    occGen = ['Occupation_Business_F', 'Occupation_Corporate_F', 'Occupation_Housewife_F',
       'Occupation_Others_F', 'Occupation_Student_F','Occupation_Business_M', 'Occupation_Corporate_M', 'Occupation_Housewife_M',
       'Occupation_Others_M', 'Occupation_Student_M']
    
    predictions = [0] * len(occGen)
    metricsDict = [0]*  len(occGen)
    totaltime = [0] * len(occGen)
    models = [0]*  len(occGen)

    for i in range( len(occGen)): 
        name = occGen[i]
        print(name)
        xTrain = xtrain[name]
        yTrain = ytrain[name]
        bp = eval_gridsearch(xTrain, yTrain)
        start = time.time()
        model = train_NN(bp, xTrain, yTrain)
        models[i] = model

        xTest = xtest[name]
        yTest = ytest[name]

        print(type(xTrain))
        predictions[i], metricsDict[i] = predict_NN(model, xTest, yTest)
        timeElapsed = time.time() - start
        totaltime[i] = timeElapsed
        print (metricsDict[i])
        print("TOTAL TIME ELAPSED: " + str(timeElapsed))
        print(model.get_params)
        #plotting(model=models[i], xTest=xTest,xTrain=xTrain)
        
if __name__ == "__main__":
    main()

