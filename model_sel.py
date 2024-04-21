#this file is for finding the optimal hyperparameters with gridsearch and random search
import argparse
import json
import pandas as pd
import time
from sklearn import metrics, model_selection, neighbors, neural_network, linear_model, ensemble



def eval_gridsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn classifier and a parameter grid to search,
    choose the optimal parameters from pgrid using Grid Search CV
    and train the model using the training dataset and evaluate the
    performance on the test dataset.

    Parameters
    ----------
    clf : sklearn.ClassifierMixin
        The sklearn classifier model 
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
    resultDict: dict
        A Python dictionary with the following 4 keys,
        "AUC", "AUPRC", "F1", "Time" and the values are the floats
        associated with them for the test set.
    roc : dict
        A Python dictionary with 2 keys, fpr, and tpr, where
        each of the values are 1-d numpy arrays of the fpr and tpr
        associated with different thresholds. You should be able to use 
        this to plot the ROC for the model performance on the test curve.
    bestParams: dict
        A Python dictionary with the best parameters chosen by your
        GridSearch. The values in the parameters should be something
        that was in the original pgrid.
    """
    start = time.time()
    grid_search= model_selection.GridSearchCV(estimator= clf,param_grid= pgrid, scoring="roc_auc")
    grid_search.fit(xTrain, yTrain)

    AUC= metrics.roc_auc_score(y_true= yTest, y_score= grid_search.predict_proba(xTest)[:, 1])
    precision, recall, thresholds = metrics.precision_recall_curve(yTest, grid_search.predict_proba(xTest)[:, 1])
    AUPRC= metrics.auc(recall, precision)
    F1= metrics.f1_score(yTest, grid_search.predict(xTest))
    fpr, tpr, thr = metrics.roc_curve(yTest, y_score=grid_search.predict_proba(xTest)[:, 1])

    parameters_dict= grid_search.best_params_
    print(parameters_dict)
    timeElapsed = time.time() - start
    return {"AUC": AUC, "AUPRC": AUPRC, "F1": F1, "Time": timeElapsed}, {"fpr": fpr, "tpr": tpr}, parameters_dict


def eval_randomsearch(clf, pgrid, xTrain, yTrain, xTest, yTest):
    """
    Given a sklearn classifier and a parameter grid to search,
    choose the optimal parameters from pgrid using Random Search CV
    and train the model using the training dataset and evaluate the
    performance on the test dataset. The random search cv should try
    at most 33% of the possible combinations.

    Parameters
    ----------
    clf : sklearn.ClassifierMixin
        The sklearn classifier model 
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
    resultDict: dict
        A Python dictionary with the following 4 keys,
        "AUC", "AUPRC", "F1", "Time" and the values are the floats
        associated with them for the test set.
    roc : dict
        A Python dictionary with 2 keys, fpr, and tpr, where
        each of the values are lists of the fpr and tpr associated
        with different thresholds. You should be able to use this
        to plot the ROC for the model performance on the test curve.
    bestParams: dict
        A Python dictionary with the best parameters chosen by your
        GridSearch. The values in the parameters should be something
        that was in the original pgrid.
    """
    start = time.time()
    rdm_search= model_selection.RandomizedSearchCV(estimator= clf,param_distributions= pgrid, n_iter= max(1, min(3, len(pgrid))), scoring="roc_auc")
    rdm_search.fit(xTrain, yTrain)

    AUC= metrics.roc_auc_score(y_true= yTest, y_score= rdm_search.predict_proba(xTest)[:, 1])
    precision, recall, thresholds = metrics.precision_recall_curve(yTest, rdm_search.predict_proba(xTest)[:, 1])
    AUPRC= metrics.auc(recall, precision)
    F1= metrics.f1_score(yTest, rdm_search.predict(xTest))
    fpr, tpr, thr = metrics.roc_curve(yTest, y_score=rdm_search.predict_proba(xTest)[:, 1])
    
    parameters_dict= rdm_search.best_params_
    timeElapsed = time.time() - start
    return {"AUC": AUC, "AUPRC": AUPRC, "F1": F1, "Time": timeElapsed}, {"fpr": fpr, "tpr": tpr}, parameters_dict



def eval_searchcv(clfName, clf, clfGrid,
                  xTrain, yTrain, xTest, yTest,
                  perfDict, rocDF, bestParamDict):
    # evaluate grid search and add to perfDict
    cls_perf, cls_roc, gs_p  = eval_gridsearch(clf, clfGrid, xTrain,
                                               yTrain, xTest, yTest)
    perfDict[clfName + " (Grid)"] = cls_perf
    # add to ROC DF
    rocRes = pd.DataFrame(cls_roc)
    rocRes["model"] = clfName
    rocDF = pd.concat([rocDF, rocRes], ignore_index=True)
    # evaluate random search and add to perfDict
    clfr_perf, _, rs_p  = eval_randomsearch(clf, clfGrid, xTrain,
                                            yTrain, xTest, yTest)
    perfDict[clfName + " (Random)"] = clfr_perf
    bestParamDict[clfName] = {"Grid": gs_p, "Random": rs_p}
    print(bestParamDict)
    return perfDict, rocDF, bestParamDict


def get_parameter_grid(mName):
    """
    Given a model name, return the parameter grid associated with it

    Parameters
    ----------
    mName : string
        name of the model (e.g., DT, KNN, LR (None))

    Returns
    -------0
    pGrid: dict
        A Python dictionary with the appropriate parameters for the model.
        The dictionary should have at least 2 keys and each key should have
        at least 2 values to try.
    """
    models= {}
    #random forest
    models["RF"]={"criterion":["gini", "entropy"],
                  "max_depth":[5, 10],
                  "min_samples_leaf": [5, 10]}
    #logistic regression
    models["LR (None)"]= {"max_iter": [50, 100, 150],
                        "tol": [1e-3, 1e-4, 1e-5],
                        }
    models["LR (L1)"]= {"max_iter": [50, 100, 150],
                        "tol": [1e-3, 1e-4, 1e-5],
                        "C": [0.1, 1, 10]}
    models["LR (L2)"]= {"max_iter": [50, 100, 150],
                        "tol": [1e-3, 1e-4, 1e-5],
                        "C": [0.1, 1, 10]}
    #neural network
    models["NN"]= {"learning_rate": ["constant", "invscaling", "adaptive"],
                   "max_iter": [150, 200, 250],
                   "tol": [1e-3, 1e-4, 1e-5]
                   }
    return models.get(mName, {})


def main():
    # set up the program to take in arguments from the command line
    parser = argparse.ArgumentParser()
    parser.add_argument("--xTrain",
                        default="space_trainx.csv",
                        help="filename for features of the training data")
    parser.add_argument("--yTrain",
                        default="space_trainy.csv",
                        help="filename for labels associated with training data")
    parser.add_argument("--xTest",
                        default="space_testx.csv",
                        help="filename for features of the test data")
    parser.add_argument("--yTest",
                        default="space_testy.csv",
                        help="filename for labels associated with the test data")
    parser.add_argument("rocOutput",
                         help="csv filename for ROC curves")
    parser.add_argument("bestParamOutput",
                         help="json filename for best parameter")
    args = parser.parse_args()
    # load the train and test data
    xTrain = pd.read_csv(args.xTrain).to_numpy()
    yTrain = pd.read_csv(args.yTrain).to_numpy().flatten()
    xTest = pd.read_csv(args.xTest).to_numpy()
    yTest = pd.read_csv(args.yTest).to_numpy().flatten()

    # preprocess the data

    perfDict = {}
    rocDF = pd.DataFrame()
    bestParamDict = {}

    print("Tuning Random Forest --------")
    # Compare Decision Tree
    dtName = "RF"
    dtGrid = get_parameter_grid(dtName)
    # fill in
    dtClf = ensemble.RandomForestClassifier()
    perfDict, rocDF, bestParamDict = eval_searchcv(dtName, dtClf, dtGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    print("Tuning Unregularized Logistic Regression --------")
    # logistic regression (unregularized)
    unregLrName = "LR (None)"
    unregLrGrid = get_parameter_grid(unregLrName)
    # fill in
    lrClf = linear_model.LogisticRegression(penalty=None)
    perfDict, rocDF, bestParamDict = eval_searchcv(unregLrName, lrClf, unregLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    # logistic regression (L1)
    print("Tuning Logistic Regression (Lasso) --------")
    lassoLrName = "LR (L1)"
    lassoLrGrid = get_parameter_grid(lassoLrName)
    # fill in
    lassoClf = linear_model.LogisticRegression(penalty="l1", solver="liblinear")
    perfDict, rocDF, bestParamDict = eval_searchcv(lassoLrName, lassoClf, lassoLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    # Logistic regression (L2)
    print("Tuning Logistic Regression (Ridge) --------")
    ridgeLrName = "LR (L2)"
    ridgeLrGrid = get_parameter_grid(ridgeLrName)
    # fill in
    ridgeClf = linear_model.LogisticRegression(penalty="l2")
    perfDict, rocDF, bestParamDict = eval_searchcv(ridgeLrName, ridgeClf, ridgeLrGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    
    # neural networks
    print("Tuning neural networks --------")
    nnName = "NN"
    nnGrid = get_parameter_grid(nnName)
    # fill in
    nnClf = neural_network.MLPClassifier()
    perfDict, rocDF, bestParamDict = eval_searchcv(nnName, nnClf, nnGrid,
                                                   xTrain, yTrain, xTest, yTest,
                                                   perfDict, rocDF, bestParamDict)
    
    perfDF = pd.DataFrame.from_dict(perfDict, orient='index')
    print(perfDF)
    # save roc curves to data
    rocDF.to_csv(args.rocOutput, index=False)
    # store the best parameters
    with open(args.bestParamOutput, 'w') as f:
        json.dump(bestParamDict, f)


if __name__ == "__main__":
    main()
