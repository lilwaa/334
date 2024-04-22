from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn import metrics
from templatesplit import Splitter
splitter= Splitter()
xtrain, xtest, ytrain, ytest= splitter.get_all_subsets()
def mlp_classifier(key_name):
    # Define MLP classifier model
    best_mlp = MLPClassifier(max_iter=100)
    '''
    # Define hyperparameters to search
    param_grid = {
        'hidden_layer_sizes': [(100,), (50, 50), (50, 25, 10)],
        'activation': ['relu', 'tanh', 'logistic'],
        'alpha': [0.0001, 0.001, 0.01],
        'solver': ['adam', 'lbfgs'],
        'learning_rate': ['constant', 'invscaling', 'adaptive']
    }
    
    # Perform grid search with cross-validation
    grid_search = GridSearchCV(mlp, param_grid, cv=5, scoring='roc_auc')
    
    # Fit the model
    xTrain = xtrain[key_name]
    yTrain = ytrain[key_name]
    xTest = xtest[key_name]
    yTest = ytest[key_name]
    grid_search.fit(xTrain, yTrain)
    
    # Get the best model from grid search
    best_mlp = grid_search.best_estimator_
    
    # Evaluate the model
    '''
    xTrain = xtrain[key_name]
    yTrain = ytrain[key_name]
    xTest = xtest[key_name]
    yTest = ytest[key_name]
    best_mlp.fit(xTrain, yTrain)
    AUC = metrics.roc_auc_score(y_true=yTest, y_score=best_mlp.predict_proba(xTest)[:, 1])
    precision, recall, thresholds = metrics.precision_recall_curve(yTest, best_mlp.predict_proba(xTest)[:, 1])
    AUPRC = metrics.auc(recall, precision)
    F1 = metrics.f1_score(yTest, best_mlp.predict(xTest))
    
    #return best_mlp.coefs_, {"AUC": AUC, "AUPRC": AUPRC, "F1": F1, "Best_Params": grid_search.best_params_}
    return {"AUC": AUC, "AUPRC": AUPRC, "F1": F1}

# Example usage
def main(): 
    scores = mlp_classifier("Occupation_Student_F")
    #print("Weights (coefficients) of the MLP model:\n", coefs)
    print("Scores:\n", scores)

if __name__ == "__main__":
    main()
