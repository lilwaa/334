import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import timeit

from sklearn.datasets import make_multilabel_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, precision_recall_curve, auc, f1_score, roc_curve, roc_auc_score
from sklearn.inspection import permutation_importance
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import LabelBinarizer

# models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRFClassifier

# Random Forest: Bagged Trees!
class RandomForest(object):
    # model
    forest = None   

    # hyperparameters
    n_estimators = 0            # number of trees
    criterion = None            # calculate node impurity: GINI or entropy
    max_depth = None            # max levels of tree
    min_samples_leaf = None     # min leaf sample
    ccp_alpha = 0.0             # amount of pruning

    # initialize
    def __init__(self, n_estimators, criterion, max_depth, min_samples_leaf, ccp_alpha):
        self.n_estimators = n_estimators
        self.criterion = criterion
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.ccp_alpha = ccp_alpha

    # train Random Forest
    def train(self, xTrain, yTrain):
        self.forest = RandomForestClassifier(n_estimators=self.n_estimators, criterion=self.criterion, max_depth=self.max_depth,
                                         min_samples_leaf=self.min_samples_leaf, ccp_alpha=self.ccp_alpha)
        self.forest.fit(xTrain, yTrain)
        
        return self.forest

    # predict Random Forest
    def predict(self, xTest, yTest):
        if self.forest is None:
            raise ValueError("Forest has not been trained yet. Please train the forest using train method.")

        # Create predictions / probbilities
        predictions = self.forest.predict(xTest)
        yScore = self.forest.predict_proba(xTest)

        # Find accuracy
        accuracy = accuracy_score(yTest, predictions)

        # Find F1 score
        f1 = f1_score(yTest, predictions, average='macro')

        # Compute AUROC
        auroc = roc_auc_score(y_true=yTest, y_score = yScore, multi_class="ovr", average="macro")
  
        # Compute AUPRC
        auprc_scores = []
        classes = np.unique(yTest)
        for i, label in enumerate(classes):
            y_binary = (yTest == label).astype(int)
            precision, recall, _ = precision_recall_curve(y_binary, yScore[:, i])
            auprc = auc(recall, precision) 
            auprc_scores.append(auprc)
    
        auprc = np.mean(auprc_scores)

        # Compute ROC curves for each class
        roc = 0

        return accuracy, f1, auroc, auprc, roc
    
    # find features importance
    def features(self, feature_names, k, xTest, yTest):
        if self.forest is None:
            raise ValueError("Forest has not been trained yet. Please train the forest using train method.")

        # feature importance
        importances = self.forest.feature_importances_
        forest_ft = pd.Series(importances, index=feature_names)
        top_k = forest_ft.sort_values(ascending=False).head(k)

        result = permutation_importance(self.forest, xTest, yTest, n_repeats=10, random_state=42, n_jobs=2)
        forest_importances = pd.Series(result.importances_mean, index=feature_names)

        return top_k, forest_ft.sort_values(ascending=False), forest_importances, result

# Gradient Boosted Trees!
class GradientBoostedTrees(object):
    # model
    gbt = None   

    # hyperparameters
    n_estimators = 0            # number of trees
    learning_rate = 0.1         # learning rate
    max_depth = 3               # maximum depth
    min_samples_leaf = 1        # minimum leaf samples

    # initialize
    def __init__(self, n_estimators, learning_rate, max_depth, min_samples_leaf):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf

    # train gradient boosted trees
    def train(self, xTrain, yTrain):
        self.gbt = GradientBoostingClassifier(n_estimators=self.n_estimators, learning_rate=self.learning_rate,
                                               max_depth=self.max_depth, min_samples_leaf=self.min_samples_leaf)
        self.gbt.fit(xTrain, yTrain)

        return self.gbt

    # predict gradient boosted trees
    def predict(self, xTest, yTest):
        if self.gbt is None:
            raise ValueError("Gradient Boosted Trees model has not been trained yet. Please train the model using the train method.")

        # Create predictions / probbilities
        predictions = self.gbt.predict(xTest)
        yScore = self.gbt.predict_proba(xTest)

        # Find accuracy
        accuracy = accuracy_score(yTest, predictions)

        # Find F1 score
        f1 = f1_score(yTest, predictions, average='macro')

        # Compute AUROC
        auroc = roc_auc_score(y_true=yTest, y_score = yScore, multi_class="ovr", average="macro")
  
        # Compute AUPRC
        auprc_scores = []
        classes = np.unique(yTest)
        for i, label in enumerate(classes):
            y_binary = (yTest == label).astype(int)
            precision, recall, _ = precision_recall_curve(y_binary, yScore[:, i])
            auprc = auc(recall, precision) 
            auprc_scores.append(auprc)
    
        auprc = np.mean(auprc_scores)

        # Compute ROC curves for each class
        roc = 0

        return accuracy, f1, auroc, auprc, roc

    # find top features
    def features(self, feature_names, k, xTest, yTest):
        if self.gbt is None:
            raise ValueError("Gradient Boosted Trees model has not been trained yet. Please train the model using the train method.")

        # find feature importance
        importances = self.gbt.feature_importances_
        gbt_ft = pd.Series(importances, index=feature_names)

        top_k = gbt_ft.sort_values(ascending=False).head(k)

        result = permutation_importance(self.gbt, xTest, yTest, n_repeats=10, random_state=42, n_jobs=2)
        gbt_importances = pd.Series(result.importances_mean, index=feature_names)

        return top_k, gbt_ft.sort_values(ascending=False), gbt_importances, result

# Extreme X Gradient Boosted Trees!
class XGBoost(object):
    # Model
    xgb = None

    # Hyperparameters
    num_parallel_tree = 0              # number of trees     
    max_depth = 0                 # maximum depth
    learning_rate = 0.1           # learning rate

    # initialize model
    def __init__(self, num_parallel_tree=100, max_depth=3, learning_rate=0.1):
        self.num_parallel_tree = num_parallel_tree
        self.max_depth = max_depth
        self.learning_rate = learning_rate

    # train Extreme X Gradient
    def train(self, X_train, y_train):
        self.xgb = XGBRFClassifier(num_parallel_tree = self.num_parallel_tree, max_depth=self.max_depth, learning_rate=self.learning_rate)
        self.xgb.fit(X_train, y_train)

        return self.xgb

    # predict Extreme X Gradient
    def predict(self, xTest, yTest):
        if self.xgb is None:
            raise ValueError("XGBoost model has not been trained yet. Please train the model using train method.")

        # Create predictions / probbilities
        predictions = self.xgb.predict(xTest)
        yScore = self.xgb.predict_proba(xTest)

        # Find accuracy
        accuracy = accuracy_score(yTest, predictions)

        # Find F1 score
        f1 = f1_score(yTest, predictions, average='macro')

        # Compute AUROC
        auroc = roc_auc_score(y_true=yTest, y_score = yScore, multi_class="ovr", average="macro")
  
        # Compute AUPRC
        auprc_scores = []
        classes = np.unique(yTest)
        for i, label in enumerate(classes):
            y_binary = (yTest == label).astype(int)
            precision, recall, _ = precision_recall_curve(y_binary, yScore[:, i])
            auprc = auc(recall, precision) 
            auprc_scores.append(auprc)
    
        auprc = np.mean(auprc_scores)

        # Find roc
        roc = 0
        #fpr, tpr, _ = roc_curve(yTest, probabilities)
        #roc = {'fpr': fpr, 'tpr': tpr}

        return accuracy, f1, auroc, auprc, roc

    # find important features
    def features(self, feature_names, k):
        if self.xgb is None:
            raise ValueError("XGBoost model has not been trained yet. Please train the model using train method.")

        importance_dict = self.xgb.get_booster().get_score(importance_type='weight')
        return importance_dict


# Create subgroups
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

# Create test train split
def split_data(df):
    # target var used
    target_var = "Growing_Stress"
    y_values = df[target_var]

    x_values = df.drop(columns=[target_var])

    xTrain, xTest, yTrain, yTest = train_test_split(x_values, y_values, train_size=0.75, shuffle=True, stratify = y_values, random_state=100) #
    print("-------------TRAIN-TEST SPLITS--------------------------")
    print("X Train: ", xTrain.shape)
    print("Y Train: ", yTrain.shape)

    print("X Test: ", xTest.shape)
    print("Y Test: ", yTest.shape)
    return xTrain, xTest, yTrain, yTest

# Preprocess
def preprocess(df):

    return df

# Tune hyperparameters generating line plot
def eval_plot(xTrain, yTrain, xTest, yTest, label):
    row = []
    arr = np.arange(1, 150, 3)
    #arr = [0.0001, 0.001, 0.01, 0.1, 0.2, 0.3, 0.5, 0.8]
    for a in arr:
        print("testing ", a)
        #forest = RandomForest(19, 'gini', 9, 20, a)
        #forest.train(xTrain, yTrain)
        #gbt = GradientBoostedTrees(15, 0.1, 5, a)
        #gbt.train(xTrain, yTrain)
        xbt = XGBoost(a, 10, 0.1)
        xbt.train(xTrain, yTrain)
        accuracy = np.mean(xbt.predict(xTest, yTest))

        row.append([a, accuracy, label])
    
    return row  

# Tune hyperparameters using GridSearch
def eval_gridsearch(clf, pgrid, xTrain, yTrain):
    # grid search to find the best parameter
    grid_search = GridSearchCV(clf, pgrid, cv=5, scoring='f1_macro')
    grid_search.fit(xTrain, yTrain)

    # find best parameters
    best_params = grid_search.best_params_
    print("best parameters: ", best_params)
    
    return best_params

# Graph
def graph_hm(df, title, cmap):
    print(df)
    sns.heatmap(df, xticklabels=True)
    plt.tight_layout()
    plt.title(title)
    plt.show()


# Graph feature permutation importances
def graph_permutation(forest_importances, result):
    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

# Main function
def main():
    # Import dataset
    df = pd.read_csv("../clean-data/multiclass.csv")

    # Create subgroups in based on following criteria
    print("Creating Subgroups...")
    
    # Split based on gender, drop gender column
    male_df = df[df['Gender_bin'] == 0]
    male_df.drop(columns=['Gender_bin'], inplace=True)
    female_df = df[df['Gender_bin'] == 1]
    female_df.drop(columns=['Gender_bin'], inplace=True)

    # Subgroups
    print("---------------------------------------Sizes of subgroups--------------------------------------")
    
    # Split based on Occupation
    d_col = ['Occupation_Corporate', 'Occupation_Business', 'Occupation_Housewife', 'Occupation_Others', 'Occupation_Student']
    lab0, arr0_df = find_subgroups(d_col, male_df, "Male")
    lab1, arr1_df = find_subgroups(d_col, female_df, "Female")

    print()
    
    # Store important features
    forest_ft = {}
    gbt_ft = {}
    xgb_ft = {}

    # Store for eval plot
    trees = []

    # Store label, hyperparameters, accuracy
    forest_result = []
    gbt_result = []
    xgb_result = []
    
    # Run Male subgroup data
    for label, arr in zip(lab0, arr0_df):
        print(label)
        
        # test train split
        xTrain, xTest, yTrain, yTest = split_data(arr)
        
        # save feature names
        feature_names = xTrain.columns
        
        print("---------------Random Forest model---------------------")
        # tune hyperparameters: num_trees, criterion, max_depth, min_samples_leaf, ccp_alpha
       
        # eval_plot to find pgrid values
        #row = eval_plot(xTrain, yTrain, xTest, yTest, label)
        #trees.append(row)

        # Parameter grid for grid search (values selected based on eval_plot)
        print("tuning hyperparameters...")
        
        param_grid = {
            'n_estimators': [13, 15, 17, 19, 61, 63, 65, 70, 80, 85, 115], 
            'criterion': ['gini'],
            'max_depth': [9, 11],
            'min_samples_leaf': [20], 
            'ccp_alpha':[0]
        }

        # Perform grid search
        best_params = eval_gridsearch(RandomForestClassifier(), param_grid, xTrain, yTrain)
        print(best_params)

        # Train, test random forest
        forest = RandomForest(**best_params)

        start = timeit.default_timer()

        forest.train(xTrain, yTrain)
        accuracy, f1, auroc, auprc, roc = forest.predict(xTest, yTest)

        time_lr = timeit.default_timer() - start

        # Store result
        forest_result.append([label, best_params, accuracy])
        print("Accuracy: ", accuracy, ", AUPRC: ", auprc, ", AUROC: ", auroc, ", F1: ", f1, "time: ", time_lr)

        # Find top features via feature importance using mean decrease in impurity
        k = 3   # store k top features
        top_k, features, forest_importances, result= forest.features(feature_names, k, xTest, yTest)
        forest_ft[label] = features
        
        # Create graph on feature importance
        graph_permutation(forest_importances, result)

        forest_result.append([label, accuracy, top_k, "Random Forest"])
        print()
        
        print("---------------Gradient Boosted Trees model---------------------")
        # tune hyperparameters: learning_rate, n_estimators, criterion, max_depth, min_samples_leaf
        # eval_plot to find pgrid values
        #row = eval_plot(xTrain, yTrain, xTest, yTest, label)
        #trees.append(row)
        
        # Parameter grid for grid search (values selected based on eval_plot)
        print("tuning hyperparameters...")
        
        param_grid = {
            'learning_rate':[0.1],
            'n_estimators': [40, 80, 100], 
            'max_depth': [9, 11],
            'min_samples_leaf': [20], 
        }

        # Perform grid search
        best_params = eval_gridsearch(GradientBoostingClassifier(), param_grid, xTrain, yTrain)
        print(best_params)

        # Train, test random forest
        gbt = GradientBoostedTrees(**best_params)
        start = timeit.default_timer()
        gbt.train(xTrain, yTrain)
        accuracy, f1, auroc, auprc, roc = gbt.predict(xTest, yTest)
        time_lr = timeit.default_timer() - start

        # Store result
        gbt_result.append([label, best_params, accuracy])
        print("Accuracy: ", accuracy, ", AUPRC: ", auprc, ", AUROC: ", auroc, ", F1: ", f1, ", time: ", time_lr)

        # Find top features via feature importance using mean decrease in impurity
        k = 3   # store k top features
        top_k, features, gbt_importances, result = gbt.features(feature_names, k, xTest, yTest)
        gbt_ft[label] = features

        gbt_result.append([label, accuracy, top_k, "Gradient Boosting Trees"])
    
        print("---------------Extreme Gradient Boosted Trees model---------------------")
        # tune hyperparameters: learning_rate, n_estimators, criterion, max_depth, min_samples_leaf
        # eval_plot to find pgrid values
        #row = eval_plot(xTrain, yTrain, xTest, yTest, label)
        #trees.append(row)

        # Parameter grid for grid search (values selected based on eval_plot)
        print("tuning hyperparameters...")
        
        param_grid = {
            'learning_rate':[0.1],
            'num_parallel_tree': [15, 40, 80], 
            'max_depth': [10]
        }

        # Perform grid search
        best_params = eval_gridsearch(XGBRFClassifier(), param_grid, xTrain, yTrain)
        print(best_params)

        # Train, test random forest
        xgb = XGBoost(**best_params)
        start = timeit.default_timer()
        xgb.train(xTrain, yTrain)
        accuracy, f1, auroc, auprc, roc = xgb.predict(xTest, yTest)
        time_lr = timeit.default_timer() - start

        # Store result
        xgb_result.append([label, best_params, accuracy])
        print("Accuracy: ", accuracy, ", AUPRC: ", auprc, ", AUROC: ", auroc, ", F1: ", f1, ", time: ", time_lr)

      
        # Find top features via feature importance using mean decrease in impurity
        k = 3   # store k top features
        features = xgb.features(feature_names, k)
        xgb_ft[label] = features

        xgb_result.append([label, accuracy, "xTreme Gradient Boosted Trees"])
    

    
    
    # Run Female subgroup data
    for label, arr in zip(lab1, arr1_df):
        print(label)
        
        # test train split
        xTrain, xTest, yTrain, yTest = split_data(arr)
        
        # save feature names
        feature_names = xTrain.columns
        print("---------------Random Forest model---------------------")
        # tune hyperparameters: num_trees, criterion, max_depth, min_samples_leaf, ccp_alpha
        # eval_plot to find pgrid values
        #row = eval_plot(xTrain, yTrain, xTest, yTest, label)
        #trees.append(row)

        # Parameter grid for grid search (values selected based on eval_plot)
        print("tuning hyperparameters...")
        
        param_grid = {
            'n_estimators': [13, 15, 17, 19, 61, 63, 65, 70, 80, 85, 115], 
            'criterion': ['gini'],
            'max_depth': [9, 11],
            'min_samples_leaf': [20], 
            'ccp_alpha':[0]
        }

        # Perform grid search
        best_params = eval_gridsearch(RandomForestClassifier(), param_grid, xTrain, yTrain)
        print(best_params)

        # Train, test random forest
        forest = RandomForest(**best_params)
        start = timeit.default_timer()
        forest.train(xTrain, yTrain)
        accuracy, f1, auroc, auprc, roc = forest.predict(xTest, yTest)
        time_lr = timeit.default_timer() - start

        # Store result
        forest_result.append([label, best_params, accuracy])
        print("Accuracy: ", accuracy, ", AUPRC: ", auprc, ", AUROC: ", auroc, ", F1: ", f1, "time: ", time_lr)

        # Find top features via feature importance using mean decrease in impurity
        k = 3   # store k top features
        top_k, features, forest_importances, result = forest.features(feature_names, k, xTest, yTest)
        forest_ft[label] = features

        forest_result.append([label, accuracy, top_k, "Random Forest"])

        # Create graph on feature importance
        graph_permutation(forest_importances, result)
        
        print()
        
        print("---------------Gradient Boosted Trees model---------------------")
        # tune hyperparameters: learning_rate, n_estimators, criterion, max_depth, min_samples_leaf
        # eval_plot to find pgrid values
        #row = eval_plot(xTrain, yTrain, xTest, yTest, label)
        #trees.append(row)

        # Parameter grid for grid search (values selected based on eval_plot)
        print("tuning hyperparameters...")
        
        param_grid = {
            'learning_rate':[0.1],
            'n_estimators': [40, 80, 100], 
            'max_depth': [9, 11],
            'min_samples_leaf': [20], 
        }

        # Perform grid search
        best_params = eval_gridsearch(GradientBoostingClassifier(), param_grid, xTrain, yTrain)
        print(best_params)

        # Train, test random forest
        gbt = GradientBoostedTrees(**best_params)
        start = timeit.default_timer()
        gbt.train(xTrain, yTrain)
        accuracy, f1, auroc, auprc, roc = gbt.predict(xTest, yTest)
        time_lr = timeit.default_timer() - start


        # Store result
        gbt_result.append([label, best_params, accuracy])
        print("Accuracy: ", accuracy, ", AUPRC: ", auprc, ", AUROC: ", auroc, ", F1: ", f1, ", time: ", time_lr)

        # Store result
        gbt_result.append([label, best_params, accuracy])
        print("Accuracy: ", accuracy)

        # Find top features via feature importance using mean decrease in impurity
        k = 3   # store k top features
        top_k, features, gbt_importances, result = gbt.features(feature_names, k, xTest, yTest)
        gbt_ft[label] = features

        gbt_result.append([label, accuracy, top_k, "Gradient Boosting Trees"])
        
        print()
        
        print("---------------Extreme Gradient Boosted Trees model---------------------")
        # tune hyperparameters: learning_rate, n_estimators, criterion, max_depth, min_samples_leaf
        # eval_plot to find pgrid values
        #row = eval_plot(xTrain, yTrain, xTest, yTest, label)
        #trees.append(row)

        # Parameter grid for grid search (values selected based on eval_plot)
        print("tuning hyperparameters...")
        
        param_grid = {
            'learning_rate':[0.1],
            'num_parallel_tree': [15], 
            'max_depth': [10]
        }

        # Perform grid search
        best_params = eval_gridsearch(XGBRFClassifier(), param_grid, xTrain, yTrain)
        print(best_params)

        # Train, test random forest
        xgb = XGBoost(**best_params)
        start = timeit.default_timer()
        xgb.train(xTrain, yTrain)
        accuracy, f1, auroc, auprc, roc = xgb.predict(xTest, yTest)
        time_lr = timeit.default_timer() - start

        # Store result
        xgb_result.append([label, best_params, accuracy])
        print("Accuracy: ", accuracy, ", AUPRC: ", auprc, ", AUROC: ", auroc, ", F1: ", f1, ", time: ", time_lr)

      
        # Find top features via feature importance using mean decrease in impurity
        k = 3   # store k top features
        features = xgb.features(feature_names, k)
        xgb_ft[label] = features

        xgb_result.append([label, accuracy, "xTreme Gradient Boosted Trees"])
        
       
    # graph
    graph_hm((pd.DataFrame(forest_ft)).T, "Random Forest Feature Importance Using Mean Decrease in Impurity", cmap="YlGnBu")
    graph_hm((pd.DataFrame(gbt_ft)).T, "Gradient Boosting Trees Feature Importance Using Mean Decrease in Impurity", cmap='coolwarm')
    graph_hm((pd.DataFrame(xgb_ft)).T, "xTreme Gradient Boosting Trees Feature Importance Using Mean Decrease in Impurity", cmap="crest")
    """
    # USED FOR EVAL_PLOT (plot line graph of the hyperparameters)

    flat_trees = [item for sublist in trees for item in sublist]
    trees_df = pd.DataFrame(flat_trees, columns=['X', 'Accuracy', 'Category'])

    print(trees)
    sns.set_style("whitegrid")
    sns.lineplot(data=trees_df, x='X', y='Accuracy', hue='Category')

    plt.xlabel('num_parallel_trees')
    plt.ylabel('Testing Accuracy')
    plt.title('Hyperparameter tested: num_parallel_trees')

    # Show plot
    plt.show()

    """
if __name__ == "__main__":
    main()
