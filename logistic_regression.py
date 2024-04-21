import argparse
import json
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
lr= LogisticRegression()
#get all subsets in 
def get_subsets()
def main():
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

   
    # save roc curves to data
    rocDF.to_csv(args.rocOutput, index=False)
    # store the best parameters
    with open(args.bestParamOutput, 'w') as f:
        json.dump(bestParamDict, f)


if __name__ == "__main__":
    main()
