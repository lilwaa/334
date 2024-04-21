'''
Class Splitter will output xtrain, xtest, ytrain and ytest dictionaris with demographics-based subsets
------------------------------------------------------------------------------------------------------
How to use that:
splitter = Splitter() #Note: you need to go to the templatesplit.py to change the target column if needed
xtrain, xtest, ytrain, ytest = splitter.get_all_subsets()

------------------------------------------------------------------------------------------------------
the format of key in those dictionaries 
Occupation_Name_F or Occupation_name_M
e.g you want to get the xtrain data for male students:
xtrain["Occupation_Student_M"]
'''
import pandas as pd
from sklearn.model_selection import train_test_split
class Splitter:
    def __init__(self, y_col= "Growing_stress_bin"): #I set the default y variable as growing stress, you can change that when you want to test other possible y
        self.xtrain= {}
        self.xtest= {}
        self.ytrain= {}
        self.ytest= {}
        self.y_col= y_col
    

    # Find subgroups based on dcol
    def find_subgroups(self, dcol, df, train_or_test):
        drop_list = dcol.copy()  # Make a copy to avoid modifying the original list
        print(dcol, train_or_test)
        for c in dcol:
            print(c)
            drop_list.remove(c)
            df_tmp = df.copy()  # Make a copy of the dataframe to avoid modifying the original dataframe
            # Extract all rows with c occupation
            df_tmp = df_tmp[df_tmp[c] == 1]
            # Separate gender subsets
            df_f = df_tmp[df_tmp["Gender_bin"] == 1]
            df_m = df_tmp[df_tmp["Gender_bin"] == 0]
            # Check if the df is for training or testing
            if train_or_test == "train":
                label = c + "_F"
                self.ytrain[label] = df_f[self.y_col]
                self.xtrain[label] = df_f.drop(columns=self.y_col)
                label = c + "_M"
                self.ytrain[label] = df_m[self.y_col]
                self.xtrain[label] = df_m.drop(columns=self.y_col)
            else:
                label = c + "_F"
                self.ytest[label] = df_f[self.y_col]
                self.xtest[label] = df_f.drop(columns=self.y_col)
                label = c + "_M"
                self.ytest[label] = df_m[self.y_col]
                self.xtest[label] = df_m.drop(columns=self.y_col)

    
    def get_all_subsets(self):
        df_train= pd.read_csv("train.csv")
        df_test= pd.read_csv("test.csv")
        d_col = ['Occupation_Corporate', 'Occupation_Housewife', 'Occupation_Others', 'Occupation_Student','Occupation_Business']
        #process train datasets
        self.find_subgroups(d_col, df_train, "train")
        #process test datasets
        self.find_subgroups(d_col, df_test,"test")
        return self.xtrain, self.xtest, self.ytrain, self.ytest
def main():
    splitter = Splitter()
    xtrain, xtest, ytrain, ytest = splitter.get_all_subsets()  # Call the method
    print("xtrain subsets:")
    for key, value in xtrain.items():
        print(f"{key}:")
        print(len(xtrain[key]))  # Print the first few rows for demonstration
        print("------")

    print("xtest subsets:")
    for key, value in xtest.items():
        print(f"{key}:")
        print(len(xtest[key])) # Print the first few rows for demonstration
        print("------")

    print("ytrain subsets:")
    for key, value in ytrain.items():
        print(f"{key}:")
        print(value.head())  # Print the first few rows for demonstration
        print("------")

    print("ytest subsets:")
    for key, value in ytest.items():
        print(f"{key}:")
        print(value.head())  # Print the first few rows for demonstration
        print("------")

if __name__== "__main__":
    main()