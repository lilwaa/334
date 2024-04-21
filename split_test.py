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
import os

class Splitter:
    def __init__(self, y_col="Growing_stress_bin"): #I set the default y variable as growing stress, you can change that when you want to test other possible y
        self.xtrain = {}
        self.xtest = {}
        self.ytrain = {}
        self.ytest = {}
        self.y_col = y_col

    def find_subgroups(self, dcol, df):
        for c in dcol.copy():  # Avoid modifying the original list
            #Extract all rows with c occupation
            df_tmp = df[df[c] == 1]
            #split genders
            df_f = df_tmp[df_tmp["Gender_bin"] == 1]
            df_m = df_tmp[df_tmp["Gender_bin"] == 0]
            #process the female subgroup
            df_x_f, df_y_f = self.x_y_split(df_f)
            label_f = f"{c}_F"
            self.tt_split(df_x_f, label_f, "x")
            self.tt_split(df_y_f, label_f, "y")
            #process the male subgroup
            df_x_m, df_y_m = self.x_y_split(df_m)
            label_m = f"{c}_M"
            self.tt_split(df_x_m, label_m, "x")
            self.tt_split(df_y_m, label_m, "y")

    def x_y_split(self, df):
        df_y = df[self.y_col]
        df_x = df.drop(columns=self.y_col)
        return df_x, df_y

    def get_all_subsets(self):
        df = pd.read_csv("clean-data/4-16-firstclean.csv")
        df = df.drop(columns=df.columns[0], axis=1)
        d_col = ['Occupation_Corporate', 'Occupation_Housewife', 'Occupation_Others', 'Occupation_Student', 'Occupation_Business']
        self.find_subgroups(d_col, df)
        self.xtrain = combine_csv_files("xtrain")
        self.xtest = combine_csv_files("xtest")
        self.ytrain = combine_csv_files("ytrain")
        self.ytest = combine_csv_files("ytest")
        return self.xtrain, self.xtest, self.ytrain, self.ytest

    def tt_split(self, df, label, x_or_y):
        train_index, test_index = train_test_split(df.index)
        train_df = df.loc[train_index]
        test_df = df.loc[test_index]
        if x_or_y == "x":
            train_df.to_csv(f"xtrain/{label}.csv", index=False)
            test_df.to_csv(f"xtest/{label}.csv", index=False)
        else:
            train_df.to_csv(f"ytrain/{label}.csv", index=False)
            test_df.to_csv(f"ytest/{label}.csv", index=False)

def combine_csv_files(folder_path):
    data_dict = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            key = filename[:-4]  # Remove the ".csv" extension to use as the key
            data_dict[key] = pd.read_csv(file_path)
    return data_dict

def main():
    splitter = Splitter()
    xtrain, xtest, ytrain, ytest = splitter.get_all_subsets()
    
    print("xtrain")
    for key, value in xtrain.items():
        print(f"{key}:")
        print(len(xtrain[key]))
    print("xtest")
    for key, value in xtest.items():
        print(f"{key}:")
        print(len(xtest[key]))
    print("ytrain")
    for key, value in ytrain.items():
        print(f"{key}:")
        print(len(ytrain[key]))
    print("ytest")
    for key, value in ytest.items():
        print(f"{key}:")
        print(len(ytest[key]))
    
if __name__ == "__main__":
    main()
