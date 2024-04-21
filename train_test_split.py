#get the general train, test datasets without separating x and y yet
import pandas as pd
from sklearn.model_selection import train_test_split
def main():
    df= pd.read_csv("clean-data/4-16-firstclean.csv")
    df= df.drop(columns=df.columns[0], axis=1)
    x= list(range(len(df)))
    train_index, test_index= train_test_split(x)
    train_df= df.iloc[train_index, :]
    test_df= df.iloc[test_index, :]
    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

if __name__ == "__main__":
    main()