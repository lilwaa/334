import pandas as pd
df= pd.read_csv("raw-data/mentalhealth.csv")
df_check= df.loc[df["Growing_Stress"] =="No"]
df_check= df_check.loc[df["Country"] =="United States"]
print(df_check)
print(df.loc[df["Country"] =="United States"])