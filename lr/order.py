import pandas as pd


df=pd.read_csv("lr_coeff.csv")
df_order= df.iloc[5]

df_order= df_order.transpose()
df_order= df_order.sort_values()
print(df_order)

