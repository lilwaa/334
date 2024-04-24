import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# Import dataset
df = pd.read_csv("raw-data/mentalhealth.csv", index_col=False)
# original size: 292363 x 17
# Drop rows containing NAs
df.dropna(how='any', inplace=True)
# new size: 287162 x 17

# Subset only countries containing United States
df=df[df['Country'] == 'United States']
# new size: 168056 x 17

# Drop Unnecessary columns: Timestamp, mental health interview, country
df.drop(['Timestamp', "mental_health_interview", "Country"], axis=1, inplace=True)
df['Gender_bin'] = (df['Gender']=='Female').astype(int)
df['self_employed_bin'] = (df['self_employed']=='Yes').astype(int)
df['family_history_bin'] = np.where(df['family_history'] == 'Yes', 1, 0)
df['treatment_bin'] = np.where(df['treatment'] == 'Yes', 1, 0)
df["Growing_Stress"].replace("Yes", 2, inplace= True)
df["Growing_Stress"].replace( "No", 0, inplace= True)
df["Growing_Stress"].replace( "Maybe", 1, inplace= True)
df['Coping_struggles_bin'] = np.where(df['Coping_Struggles'] == 'Yes', 1, 0)


df.drop(['Gender', "self_employed", "family_history", "treatment", "Coping_Struggles"], axis=1, inplace=True)

# One hot encode the other columns
def onehot(inputdf, colname):
    hot_encoded_data = pd.get_dummies(inputdf, columns = [colname], dtype= int)
    return hot_encoded_data

col_names = ['Occupation', 'Days_Indoors', 'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Work_Interest', 'Social_Weakness', 'care_options']
for col in col_names: 
    df = onehot(df, col)
df.to_csv('clean-data_try.csv', index= False) 