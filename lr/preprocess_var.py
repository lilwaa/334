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
# Check ratios of each class
# Convert each column to numeric
# Boolean columns are converted 0/1 (i.e. Gender, self_employed, family_history, treatment, Growing_stress)

df.insert(0,"Gender_bin", (df['Gender']=='Female').astype(int))
'''
df['self_employed_bin'] = (df['self_employed']=='Yes').astype(int)
df['family_history_bin'] = np.where(df['family_history'] == 'Yes', 1, 0)
df['treatment_bin'] = np.where(df['treatment'] == 'Yes', 1, 0)
df['Growing_stress_bin'] = np.where(df['Growing_Stress'] == 'Yes', 1, 0)
df['Coping_struggles_bin'] = np.where(df['Coping_Struggles'] == 'Yes', 1, 0)
'''
#df["Growing_Stress"].replace([["Yest", "Maybe"], "No"], [1,0], inplace= True)
df["Growing_Stress"].replace("Yes", 1, inplace= True)
df["Growing_Stress"].replace( "No", 0, inplace= True)
df.drop(df[df["Growing_Stress"]== "Maybe"].index)
df.replace(["Yes", "High"], 1, inplace=True)
df.replace(["Maybe", "Medium", "Not sure"], 0.5, inplace=True)
df.replace(["No", "Low"], 0, inplace=True)
df.replace(["Go out Every day", "1-14 days","15-30 days","31-60 days","More than 2 months"], [0,0.25,0.5,0.75,1], inplace=True)
#df.drop(['Gender', "self_employed", "family_history", "treatment", "Growing_Stress", "Coping_Struggles", 'Changes_Habits', 'Mental_Health_History', 'Mood_Swings', 'Work_Interest', 'Social_Weakness', 'care_options', 'Days_Indoors'], axis=1, inplace=True)
df.drop(columns="Gender", inplace= True)
# One hot encode the other columns
def onehot(inputdf, colname):
    hot_encoded_data = pd.get_dummies(inputdf, columns = [colname], dtype=int)
    return hot_encoded_data

col_names = ['Occupation']
for col in col_names: 
    df = onehot(df, col)
df.to_csv('clean-data-var.csv', index= False) 