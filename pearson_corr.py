import pandas as pd
import seaborn as sns
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def pearson_corr():
    # Define logistic regression model
    # Fit the model
   df= pd.read_csv("clean-data_try.csv")
   pr= df.corr(method="pearson")
   return pr
def heatmap(pr):
    plt.figure(figsize= (10,10))
    dataplot= sns.heatmap(pr, xticklabels=True, yticklabels=True,cmap="coolwarm")
    plt.tight_layout()
    dataplot.set(title= "Pearson Correlation of Features")
    plt.show()
    
# Example usage
def main(): 
    print(pearson_corr())
    heatmap(pearson_corr())
    
if __name__ == "__main__":
    main()