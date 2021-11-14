# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
from pandas import plotting 
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def main(df_data):
    plotting.scatter_matrix(df_data.iloc[:, 1:], figsize=(120, 120), alpha=0.5)
    plt.show()

    pca = PCA()
    pca.fit(df_data)
    feature = pca.transform(df_data)

    pd.DataFrame(feature, columns=["PC{}".format(x + 1) for x in range(len(df_data.columns))])
    
    plt.figure(figsize=(6, 6))
    plt.scatter(feature[:, 0], feature[:, 1], alpha=0.8)
    plt.grid()
    
    plotting.scatter_matrix(pd.DataFrame(feature, 
                            columns=["PC{}".format(x + 1) for x in range(len(df_data.columns))]), 
                            figsize=(60, 60), alpha=0.5) 
    plt.show()

if __name__=="__main__":
    
    df_data=pd.read_csv("standardized_data.csv", index_col=0, sep=",")
    
    main(df_data)
    


