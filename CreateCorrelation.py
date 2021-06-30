"""
This file create correlations for all the tickers and save them.

"""

import pandas as pd
import pickle
import matplotlib.pyplot as plt
import numpy as np
# Read price data
with open('All_1d.pickle', 'rb') as f:
    x = pickle.load(f)
#%% Visualize Data
def visualize_data():

    df_corr = x.corr()
    print(df_corr.head())
    #df_corr.to_csv('sp500corr.csv')
    data1 = df_corr.values
    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111)

    heatmap1 = ax1.pcolor(data1, cmap=plt.cm.RdYlGn)
    fig1.colorbar(heatmap1)

    ax1.set_xticks(np.arange(data1.shape[1]) + 0.5, minor=False)
    ax1.set_yticks(np.arange(data1.shape[0]) + 0.5, minor=False)
    ax1.invert_yaxis()
    ax1.xaxis.tick_top()
    column_labels = df_corr.columns
    row_labels = df_corr.index
    ax1.set_xticklabels(column_labels)
    ax1.set_yticklabels(row_labels)
    plt.xticks(rotation=90)
    heatmap1.set_clim(-1, 1)
    plt.tight_layout()
    plt.show()
# visualize_data()

#%% Create the Top-Bottom-Five
# A dataframe containing the tickers, corresponding correlated tickers, and the degree of correlations are created and
# saved in a pickle


df_corr = x.corr()
Top_Bottom_Five_Correlated_Names = pd.DataFrame()
for count,ticker in enumerate(df_corr):
    TickerC = df_corr[ticker].values
    TickerC = np.nan_to_num(TickerC)
    TickerC[count]=0
    ind1 = np.argpartition(np.array(TickerC), -5)[-5:]
    ind2 = np.argpartition(np.array(TickerC), 5)[0:5]
    # print("Top Five Correlated:")
    # print(df_corr[ticker].index[ind1])
    # print(df_corr[ticker].values[ind1])
    #
    # print("Bottom Five Correlated:")
    # print(df_corr[ticker].index[ind2])
    # print(df_corr[ticker].values[ind2])
    a1 = list(df_corr[ticker].index[ind1])
    a2 = list(df_corr[ticker].index[ind2])
    a3 = list(df_corr[ticker].values[ind1])
    a4 = list(df_corr[ticker].values[ind2])
    Top_Bottom_Five_Correlated_Names[ticker] = a1 + a2 + a3 + a4


#%%

with open('All_corr.pickle', 'wb') as f:
    pickle.dump(Top_Bottom_Five_Correlated_Names, f)

# with open('All_corr.pickle', 'rb') as f:
#     Top_Bottom_Five_Correlated_Names = pickle.load(f)
#%%

