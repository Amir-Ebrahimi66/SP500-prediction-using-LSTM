"""

Provide data for training
"""

import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib




def MyData(ticker, x, v, Top_Bottom_Five_Correlated_Names):
    # All correlated price data to main_TrainData Dataframe

    main_TrainData = pd.DataFrame()
    # Add the main ticker price
    main_TrainData[ticker] = x[ticker] # main_TrainData.join(x[ticker], how='left')
    # Add the main ticker volume
    main_TrainData[ticker+'v'] = v[ticker]

    # Add correlated data
    for i in range(10):
        myticker = Top_Bottom_Five_Correlated_Names[ticker][i]
        if main_TrainData.empty:
            main_TrainData[myticker] = x[myticker]
        else: # Error here: why x has two 2021/5/28 ?? stored in the pickle?
            main_TrainData = main_TrainData.join(x[myticker], how='left')


    # Drop unavailable data in the ticker
    main_TrainData.dropna(subset=[ticker, ticker+'v'], inplace=True)
    # Pad unavailable data in correlated tickers
    main_TrainData.fillna(method = 'ffill', inplace = True)

    # Calculate the moving average for the ticker price data
    main_TrainData['7MA'] = main_TrainData[ticker].rolling(window=7, min_periods=0).mean()
    main_TrainData['21MA'] = main_TrainData[ticker].rolling(window=21, min_periods=0).mean()
    main_TrainData['99MA'] = main_TrainData[ticker].rolling(window=99, min_periods=0).mean()

    # Calculate MACD
    tmp1 = main_TrainData[ticker].ewm(span=12, adjust=False).mean() - main_TrainData[ticker].ewm(span=26, adjust=False).mean()
    tmp2 = tmp1.ewm(span=9, adjust=False).mean()
    main_TrainData['MACD'] = tmp1 - tmp2

    # Claculate RSI
    delta = main_TrainData[ticker].diff()
    dUp, dDown = delta.copy(), delta.copy()
    dUp[dUp < 0] = 0
    dDown[dDown > 0] = 0
    RolUp = dUp.ewm(span=14,min_periods=0).mean()
    RolDown = dDown.abs().ewm(span=14, min_periods=0).mean()
    RSI = RolUp / RolDown
    RSI.replace([np.inf, -np.inf], np.nan, inplace=True)
    RSI.fillna(0, inplace=True)
    main_TrainData['RSI'] = RSI

    return main_TrainData








# temp1 = main_TrainData[ticker]
# temp2 = temp1[::-1]
# temp2 = temp2.shift(1).rolling(window=7, min_periods=0).max()
# temp2 = temp2[::-1]
# temp3 = (temp1 - temp2)*100 / temp1
# ind1 = temp3 >= 2 # Sell
# ind2 = temp3 <= -2 # Buy
# ind3 = (-2 < temp3) & (temp3 < 2) # Hold
#
# y=np.zeros(len(temp1))
# y[list(ind1)] = -1
# y[list(ind2)] = 1
# y[list(ind3)] = 0
#
# # main_TrainData['Label'][ind1] = 'Buy'
#
# colors = ['red','green','black']
#
# main_TrainData[ticker].plot()
# plt.show()
#
# p = np.array(temp1)
# h = np.linspace(0, len(p), len(p))
# plt.scatter(h, p, c=y, cmap=matplotlib.colors.ListedColormap(colors))
# plt.show()
