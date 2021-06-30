# SP500-prediction-using-LSTM
Stock market prediction using deep learning (LSTM)

This is an example of using deep learning (praticularly LSTM) for stock market prediction.

## Files:

### ReadS&P500.py
This file extract S&P500 tickers and also add commodities, currencies, indices, and crypto-currencies. The extracted data will be stored to **S&P500-Info.csv** and **S&P500-Symbols.csv**
The files look like:

![image](https://user-images.githubusercontent.com/57122652/123915757-2f945600-d9c4-11eb-8360-bc607c3cade9.png)

### ReadDataAll.py
This file reads data from each ticker using Yahoo finance API. We will store the price data of each ticker in **All_1d.pickle** and the trading volume of each ticker in **All_1dv.pickle** like:

![image](https://user-images.githubusercontent.com/57122652/123916567-25bf2280-d9c5-11eb-9181-7118f5552c38.png)

### CreateCorrelation.py
This file calculate the correlations between all the ticker. You can visualize the heat-map using **visualize_data** function:

![image](https://user-images.githubusercontent.com/57122652/123917456-2906de00-d9c6-11eb-9040-5028d711fcfa.png)

Also, this file calculate the top 5 and buttom 5 correlated tickers related to each particular ticker to store in **Top_Bottom_Five_Correlated_Names**:

![image](https://user-images.githubusercontent.com/57122652/123917796-88fd8480-d9c6-11eb-890a-b33f3414b405.png)

### TrainTicker.py
This file prepare the data for training. It concatenate price, trading volume, correlated tickers, and some indicators (MA, RSI, MACD).

### Main.py
This file train an LSTM model for a particular ticker. It calls the necessary functions and data from previous files, normalize the data, split data for training and test before creating a model. The goal is to predict the next-day price with 100 historical data. 

The current LSTM model is like:

![model](https://user-images.githubusercontent.com/57122652/123918533-51430c80-d9c7-11eb-8946-14fb9c7c1795.png)

To comapre the prediction power, here is the plot for training data of **MSFT**:

![image](https://user-images.githubusercontent.com/57122652/123915106-7e8dbb80-d9c3-11eb-9358-5fa4bcbbe8c8.png)


## Future work:
1. Use more data (older histories), or shorter time frames (hourly)

2. Use more indicators and features.

3. Predict the next 10 days instead of the next day.




