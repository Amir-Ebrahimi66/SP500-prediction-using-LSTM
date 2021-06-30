import pandas as pd
import yfinance as yf
import datetime as dt
import pickle

# Merge all in one file
# This function extract symbols in "S&P500-Symbols.csv" and download historical data.
# The whole historical data will be stored in a pickle


Tickers = pd.read_csv("S&P500-Symbols.csv")
start = dt.datetime(2015, 1, 1)
end = dt.datetime(2021, 6, 1) # dt.datetime.now()
main_df = pd.DataFrame()
for count in range(len(Tickers)):
    ticker = Tickers['Symbol'][count]
    # possible intervals:
    # Minutes: 1m, 2m, 5m, 15m, 30m, 90m
    # Hours: 1h
    # Days: 1d, 5d
    # Weeks: 1wk
    # Months: 1mo, 3mo
    df = yf.download(ticker, start, end, interval='1d')
    df.rename(columns={'Volume': ticker}, inplace=True)     # 'Adj Close' to keep the price
    df.drop(['Open', 'High', 'Low', 'Close', 'Adj Close'], 1, inplace=True)     # 'Volume' to keep the price
    if main_df.empty:
        main_df = df
    else:
        main_df = main_df.join(df, how='outer')

    if count % 10 == 0:
        print('Progress = {}%'.format(count*100.0/ len(Tickers)))

with open('All_1dv.pickle', 'wb') as f:
    pickle.dump(main_df, f)

# with open('All_1d.pickle', 'rb') as f:
#     x = pickle.load(f)