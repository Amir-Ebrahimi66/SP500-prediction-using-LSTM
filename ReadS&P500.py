"""
This file extracts S&P500 from the web and create two csv files accordingly.csv

Then The commodities and currencies are added at the end.

"""

import pandas as pd
import yfinance as yf
import datetime as dt

# Read S&P500
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
df = table[0]

# The commodities
commodities_Symbols =  ["CL=F", "GC=F", "SI=F",     "HG=F",     "NG=F",         "PL=F",     "PA=F",         "CT=F",     "CC=F",     "ZW=F",     "ALI=F",    "ZC=F",     "LE=F",         "ZS=F"]
commodities_Comments = ["Oil",  "Gold", "Silver",   "Copper",   "Natural Gas",  "Platinum", "Palladium",    "Cotton",   "Cocoa",    "Wheat",    "Aluminum", "Corn",     "Live Cattle",  "Soybean"]

# The currencies
currency_Symbols  = ["AUDUSD=X",    "AUDGBP=X", "AUDJPY=X", "AUDNZD=X", "EURUSD=X", "EURCAD=X", "JPY=X",    "GBPUSD=X", "CHF=X",    "CAD=X",    "EURJPY=X",     "GBPJPY=X", "EURGBP=X", "EURCHF=X", "USDCNY=X"]
currency_Comments = ["AUD/USD",     "AUD/GBP",  "AUD/JPY",  "AUD/NZD",  "EUR/USD",  "EUR/CAD",  "USD/JPY",  "GBP/USD",  "USD/CHF",  "USD/CAD",  "EUR/JPY",      "GBP/JPY",  "EUR/GBP",  "EUR/CHF",  "USD/CNY"]

# The indices
indices_Symbols =  ["^AXJO",   "^DJI",      "^GSPC",   "^IXIC",            "^N225",        "^STOXX50E", "^FTSE", "^GDAXI", "^HSI",  "^FCHI"]
indices_Comments = ["ASX 200", "Dow Jones", "S&P 500", "NASDAQ Composite", "NIKKEI 225",   "ESTX50",    "UK100", "GER30",  "HKG50", "FRA40"]

# Crypto-currencies
crypto_Symbols =  ["BTC-USD",  "ETH-USD",   "ADA-USD",  "BNB-USD",      "XRP-USD",  "LTC-USD",  "BCH-USD",     "TRX-USD", "DOT1-USD"]
crypto_Comments = ["Bitcoin",  "Ethereum",  "Cardano",  "BinanceCoin",  "XRP",      "Litecoin", "BitcoinCash", "TRON",    "Polkadot "]

df_length = len(df)
for i in range(len(commodities_Symbols)):
    df.loc[df_length+i, ["Symbol", "Security"]] = [commodities_Symbols[i], commodities_Comments[i]]


df_length = len(df)
for i in range(len(currency_Symbols)):
    df.loc[df_length+i, ["Symbol", "Security"]] = [currency_Symbols[i], currency_Comments[i]]


df_length = len(df)
for i in range(len(indices_Symbols)):
    df.loc[df_length+i, ["Symbol", "Security"]] = [indices_Symbols[i], indices_Comments[i]]


df_length = len(df)
for i in range(len(crypto_Symbols)):
    df.loc[df_length+i, ["Symbol", "Security"]] = [crypto_Symbols[i], crypto_Symbols[i]]

df.to_csv('S&P500-Info.csv', index=False)
df.to_csv("S&P500-Symbols.csv", columns=['Symbol'], index=False)