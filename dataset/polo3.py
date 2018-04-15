import pandas as pd
import numpy as np

from poloniex import Poloniex # http://www.pythonandtrading.com/tag/poloniex/
polo = Poloniex()

# print(polo('returnTicker')['BTC_ETH'])
Df = pd.DataFrame(polo.returnChartData('USDT_BTC', 86400, 1514764800))

# Df.rename(str.upper, axis='columns')
# Df.rename(columns={"close": "CLOSE", "open": "OPEN"})
Df.columns=["CLOSE",  "TRADEDATE", "HIGH", "LOW", "OPEN", "VOLRUR", "VOLRUR1", "weightedAverage"]

Df['TRADEDATE'] = pd.to_datetime(Df["TRADEDATE"],unit='s')

# nn = pd.DataFrame()
# nn['Date'] = pd.to_datetime(Df["TRADEDATE"],unit='s')
# print(nn['Date'].tail(10))
print(Df['TRADEDATE'].tail(3))
print(Df['TRADEDATE'].head(3))
# Df.info()
Df.to_csv('BTC_USD.csv')