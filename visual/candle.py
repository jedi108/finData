from mpl_finance import candlestick2_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
# import datetime as datetime
from matplotlib.dates import date2num, DayLocator, DateFormatter

import pandas as pd
import numpy as np

# quotes = pd.read_csv('USD000UTSTOM.csv', sep=',')
quotes = pd.read_csv('BTC_USD.csv', sep=',')
# USD000UTSTOM.csv

fig, ax = plt.subplots()
candlestick2_ohlc(ax,quotes['OPEN'],quotes['HIGH'],quotes['LOW'],quotes['CLOSE'],width=0.6)

xdate = date2num(pd.to_datetime(quotes['TRADEDATE']).tolist())

xdate = quotes['TRADEDATE']
print(xdate.head(10))
# raise SystemExit(1)


# ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))

def mydate(x,pos):
    try:
        return xdate[int(x)]
    except IndexError:
        return ''



# ax.xaxis.set_major_formatter(DateFormatter('%Y-%m-%d'))
ax.minorticks_on()
ax.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))
ax.xaxis.set_major_locator(DayLocator())

fig.autofmt_xdate()
fig.tight_layout()



plt.show()