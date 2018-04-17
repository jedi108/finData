# For data manipulation
import pandas as pd

# To plot
from mpl_finance import candlestick2_ohlc
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.dates as mdates
from matplotlib.dates import date2num, DayLocator, DateFormatter

import datetime as datetime

class PlotyChart():
    def __init__ (self, data_frame):
        self.df = data_frame

    def Plot(self):
        self.df.CLOSE.plot(figsize=(10,5))
        plt.show()

    def PlotCandle2(self):
        fig, ax = plt.subplots()
        candlestick2_ohlc(ax,self.df['OPEN'],self.df['HIGH'],self.df['LOW'],self.df['CLOSE'],width=0.6)
        fig.autofmt_xdate()
        fig.tight_layout()
        plt.show()

    def PlotCandle(self):
        xdate = self.df['TRADEDATE']
        def mydate(x, pos):
            try:
                return xdate[int(x)]
            except IndexError:
                return ''

        self.fig = plt.figure()
        ax1 = plt.subplot2grid((6, 4), (1, 0), rowspan=4, colspan=4, axisbg='w')

        # xdate = [datetime.datetime.fromtimestamp(i) for i in self.df['TRADEDATE']]
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))

        xdate = date2num(pd.to_datetime(self.df['TRADEDATE']).tolist())

        candlestick2_ohlc(ax1,
                          self.df['OPEN'],self.df['HIGH'],self.df['LOW'],self.df['CLOSE'],width=0.6)
        ax1.grid(True)


        for label in ax1.xaxis.get_ticklabels():
            label.set_rotation(45)

        ax1.minorticks_on()
        ax1.xaxis.set_major_formatter(ticker.FuncFormatter(mydate))
        ax1.xaxis.set_major_locator(DayLocator())

        self.fig.autofmt_xdate()
        self.fig.tight_layout()

        # for label in ax1.xaxis.get_ticklabels():
        #     label.set_rotation(90)


        # plt.subplots_adjust(left=.10, bottom=.19, right=.93, top=.95, wspace=.20, hspace=0)
        # plt.suptitle(' Stock Price')
        plt.show()



# https://www.kaggle.com/mattwills8/bollinger-band-backtesting

TIME = 'TRADEDATE'
OPEN = 'OPEN'
CLOSE = 'CLOSE'
HIGH = 'HIGH'
LOW = 'LOW'
VOLUME = 'Volume'

ROLLING_AVERAGE = 'Rolling Average'
ROLLING_STD = 'Rolling St Dev'
BOLLINGER_TOP = 'Bollinger Top'
BOLLINGER_BOTTOM = 'Bollinger Bottom'

CROSSED_BOLLINGER_BOTTOM_DOWN = 'Crossed Bollinger Bottom Down'
CROSSED_BOLLINGER_BOTTOM_UP = 'Crossed Bollinger Bottom Up'
CROSSED_BOLLINGER_TOP_DOWN = 'Crossed Bollinger Top Down'
CROSSED_BOLLINGER_TOP_UP = 'Crossed Bollinger Top Up'

class Bollingers:

    def __init__(self, df, k, window):
        self.df = df
        self.k = k
        self.window = window

        # compute rolling calculations
        self.df[ROLLING_AVERAGE] = self.df[CLOSE].rolling(window=self.window,center=False).mean()
        self.df[ROLLING_STD] = self.df[CLOSE].rolling(window=self.window,center=False).std() 

        # compute bollingers
        self.df[BOLLINGER_TOP] = self.df.apply(lambda row: self.bollinger('top', row[ROLLING_AVERAGE], row[ROLLING_STD]), axis=1)
        self.df[BOLLINGER_BOTTOM] = self.df.apply(lambda row: self.bollinger('bottom', row[ROLLING_AVERAGE], row[ROLLING_STD]), axis=1)

        # add bools for price crossing bollingers
        self.df[CROSSED_BOLLINGER_BOTTOM_DOWN] = self.df.apply(lambda row: self.crossed_down(row[HIGH], row[CLOSE], row[BOLLINGER_BOTTOM]), axis=1)
        self.df[CROSSED_BOLLINGER_BOTTOM_UP] = self.df.apply(lambda row: self.crossed_up(row[LOW], row[CLOSE], row[BOLLINGER_BOTTOM]), axis=1)
        self.df[CROSSED_BOLLINGER_TOP_DOWN] = self.df.apply(lambda row: self.crossed_down(row[HIGH], row[CLOSE], row[BOLLINGER_TOP]), axis=1)
        self.df[CROSSED_BOLLINGER_TOP_UP] = self.df.apply(lambda row: self.crossed_up(row[LOW], row[CLOSE], row[BOLLINGER_TOP]), axis=1)

    def bollinger(self, top_or_bottom, rolling_av, rolling_std):
        if top_or_bottom == 'top':
            return rolling_av + self.k*rolling_std
        elif top_or_bottom == 'bottom':
            return rolling_av - self.k*rolling_std
        else:
            raise ValueError('Expect "top" or "bottom" for top_or_bottom')

    def crossed_down(self, High, Close, Bollinger):
        if High >= Bollinger and Close < Bollinger:
            return 1
        else:
            return 0

    def crossed_up(self, Low, Close, Bollinger):
        if Low <= Bollinger and Close > Bollinger:
            return 1
        else:
            return 0

    def visualise(self, start_row, end_row):
        df_sample = self.df[start_row:end_row]
        time_range = range(0, len(df_sample[TIME]))

        fig = plt.figure(figsize=(15,10))
        ax = fig.add_subplot(111)

        candlestick2_ohlc(ax,
                          opens=df_sample[OPEN],
                          closes=df_sample[CLOSE],
                          highs=df_sample[HIGH],
                          lows=df_sample[LOW],
                          width=1,
                          colorup='g',
                          colordown='r',
                          alpha=0.75
                          )

        ax.plot(time_range, df_sample[ROLLING_AVERAGE])
        ax.plot(time_range, df_sample[BOLLINGER_TOP])
        ax.plot(time_range, df_sample[BOLLINGER_BOTTOM])

        plt.ylabel("Price")
        plt.xlabel("Time Periods")
        plt.legend()

        plt.show()