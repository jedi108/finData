from pandas_datareader import data, wb
import pandas_datareader.data as web

Df1 = web.DataReader('USD000UTSTOM', 'moex', start='2012-01-01', end='2014-04-10')
Df = Df1[Df1.BOARDID == 'CETS']

Df.to_csv('USD000UTSTOM.csv')