# For data manipulation
import pandas as pd
import numpy as np

import visual.plot
from visual.plot import PlotyChart
from visual.plot import Bollingers

# Df = pd.read_csv('../BTC_USD.csv', sep=',')
Df = pd.read_csv('../USD000UTSTOM.csv', sep=',')


Df.LOW = Df.LOW.round(4)


df_sample = Df[1:50]
# df_sample.info()
print(df_sample.head(5))
miny = min([low for low in df_sample.LOW if low != -1])
print(miny)
# miny.info()

Df.info()
# raise SystemExit(1)

bollingers = Bollingers(Df, 2, 20)
bollingers.visualise(50, 300)

Pc = visual.plot.PlotyChart(Df)
Pc.PlotCandle()
# Pc.Plot()