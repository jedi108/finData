# For data manipulation
import pandas as pd
import numpy as np

import visual.plot
from visual.plot import PlotyChart
from visual.plot import Bollingers

# Df = pd.read_csv('BTC_USD.csv', sep=',')
Df = pd.read_csv('USD000UTSTOM.csv', sep=',')
#USD000UTSTOM.csv

bollingers = Bollingers(Df, 3, 20)
bollingers.visualise(300, 400)

# pchart = visual.plot.PlotyChart(Df)
# pchart.Plot()
# pchart.PlotCandle()