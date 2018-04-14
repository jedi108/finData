# http://profitraders.com/Python/FinamDownloader.html

import datetime
import numpy as np
import pandas as pd
import sklearn
from pandas_datareader import data, wb

import pandas_datareader.data as web

finRead = web.DataReader('MICEX10INDEX', 'moex', start='2017-01-01', end='2018-05-10')
print(finRead.head(10))