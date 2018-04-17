# https://www.quantinsti.com/blog/machine-learning-classification-strategy-python/

# machine learning classification

from sklearn.svm import SVC
from sklearn.metrics import scorer
from sklearn.metrics import accuracy_score

# For data manipulation
import pandas as pd
import numpy as np

# To plot
import matplotlib.pyplot as plt
import seaborn
# To fetch data
from pandas_datareader import data, wb
import pandas_datareader.data as web


# Df = pd.read_csv('USD000UTSTOM.csv', sep=',')
Df = pd.read_csv('BTCETH.csv', sep=',')

def PlotDataframe(Df):
    Df.CLOSE.plot(figsize=(10,5))
    plt.ylabel("USDRUB")
    plt.show()
    raise SystemExit(1)

# PlotDataframe(Df)

# data = []
# data.insert(0, {'BOARDID': 'CNGD', 'VOLRUR': 0, 'OPEN': 0, 'LOW': 0, 'HIGH': 0, 'CLOSE': 1 })
# data.insert(0, {'BOARDID': 'CNGD', 'VOLRUR': 0, 'OPEN': 0, 'LOW': 0, 'HIGH': 0, 'CLOSE': 1 })
# data.insert(0, {'BOARDID': 'CNGD', 'VOLRUR': 0, 'OPEN': 0, 'LOW': 0, 'HIGH': 0, 'CLOSE': 1 })
# Df = pd.concat([pd.DataFrame(data), Df1], ignore_index=True)

Df['Open-Close'] = Df.OPEN - Df.CLOSE
Df['High-Low'] = Df.HIGH - Df.LOW
Df['vol-prev0'] = Df['VOLRUR'].shift(1)

# ---------------------------------------------------------------
# SAVE FOR PREDICT ----------------------------------------------

# Df['Y'] = np.where(Df['CLOSE'].shift(-1) > Df['CLOSE'],1,-1)

# Df.dropna(axis = 0, how='any', inplace = True)
# Df = Df.drop(columns=['TRADEDATE', 'weightedAverage', 'VOLRUR1', 'Unnamed: 0'])
# print(Df.head(10))
# # Df.info()
# Df.to_csv('PREDICT_BTCETH.csv', header=False)
# raise SystemExit(1)

# SAVE FOR PREDICT ----------------------------------------------
# ---------------------------------------------------------------

Df.dropna(axis = 0, how='any', inplace = True)

X=Df[['Open-Close','High-Low', 'VOLRUR', 'HIGH', 'LOW', 'vol-prev0', 'CLOSE']]


y = np.where(Df['CLOSE'].shift(-1) > Df['CLOSE'],1,-1)

# print(X.info())
# print(len(y))
# raise SystemExit(1)

split_percentage = 0.5

split = int(split_percentage*len(Df))

X_train = X[:split]
y_train = y[:split]
X_test = X[split:]
y_test = y[split:]

cls = SVC().fit(X_train, y_train)
accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))
print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))


Df['Predicted_Signal'] = cls.predict(X)
Df['Return'] = np.log(Df.CLOSE.shift(-1) / Df.CLOSE)*100
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal
Df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()
