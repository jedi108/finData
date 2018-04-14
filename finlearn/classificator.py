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

Df1 = web.DataReader('USD000UTSTOM', 'moex', start='2012-01-01', end='2014-04-10')
Df = Df1[Df1.BOARDID == 'CETS']



# print(Df.head(10))
# def PlotDataframe(Df):
#     Df.CLOSE.plot(figsize=(10,5))
#     plt.ylabel("USDRUB")
#     plt.show()

#PlotDataframe(Df)

# data = []
# data.insert(0, {'BOARDID': 'CNGD', 'VOLRUR': 0, 'OPEN': 0, 'LOW': 0, 'HIGH': 0, 'CLOSE': 1 })
# data.insert(0, {'BOARDID': 'CNGD', 'VOLRUR': 0, 'OPEN': 0, 'LOW': 0, 'HIGH': 0, 'CLOSE': 1 })
# data.insert(0, {'BOARDID': 'CNGD', 'VOLRUR': 0, 'OPEN': 0, 'LOW': 0, 'HIGH': 0, 'CLOSE': 1 })
# Df = pd.concat([pd.DataFrame(data), Df1], ignore_index=True)

Df['Open-Close'] = Df.OPEN - Df.CLOSE
Df['High-Low'] = Df.HIGH - Df.LOW
# Df['vol-prev0'] = Df['VOLRUR'].shift(1)
# Df['vol-prev1'] = Df['VOLRUR'].shift(2)
# Df['vol-prev2'] = Df['VOLRUR'].shift(3)

# Df.dropna(subset=['OPEN'], inplace = True)
# Df.dropna(subset=['LOW'], inplace = True)
# Df.dropna(subset=['HIGH'], inplace = True)
# Df.dropna(subset=['CLOSE'], inplace = True)
# Df.dropna(subset=['vol-prev0'], inplace = True)
# Df.dropna(subset=['vol-prev1'], inplace = True)
# Df.dropna(subset=['vol-prev2'], inplace = True)
# Df.dropna(subset=['VOLRUR'], inplace = True)
# Df.dropna(subset=['Open-Close'], inplace = True)

# X=Df[['Open-Close','High-Low', 'VOLRUR', 'vol-prev0', 'vol-prev1', 'vol-prev2', 'HIGH', 'LOW']]
X=Df[['Open-Close','High-Low', 'VOLRUR', 'HIGH', 'LOW']]

y = np.where(Df['CLOSE'].shift(-1) > Df['CLOSE'],1,-1)

#X.dropna(how='any')
#X.dropna(subset=['vol-prev0'], inplace = True)
#X.dropna(subset=['VOLRUR'], inplace = True)
#X.dropna(subset=['Open-Close'], inplace = True)
#X.head(10)

split_percentage = 0.5

split = int(split_percentage*len(Df))
# Train data set

X_train = X[:split]
y_train = y[:split]

# Test data set

X_test = X[split:]
y_test = y[split:]


cls = SVC().fit(X_train, y_train)
accuracy_train = accuracy_score(y_train, cls.predict(X_train))
accuracy_test = accuracy_score(y_test, cls.predict(X_test))
print('\nTrain Accuracy:{: .2f}%'.format(accuracy_train*100))
print('Test Accuracy:{: .2f}%'.format(accuracy_test*100))


Df['Predicted_Signal'] = cls.predict(X)
# Calculate log returns
Df['Return'] = np.log(Df.CLOSE.shift(-1) / Df.CLOSE)*100
Df['Strategy_Return'] = Df.Return * Df.Predicted_Signal
Df.Strategy_Return.iloc[split:].cumsum().plot(figsize=(10,5))
plt.ylabel("Strategy Returns (%)")
plt.show()

#PlotDataframe(Df)