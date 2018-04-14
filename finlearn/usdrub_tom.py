# http://profitraders.com/MachineLearning/scikit-learn-01.html

import datetime
import numpy as np
import pandas as pd
import sklearn
from pandas_datareader import data, wb

import pandas_datareader.data as web

finRead = web.DataReader('USD000UTSTOM', 'moex', start='2017-01-01', end='2018-05-10')

#yahoo
#symbol = "^GSPC"  # Код финансового инструмента на сайте Yahoo Finance
#start_date = datetime.datetime(2013, 7, 1) # Начальная дата - 1 июля 2013
#end_date = datetime.datetime(2014, 4, 30)  # Конечная дата - 30 апреля 2014
#ts1 = web.DataReader(symbol, "yahoo", start_date, end_date)
#ts1.head(3) # Вывод первых трёх строк таблицы
#ts1.tail(2) # Вывод последних двух строк таблицы

# google 
#start = datetime.datetime(2010, 1, 1)
#end = datetime.datetime(2013, 1, 27)
#f = web.DataReader('F', 'google', start, end)
ts1 = finRead[finRead.BOARDID == 'CETS']
#del ts1['BOARDID','SHORTNAME','SECID']


# Создадим новый набор данных, с тем же количеством строк, но нам нужны только цены закрытия и объёмы:
ts2 = pd.DataFrame(index=ts1.index)
ts2["Today"] = ts1["CLOSE"]
ts2["Volume"] = ts1["VOLRUR"]


# Добавим к новому набору данных 2 столбца, которые будут хранить задержанные цены закрытия, сдвинутые на 1 день (столбец Lag1) и на 2 дня (Lag2):
count = 2
for i in range(0, count):
    ts2["Lag%s" % str(i+1)] = ts1["CLOSE"].shift(i+1)

# Создадим ещё один набор данных, только в каждой i-й строке вместо цены закрытия pi будем хранить изменение этой цены по сравнению с предыдущим днём, выраженное в процентах по отношению к предыдущему дню, по формуле
ts = pd.DataFrame(index=ts2.index)
ts["Volume"] = ts2["Volume"]
ts["Today"] = ts2["Today"].pct_change()*100.0

#Заменим нулевые значения малыми числами (например, 0.0001), т.к. нулевые значения будут мешать работе некоторых алгоритмов обучения (см. ниже).
for i,x in enumerate(ts["Today"]):
        if (abs(x) < 0.0001):
            ts["Today"][i] = 0.0001

# Добавим к новому набору данных те же 2 столбца для задержанных данных, но вместо цены закрытия будем хранить процентное изменение. Кроме того, нам потребуется ещё один столбец Direction (Направление): если цена закрытия выросла по сравнению с предыдущим днём, то в этот столбец запишем 1, а если цена закрытия уменьшилась, то значение -1.
for i in range(0, count):
    ts["Lag%s" % str(i+1)] = ts2["Lag%s" % str(i+1)].pct_change()*100.0

ts["Direction"] = np.sign(ts["Today"])
ts = ts[count+1:]  # Пропустили первые count дней, т.к. для них некоторые данные не определены (NaN)

# Разделим наш набор данных на две части: обучающую и тестовую. Первую будем использовать для обучения модели, а вторую – для тестирования результатов.
start_test = datetime.datetime(2018,1,1) # Начало тестового набора данных - 1 января 2014

# Для обучения модели будем использовать значения из столбцов Lag1 и Lag2:
cols = []
for i in range(0, count):
    cols.extend(["Lag%s" % str(i+1)])

# Целевые значения – направления изменения цены закрытия, т.е. значения из столбца Direction.
x = ts[cols] # Входные данные для обучения
y = ts["Direction"]     # Целевые значения

x_train = x[x.index < start_test] # Входные данные для обучения (март)
print (x_train[0:5]) # Вывели для проверки первые 5 строк обучающей выборки
x_test = x[x.index >= start_test] # Входные данные для теста (апрель)
print (x_test[0:5]) # Вывели для проверки первые 5 строк тестовой выборки

y_train = y[y.index < start_test] # Целевые значения для обучения (март)
print (y_train[0:5])
y_test = y[y.index >= start_test] # Целевые значения для теста (апрель)
print (y_test[0:5])

#from sklearn import preprocessing
#scaler = preprocessing.StandardScaler().fit(x_train)
#x_train = scaler.transform(x_train)
#x_test = scaler.transform(x_test)
#print x_train[0:5] # Вывели для проверки первые 5 строк обучающей выборки после масштабирования

d = pd.DataFrame(index=y_test.index) # Набор данных для проверки модели
d["Actual"] = y_test                 # Реальные изменения цен закрытия

# Отобразим получившиеся значения на графике: красные точки соответствуют дням, когда цена снижалась, зелёные - когда повышалась.
import matplotlib.pyplot as plt
x_tr = x_train.values[:]
y_tr = y_train.values[:]
print (x_tr[0:5])
print (y_tr[0:5])

xs = x_tr[:, 0][y_tr == 1]
ys = x_tr[:, 1][y_tr == 1]
plt.scatter(xs, ys, c="green")

xs = x_tr[:, 0][y_tr == -1]
ys = x_tr[:, 1][y_tr == -1]
plt.scatter(xs, ys, c="red")

plt.legend(["Up", "Down"])
plt.xlabel('Lag1')
plt.ylabel('Lag2')

# Сначала испробуем модель логистической регрессии LogisticRegression:
from sklearn.linear_model import LogisticRegression
model1 = LogisticRegression()
model1.fit(x_train, y_train)     # Обучение (подбор параметров модели)
d['Predict_LR'] = model1.predict(x_test) # Тест

# Считаем процент правильно предсказанных направлений изменения цены:
d["Correct_LR"] = (1.0+d['Predict_LR']*d["Actual"])/2.0
print (d)
hit_rate1 = np.mean(d["Correct_LR"])
print ("Процент верных предсказаний: %.1f%%" % (hit_rate1*100))