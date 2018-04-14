#
#  https://habrahabr.ru/post/206306/
#

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.cross_validation import train_test_split

calories = pd.read_csv("D:\\ProgramData\\Anaconda3\\Scripts\\lab\\anaconda\ML\\calories.csv")
exercise = pd.read_csv("D:\\ProgramData\\Anaconda3\\Scripts\\lab\\anaconda\ML\\exercise.csv")
new1 = pd.merge(calories, exercise, how='inner', on=['User_ID'])
print(new1.head())
#%matplotlib inline

import seaborn as sns
num_cols = ["Age", "Height", "Weight", "Duration", "Heart_Rate", "Body_Temp", "Calories"]

#plt.figure()
#sns.pairplot(new1[num_cols], size=2)
#plt.savefig("D:\\dev\\go\\src\\finrgo\\learn\\1_seaborn_pair_plot.png")

trg = new1[['Calories']]
trn = new1.drop(['Calories','Gender','User_ID'], axis=1)

models = [LinearRegression(), # метод наименьших квадратов
	          RandomForestRegressor(n_estimators=100, max_features ='sqrt'), # случайный лес
	          KNeighborsRegressor(n_neighbors=6), # метод ближайших соседей
	          SVR(kernel='linear'),#, # метод опорных векторов с линейным ядром
	          LogisticRegression() # логистическая регрессия
	          ]

Xtrn, Xtest, Ytrn, Ytest = train_test_split(trn, trg, test_size=0.4)

#создаем временные структуры
TestModels = pd.DataFrame()
tmp = {}
#для каждой модели из списка
for model in models:
    #получаем имя модели
    m = str(model)
    tmp['Model'] = m[:m.index('(')]    
    #для каждого столбцам результирующего набора
    for i in  range(Ytrn.shape[1]):
        #обучаем модель
        model.fit(Xtrn,Ytrn.iloc[:,i]) 
        #вычисляем коэффициент детерминации
        tmp['R2_Y%s'%str(i+1)] = r2_score(Ytest.iloc[:,0], model.predict(Xtest))
    #записываем данные и итоговый DataFrame
    TestModels = TestModels.append([tmp])
#делаем индекс по названию модели
TestModels.set_index('Model', inplace=True)

plt.figure()
fig, axes = plt.subplots(ncols=2, figsize=(10,4))
TestModels.R2_Y1.plot(ax=axes[0], kind='bar', title='R2_Y1')
plt.savefig("D:\\dev\\go\\src\\finrgo\\learn\\reg1.png")
TestModels.R2_Y1.plot(ax=axes[1], kind='bar', color='green', title='R2_Y1')
plt.savefig("D:\\dev\\go\\src\\finrgo\\learn\\reg2.png")

