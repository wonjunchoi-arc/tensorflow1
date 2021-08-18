from re import X
import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from xgboost import XGBRegressor, XGBClassifier



datasets = pd.read_csv('../data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)


# print(datasets)
print(datasets.shape) #(4898, 12)
print(datasets.info())
print(datasets.describe())
print(datasets.head())

# #1. 데이터 처리
datasets=datasets.values
print(type(datasets))

from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split

x= datasets[:,:-1]
y= datasets[:,-1:]
print(x.shape, y.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,random_state=66,shuffle=True, train_size=0.8
)

scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델
model = XGBClassifier(n_jobs=-1)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test,y_test)

print("acc:", score)
