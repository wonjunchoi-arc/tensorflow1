#아웃라이어 확인
#

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

def outliers(data_out,column):
    quartile_1,q2,quartile_3 = np.percentile(data_out,[25,50,75])#분위수
    print("1사분위:",quartile_1)
    print("2사분위:",q2)
    print("3사분위:",quartile_3)
    iqr = quartile_3-quartile_1 #전체의 50% 범위에 해당
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    data1 = data_out[data_out[column] > upper_bound]     
    data2 = data_out[data_out[column] < lower_bound]  
    return np.where((data_out>upper_bound) | (data_out<lower_bound)), np.count

outliers_loc = outliers(x_train)
# outliers_count = outliers(x_train).count()
print("이상치의 위치: ", outliers_loc)
print("이상치의 갯수:",outliers_count)

#################################################아웃라이어의 갯수를 count하는 기능 추가할 것!################
#아웃라이어가 실측값이라 그렇게 큰 영향은 미치지 않을 것!

# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)


# #2. 모델
# model = XGBClassifier(n_jobs=-1)

# #3. 훈련
# model.fit(x_train, y_train)

# #4. 평가, 예측
# score = model.score(x_test,y_test)

# print("acc:", score)
