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


def outliers_count(data_out,column):
    quartile_1,q2,quartile_3 = np.percentile(data_out[column],[25,50,75])#분위수
    print("1사분위:",quartile_1)
    print("2사분위:",q2)
    print("3사분위:",quartile_3)
    iqr = quartile_3-quartile_1 #전체의 50% 범위에 해당

    # outlier cutoff 계산하기      
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)


    # lower와 upper bound 값 구하기   
    data1 = data_out[data_out[column] > upper_bound]     
    data2 = data_out[data_out[column] < lower_bound]  
    return data1, data2
# def outlier_iqr(data, column): 

#     # lower, upper 글로벌 변수 선언하기     
#     global lower, upper    
    
#     # 4분위수 기준 지정하기     
#     q25, q75 = np.quantile(data[column], 0.25), np.quantile(data[column], 0.75)          
    
#     # IQR 계산하기     
#     iqr = q75 - q25    
    
#     cut_off = iqr * 1.5          
    

data1, data2 = outliers_count(datasets,'volatile acidity')
print('data1',data1)
print('총 이상치 개수는', data1.shape[0] + data2.shape[0], '이다.')

"""
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


def outliers(data_out):
    quartile_1,q2,quartile_3 = np.percentile(data_out,[25,50,75])#분위수
    print("1사분위:",quartile_1)
    print("2사분위:",q2)
    print("3사분위:",quartile_3)
    iqr = quartile_3-quartile_1 #전체의 50% 범위에 해당
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))


outliers_loc = outliers(x_train)
print("이상치의 위치: ", outliers_loc)

# '''
#  #   Column                Non-Null Count  Dtype
# ---  ------                --------------  -----
#  0   fixed acidity         4898 non-null   float64
#  1   volatile acidity      4898 non-null   float64
#  2   citric acid           4898 non-null   float64
#  3   residual sugar        4898 non-null   float64
#  4   chlorides             4898 non-null   float64
#  5   free sulfur dioxide   4898 non-null   float64
#  6   total sulfur dioxide  4898 non-null   float64
#  7   density               4898 non-null   float64
#  8   pH                    4898 non-null   float64
#  9   sulphates             4898 non-null   float64
#  10  alcohol               4898 non-null   float64
#  11  quality               4898 non-null   int64
# '''
# #################################################아웃라이어의 갯수를 count하는 기능 추가할 것!################
# #아웃라이어가 실측값이라 그렇게 큰 영향은 미치지 않을 것!

# # scaler = StandardScaler()
# # scaler.fit(x_train)
# # x_train = scaler.transform(x_train)
# # x_test = scaler.transform(x_test)


# # #2. 모델
# # model = XGBClassifier(n_jobs=-1)

# # #3. 훈련
# # model.fit(x_train, y_train)

# # #4. 평가, 예측
# # score = model.score(x_test,y_test)

# # print("acc:", score)
"""