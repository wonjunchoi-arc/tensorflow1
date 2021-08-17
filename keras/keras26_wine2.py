from re import X
import numpy as np
import pandas as pd
from pandas.core.arrays import categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense



datasets = pd.read_csv('../data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)
#데이터가 ; 으로 분리 되어 있어서 sep 세퍼레이트 를 해준다..
# 데이터가 a열에 있었으므로 인덱스 컬럼은 없고 맨위의 행이 head기 때문에 0을 줌 

# winequality-white
# ./ : 현재 폴더
# ../ : 상위폴더 

# print(datasets)
print(datasets.shape) #(4898, 12)
print(datasets.info())
print(datasets.describe())

#1. 데이터 처리
datasets=np.array(datasets.values)
# print(datasets.shape)

x= datasets[:,:-1]
y= datasets[:,-1:]
print(x.shape, y.shape)

print(np.unique(y))

from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder()
y = en.fit_transform(y).toarray()

# from tensorflow.keras.utils import to_categorical
# y = to_categorical(y)

print(y.shape)
print(y[:5])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.1, random_state=66)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer, RobustScaler
# scaler =MinMaxScaler()
# scaler = StandardScaler()
# scaler =QuantileTransformer()
scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(240, input_shape=(11,)))
model.add(Dense(120, activation='relu'))
model.add(Dense(120, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(7, activation='softmax'))


#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience= 100, mode= 'min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=300,
 batch_size=50, validation_split=0.2,verbose=3)

#4. 예측 평가
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])



# # dataset 라벨 , 셋 자체에 대한 처리로 정확도 올릴 수 있음 

# #다중분류 
# #모델링하고
# #0.8 이상 완성!!
# #sklearn의 onehot 사용할것
# #4. y의 라벨을 확인할 것 np.unique(y)
# #5. y의 shape 확인 (4898,) -> (4898,7)