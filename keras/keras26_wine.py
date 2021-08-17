import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_wine()



x= datasets.data
y = datasets.target

print(datasets.DESCR)
print(x.shape, y.shape)
print(y)

#1 .데이터

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y[:5])

'''
[[1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]
 [1. 0. 0.]]

'''

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)



from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer, RobustScaler
# scaler =MinMaxScaler()
# scaler = StandardScaler()
# scaler =QuantileTransformer()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_shape=(13,)))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(3, activation='softmax'))


#3. 컴파일 및 훈련


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 100, mode= 'min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=500,
 batch_size=8, validation_split=0.2, callbacks=[es])


#4. 예측 평가
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])

# y_predict = model.predict(x_test[:5])
# print('y_predict', y_predict)
# print(y_test[:5])

'''
1. min max
loss 0.21912741661071777
accuracy 0.9679999947547913

2. Standard
loss 0.07112745195627213
accuracy 0.9760000109672546

3. ROBust
loss 0.05244048312306404
accuracy 0.9919999837875366
'''

#완성하시오
#acc 0.8 이상 만들 것!! 