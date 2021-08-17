##데이터 전처리의 중요성 종속변수를 종속변수의 최대값으로 나누기!!

from re import M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import random

from sklearn.datasets import load_boston
datasets= load_boston()
x= datasets.data
y = datasets.target


print(np.min(x), np.max(x)) #0.0 711.0

#711.0 같은 경우에는 연산이 많이 질 수록 그 수가 한없이 커지기 때문에 얘를 
#0~1사이 값으로 바꾸는 것이 좋다. 즉 숫자를 0.~단위로 만들어주게 나눠주고 연산 결과가 끝나고 
#원래값으로 돌려주는 방법이 있다. —> 이렇게 해줘도 각 데이터 간의 비율을 바뀌지 않기 때문에 이것이 성립하는 것이다. 


#데이터 전처리
# x = x/ 711.
# x = x/np.max(x)
# x = (x - np.min(x))/ (np.max(x) - np.min(x)) 
#얘를 column별로 정규화 해주는 것이 좋다!! for 문을 가져다 쓰면 좋을 듯
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
train_size=0.7, random_state=66)


from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#이는 test 데이터는 train데이터에 관여하면 안된다는거

 #실행 시킨 애를 변환시킴
# 텐서에서 훈련시키다 실행시키다는 fit으로 조진다.

# print(np.min(x_scale), np.max(x_scale))



print(x.shape) #(506, 13)
print(y.shape) #(506,)





#정규화 하는 것에 대해서 train할 데이터만 정규화를 해야한다. 
#왜냐하면 우리가 미래를 예측할 데이터를 어떻게 알고 전부 정규화를 시켜놓고 트레인 시키겠는가?

# print(x_train)



#2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=13))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(1))


from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 5, mode= 'min', verbose=1)
#최저값에서 더떨어지지 않고 5번을 버티면 거기서 멈추겠다. 만약에 더 떨어지면 거기서 다시 5번을 카운팅함.
#얘를 fit에서 적용함.

#3 . 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train, epochs=100, batch_size=8, 
verbose=1, validation_split=0.2, callbacks=[es])

print(hist)
#<tensorflow.python.keras.callbacks.History object at 0x000001538E8672E0>

print(hist.history.keys())
#dict_keys(['loss', 'val_loss'])
print("-------------------------------")
print(hist.history['loss'])
print("-------------------------------")
print(hist.history['val_loss'])

print("-------------------------------")

'''
이런식으로 history란 곳에 loss와 val_loss 데이터가 저장되어 있음
[381.166259765625, 50.29790496826172, 25.839494705200195, 19.122135162353516, 18.09088897705078,  8.856392860412598, 9.058024406433105, 8.673643112182617, 9.060768127441406]
-------------------------------
[93.88968658447266, 36.961612701416016, 32.81143569946289, 28.218332290649414, 28.1376953125,  23.84251594543457, 21.305097579956055, 27.417341232299805]

'''
from matplotlib import font_manager, rc
font_path = "C:\Windows\Fonts\H2GSRB.TTF"
font = font_manager.FontProperties(fname=font_path).get_name()
rc('font', family=font)

plt.plot(hist.history['loss'])   # x: epoch/ y : hist.history['loss']
plt.plot(hist.history['val_loss'])   # x: epoch/ y : hist.history['loss']

plt.title("로스, 발로스")
plt.xlabel('epochs')
plt.ylabel("loss, val_loss")
plt.show()


# #4. 평가 예측
# loss = model.evaluate(x_test, y_test)
# print('loss', loss)

# y_predict = model.predict(x_test)
# print('y_predict', y_predict)


# from sklearn.metrics import r2_score
# r2 = r2_score(y_test,y_predict)
# print('r2스코어',r2)



''' 
전처리 이전
r2스코어 0.5861025717155626

전처리 이후
r2스코어 0.8346791942618095

validation 때는 왜 r2가 낮지?

minmax 이후
r2스코어 0.9298092798682422

train data minmax 이후
r2스코어 0.9269240242045239

45/45 [==============================] - 0s 4ms/step - loss: 5.3086
Epoch 34/100
45/45 [==============================] - 0s 3ms/step - loss: 5.3088
Epoch 35/100
45/45 [==============================] - 0s 4ms/step - loss: 5.8397
Epoch 36/100
45/45 [==============================] - 0s 4ms/step - loss: 5.2758
Epoch 37/100
45/45 [==============================] - 0s 4ms/step - loss: 4.9039
Epoch 38/100
45/45 [==============================] - 0s 3ms/step - loss: 5.1553
Epoch 00038: early stopping
5/5 [==============================] - 0s 2ms/step - loss: 7.6915
loss 7.691516876220703
y_predict [[ 9.939454 ]

위의 loss값들은 연산되는 것에 비해 출력되는 것이 정확하지 않을 수 있기 때문에 early stopping의 기준을 val loss잡는 것이
조금더 정확할 수 있다. 


5/5 [==============================] - 0s 2ms/step - loss: 13.3899
loss 13.389930725097656
마지막줄 loss는 evaluate에 대한 loss값 출력 

위에서 5/5는 test의 데이터를 배치사이즈로 돌려서 나온것이다. 
evaluater의 batch size 는 32로 정해져 있기 때문이다. 
'''



# print(datasets.feature_names) #['CRIM' 'ZN' 'INDUS' 'CHAS' 'NOX' 'RM' 'AGE' 'DIS' 'RAD' 'TAX' 'PTRATIO''B' 'LSTAT']
# print(datasets.DESCR)

#loss 출력 v , R2출력 v, 컬럼 어떻게 생격는지도 확인해보자