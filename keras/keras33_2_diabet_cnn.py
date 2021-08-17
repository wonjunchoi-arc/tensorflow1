# 실습 diabets
#1.  loss와 R2로 평가함
#2. MinMax와 Stanard 결과들 표시

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPool1D,GlobalAveragePooling1D,Dropout
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape , y.shape)
print(y.shape)
print(np.unique(y))
y = y.reshape(442,1)

# # # print(x[:,1])


# print(datasets.feature_names)
# #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']        
# print(datasets.DESCR)
# print(y[:30])
# print(np.min(y), np.max(y))

x_train, x_test, y_train, y_test =train_test_split(
    x,y, train_size=0.7,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer
scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)
print(x_train.shape)
print(x_test.shape)


x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1)


# # print(x_train.shape)

# #2. 모델 구성
model = Sequential()
model.add(Conv1D(30, kernel_size=2, padding='same', input_shape=(10,1)))
model.add(Dropout(0.2))
model.add(Conv1D(30, 2,padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(30, 2,padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv1D(30, 2,padding='same', activation='relu'))
model.add(MaxPool1D())
model.add(GlobalAveragePooling1D())
model.add(Dense(1,))

#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 50, mode= 'min', verbose=3)

import time
start_time = time.time()

model.compile(loss='mse', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=1000,
batch_size=10, validation_split=0.3,verbose=1),callbacks=[es]
end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print('accuracy', loss[1])
print("걸린시간 : ", end_time)

y_predict = model.predict(x_test)

from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2', r2)

#시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(9,5))

#1
plt.subplot(2,1,1)
plt.plot(hist.history['loss'], marker='.', c='red', label='loss')
plt.plot(hist.history['val_loss'], marker='.', c='blue', label='val_loss')
plt.grid()
plt.title('loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(loc='upper right')

#2
plt.subplot(2,1,2) # 2개를 하고 1행 2열
plt.plot(hist.history["acc"])
plt.plot(hist.history['val_acc'])
plt.grid()
plt.title('acc')
plt.ylabel('acc')
plt.xlabel('epoch')
plt.legend(['acc', 'val_acc'])
plt.show()





'''
1. Standard
loss 5788.34912109375
r2 -0.026863795789244582

2. MinMax 
loss 3219.92431640625
r2 0.3976042539059982  여기가 높게 나오는걸 봐서는 데이터가 넓게 골고루 퍼져있지는 않은것으로 보인다. 



cnn

5/5 [==============================] - 0s 2ms/step - loss: 4771.6426 - acc: 0.0000e+00
loss 4771.642578125
acc 0.0
r2 0.2528121291409815


'''






# plt.scatter(x[:,7],y)
# plt.show()
