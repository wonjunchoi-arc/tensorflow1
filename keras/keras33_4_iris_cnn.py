## 다중분류 모델!!! y값을 어떻게 숫자의 데이터 값이 아닌 라벨화 시킬지 !
# 값이 있는 위치에만 1을 지정해준다. 즉 영이란 값은 첫번째 자리에 1의 값을
#1은 두번째 자리에서 1의 값을, 2의 값은 2번째 자리에서 1의 값을 주는 것이다. !

import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, GlobalAveragePooling2D


datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
print(np.unique(y))
'''
데이터 확인 셔플 안하면 조지는 데이터가 있을 수 있음 이렇게
[0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 2 2 2 2 2 2 2 2 2 2 2
 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2
 2 2]
'''

#원핫인코딩 One Hot Encoding (150, 0) -> (150 , 3)
# 0 -> [1, 0, 0]
# 1 -> [0, 1, 0]
# 2 -> [0, 0, 1]

#[0,1,2,1]
#[[1,0,0]
#[0,1,0]
#[0,0,1]
#[0,1,0]]

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)
print(y[:5])



from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, train_size=0.7, random_state=66)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler =MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)


x_train = x_train.reshape(105, 2, 2, 1)
x_test = x_test.reshape(45,2,2,1)


#2. 모델 구성
model = Sequential()
model.add(Conv2D(30, kernel_size=(2,2), padding='same', input_shape=(2, 2, 1)))
model.add(Conv2D(30, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(30, (2,2),padding='same', activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='softmax'))



#3. 컴파일 및 훈련

from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 5, mode= 'min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=600,
 batch_size=8, validation_split=0.2, callbacks=[es])



#4. 예측 평가
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])

y_predict = model.predict(x_test[:5])
print('y_predict', y_predict)
print(y_test[:5])



# import matplotlib.pyplot as plt

# plt.plot(hist.history['loss'])   # x: epoch/ y : hist.history['loss']
# plt.plot(hist.history['val_loss'])   # x: epoch/ y : hist.history['loss']

# plt.title("loss, val_loss")
# plt.xlabel('epochs')
# plt.ylabel("loss, val_loss")
# plt.legend(['train loss','val loss'])
# plt.show()

'''
dnn
loss 0.21113178133964539
accuracy 0.8666666746139526

cnn 
loss 0.0979764312505722
accuracy 0.9555555582046509
'''