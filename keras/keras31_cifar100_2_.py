import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test)= cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#  (10000, 32, 32, 3) (10000, 1)



from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder(sparse=False)
y_train = en.fit_transform(y_train)
y_test = en.fit_transform(y_test)

x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)

# print(x_train[:1])

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer,PowerTransformer
# scaler =StandardScaler()
scaler =MinMaxScaler()
# scaler =RobustScaler()
# scaler = QuantileTransformer()
# scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



print(x_train.shape)
# print('=============================')


x_train = x_train.reshape(50000,32, 32, 3)
x_test = x_test.reshape(10000,32,32,3)

print(x_train.shape)

# 모델 
model =Sequential()
model.add(Conv2D(30, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(30, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())


model.add(Conv2D(120, (2,2),padding='same', activation='relu'))
model.add(Conv2D(240, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='accuracy', patience= 7, mode= 'max', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=5,
 batch_size=180, validation_split=0.01,verbose=3)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])


'''
1.
model =Sequential()
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(20, (2,2),padding='same', activation='relu'))
model.add(Conv2D(30, (2,2),padding='same', activation='relu'))
model.add(Conv2D(40, (2,2),padding='same', activation='relu'))
model.add(Conv2D(50, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(70, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))

313/313 [==============================] - 1s 3ms/step - loss: 3.1384 - accuracy: 0.2829     
loss 3.1383986473083496
accuracy 0.28290000557899475


2.
model =Sequential()
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(20, (2,2),padding='same', activation='relu'))
model.add(Conv2D(30, (2,2),padding='same', activation='relu'))
model.add(Conv2D(40, (2,2),padding='same', activation='relu'))
model.add(Conv2D(50, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(60, (2,2),padding='same', activation='relu'))
model.add(Conv2D(70, (2,2),padding='same', activation='relu'))
model.add(Conv2D(80, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

loss 13.946043968200684
accuracy 0.2865999937057495

클래시피케이션단을 건드려도 효율은

3.model =Sequential()
model.add(Conv2D(32, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(128, (2,2),padding='same', activation='relu'))
model.add(Conv2D(256, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(100, activation='softmax'))

313/313 [==============================] - 2s 7ms/step - loss: 10.4993 - accuracy: 0.3235
loss 10.499269485473633
accuracy 0.32350000739097595
PS D:\study> 

앞의 노드가 커질수록 효율 올라감

4.model =Sequential()
model.add(Conv2D(100, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(100, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(100, (2,2),padding='same', activation='relu'))
model.add(Conv2D(100, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

loss 7.595758438110352
accuracy 0.35690000653266907

노드별 계산후 max풀링을 통해 적당히 특징화 시켜주는 것이 더 정확하게 이미지를 찾는 방법 인것으로 여겨짐
그러므로 각 노드의 단계를 걸친후 max풀링을 해보자 1단계식 할때마다 max풀 해볼까?

5

model =Sequential()
model.add(Conv2D(100, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(100, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(130, (2,2),padding='same', activation='relu'))
model.add(Conv2D(130, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(260, (2,2),padding='same', activation='relu'))
model.add(Conv2D(260, (2,2),padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))
너무 커지니깐 오히려 값이 않좋아짐

6
이에 배치사이즈는1000-> 500 줄이고 각 노드를 줄인후 결과값 확인

model =Sequential()
model.add(Conv2D(64, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(100, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(130, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(260, (2,2),padding='same', activation='relu'))
model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))

loss 7.237964153289795
accuracy 0.35280001163482666

7

model =Sequential()
model.add(Conv2D(40, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(40, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(80, (2,2),padding='same', activation='relu'))
model.add(Conv2D(80, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(160, (2,2),padding='same', activation='relu'))
model.add(Conv2D(160, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 10, mode= 'min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=130,
 batch_size=500, validation_split=0.1,verbose=3)

 loss 8.095170974731445
accuracy 0.3668999969959259
PS D:\study> 

8  에포를 극단적으로 늘리고 거기에 얼리스탑을 걸어서 어디까지 내려갈지 보자
model =Sequential()
model.add(Conv2D(40, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(40, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(80, (2,2),padding='same', activation='relu'))
model.add(Conv2D(80, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Conv2D(160, (2,2),padding='same', activation='relu'))
model.add(Conv2D(160, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(100, activation='softmax'))


#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 20, mode= 'min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=200,
 batch_size=500, validation_split=0.005,verbose=3)

'''