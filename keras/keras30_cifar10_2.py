import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test)= cifar10.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#  (10000, 32, 32, 3) (10000, 1)



from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)

# print(x_train[:1])

from sklearn.preprocessing import MinMaxScaler
scaler =MinMaxScaler()
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


#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 10, mode= 'min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=50,
 batch_size=2000, validation_split=0.1,verbose=3)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])


'''
1.
model =Sequential()
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(30, (2,2), activation='relu'))
model.add(Conv2D(40, (2,2), activation='relu'))
model.add(Conv2D(50, (2,2), activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(70, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16))
model.add(Dense(10, activation='softmax'))


#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 10, mode= 'min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=50,
 batch_size=2000, validation_split=0.2,verbose=3)


Epoch 50/50
313/313 [==============================] - 1s 3ms/step - loss: 1.4212 - accuracy: 0.6505
loss 1.421204686164856
accuracy 0.6504999995231628


2.
1번과 동일 하지만 val split을 0.1로 줄임

313/313 [==============================] - 1s 3ms/step - loss: 1.3765 - accuracy: 0.6651
loss 1.3764944076538086
accuracy 0.6650999784469604

3. 2번의 방식에서 kernel size만 3,3d으로 늘려봄
연산속도는 빨라지는 듯 하나 정확도는 떨어지는 모습을 보임

loss 1.783265233039856
accuracy 0.6360999941825867

4. 2번의 방식에 5번까지의 conv에 padding을 줘 
가장자리 데이터의 손실을 최소화하기 위해 노력


'''