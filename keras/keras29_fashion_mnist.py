from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape) #(10000, 28, 28) (10000,)

print(np.unique(y_test))

x_train = x_train.reshape(60000, 784)  
x_test = x_test.reshape(10000, 784)
y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)

from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder()
y_train =en.fit_transform(y_train).toarray()
y_test =en.fit_transform(y_test).toarray()

x_train = x_train.reshape(60000, 28, 28, 1)
x_test = x_test.reshape(10000, 28, 28, 1)

print(y_test.shape)

#1. 데이터 전처리


# #2. 모델 구성
model = Sequential()
model.add(Conv2D(10, kernel_size=(2,2), padding='same', input_shape=(28,28,1)))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(30, (2,2), activation='relu'))
model.add(Conv2D(40, (2,2), activation='relu'))
model.add(Conv2D(50, (2,2), activation='relu'))
model.add(MaxPool2D())   
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(10, activation='softmax'))




#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss', patience=10, mode='min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=100, batch_size=1000, validation_split=0.2, verbose=3)



#4. 평가 및 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])

'''
loss 1.0693103075027466
accuracy 0.9100000262260437
'''


#완성하시오
