from tensorflow.keras.datasets import fashion_mnist
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, LSTM,Conv1D

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


print(y_test.shape)


x_train = x_train.reshape(x_train.shape[0], 28*28, 1)
x_test = x_test.reshape(x_test.shape[0], 28*28, 1) 

#2. 모델 구성


model = Sequential()
model.add(Conv1D(64, 2, input_shape=(x_test.shape[1],1)))
model.add(Flatten())
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(10, activation="softmax"))



#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)

import time
start_time = time.time()

hist = model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=10, batch_size=1000, validation_split=0.2, verbose=3,callbacks=[es])

end_time = time.time() - start_time


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print('accuracy', loss[1])
print("걸린시간 : ", end_time)

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
loss 1.0693103075027466
accuracy 0.9100000262260437

dnn

loss 0.6871653199195862
accuracy 0.871399998664856


LSTm
accuracy 0.10000000149011612

Conv1D
loss 0.5423886179924011
accuracy 0.8220000267028809

'''


#완성하시오
