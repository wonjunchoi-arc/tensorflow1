# overfit 극복!!!
#1.  전체 훈련데이터를 마니 마니
#2.  normaliztion 정규화 시킨다. standardiztion과는 다름, minmax와 유사할지도?
# 여기서의 정규화는 레이어별로 해주는 것이 어떨가라는 접근 개념
#3.  dropout 완벽하게 fully connected 됫을때보다  node사이의 신경망을 몇개씩
#탈락 시켜준것이 오히려 더 효율적일 때가 있었다!!  잘라줄때도 일정하게 잘라주면 다시 그것에 과적합 될 수 있으므로
#랜덤하게 잘라 준다는 개념

import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar100
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D,Dropout,GlobalAveragePooling2D
from tensorflow.python.keras.datasets.mnist import load_data

(x_train, y_train), (x_test, y_test)= cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#  (10000, 32, 32, 3) (10000, 1)

# from tensorflow.keras.utils import to_categorical
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
#원핫 인코더할때 얘를 써도 괜찮다. 얘가 좀더 형식에서 자유롭기 때문이다. 

from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder(sparse=False)
y_train = en.fit_transform(y_train)
y_test = en.fit_transform(y_test)

x_train = x_train.reshape(50000,3072)
x_test = x_test.reshape(10000,3072)

# print(x_train[:1])

from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer,PowerTransformer
scaler =StandardScaler()
# scaler =MinMaxScaler()
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
model.add(Conv2D(32, kernel_size=(2,2), padding='same', input_shape=(32, 32, 3)))
model.add(Dropout(0.2)) # 20%만큼 노드를 랜덤으로 탈락시켜준다. 
model.add(Conv2D(32, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())

model.add(Conv2D(128, (2,2),padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(128, (2,2),padding='valid', activation='relu'))
# model.add(Dropout(0.2))
model.add(MaxPool2D())

model.add(Conv2D(64, (2,2),padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))
model.add(Dropout(0.2))
model.add(Conv2D(64, (2,2),padding='same', activation='relu'))
model.add(MaxPool2D())

# model.add(Flatten())
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
# model.add(Dropout(0.2))
# model.add(Dense(128, activation='relu'))
model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))

model.summary()


#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 10, mode= 'min', verbose=3)

import time
start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=60,
batch_size=150, validation_split=0.25,verbose=1)
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
loss 2.04148268699646
accuracy 0.47130000591278076
걸린시간 :  274.81814455986023

loss 1.987538456916809
accuracy 0.47760000824928284
걸린시간 :  242.58861804008484
'''
