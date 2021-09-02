
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D
from tensorflow.python import distribute
from tensorflow.python.keras.datasets.mnist import load_data
import tensorflow as tf


#1. data
(x_train, y_train), (x_test, y_test)= mnist.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#  (10000, 32, 32, 3) (10000, 1)


from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


x_train = x_train.reshape(60000,28,28,1).astype('float32')/255
x_test = x_test.reshape(10000,28,28,1).astype('float32')/255

# print(x_train[:1])


#분산처리 코드
# strategy = tf.distribute.MirroredStrategy()
strategy =tf.distribute.MirroredStrategy(cross_device_ops=  \
   tf.distribute.HierarchicalCopyAllReduce() )
#이코드가 맞아용~~

strategy = tf.distribute.MirroredStrategy(
    # devices=['/gpu:0'],
    # devices=['/gpu:1']
    devices=['/cpu','/gpu:0']
)
strategy =tf.distribute.experimental.CentralStorageStrategy()
strategy =tf.distribute.experimental.MultiWorkerMirroredStrategy()
strategy =tf.distribute.experimental.MultiWorkerMirroredStrategy(
    tf.distribute.experimental.CollectiveCommunication.RING
    # tf.distribute.experimental.CollectiveCommunication.NCCL
    # tf.distribute.experimental.CollectiveCommunication.AUTO
)
#위에꺼 뭘해도 상관없이 GPU분산처리 떄린다.





with strategy.scope():
# 모델 
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

    #3. 컴파일, 훈련 , metrics=['acc']
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])



hist = model.fit(x_train,y_train, epochs=100,
batch_size=100, validation_split=0.3,verbose=1)


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print('accuracy', loss[1])

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
