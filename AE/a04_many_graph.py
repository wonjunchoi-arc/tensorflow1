
import numpy as np
from scipy.sparse.construct import random
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.engine.input_layer import Input
#1. 데이터
(x_train, _),(x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255

#.2 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense


def autoencoder1(hidden_layer_size): 
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, #dense의 아웃풋은 unit이다.
     input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model


model_01 = autoencoder1(hidden_layer_size=1)
model_02 = autoencoder1(hidden_layer_size=2)
model_04 = autoencoder1(hidden_layer_size=4)
model_08 = autoencoder1(hidden_layer_size=8)
model_16 = autoencoder1(hidden_layer_size=16)
model_32 = autoencoder1(hidden_layer_size=32)

print('###############node 1개 시작################')
model_01.compile(optimizer='adam', loss='binary_crossentropy')
model_01.fit(x_train, x_train, epochs=10)


print('###############node 2개 시작################')
model_02.compile(optimizer='adam', loss='binary_crossentropy')
model_02.fit(x_train, x_train, epochs=10)

print('###############node 4개 시작################')
model_04.compile(optimizer='adam', loss='binary_crossentropy')
model_04.fit(x_train, x_train, epochs=10)

print('###############node 8개 시작################')
model_08.compile(optimizer='adam', loss='binary_crossentropy')
model_08.fit(x_train, x_train, epochs=10)

print('###############node 16개 시작################')
model_16.compile(optimizer='adam', loss='binary_crossentropy')
model_16.fit(x_train, x_train, epochs=10)

print('###############node 32개 시작################')
model_32.compile(optimizer='adam', loss='binary_crossentropy')
model_32.fit(x_train, x_train, epochs=10)


output_01 = autoencoder1(hidden_layer_size=1)
output_02 = autoencoder1(hidden_layer_size=2)
output_04 = autoencoder1(hidden_layer_size=4)
output_08 = autoencoder1(hidden_layer_size=8)
output_16 = autoencoder1(hidden_layer_size=16)
output_32 = autoencoder1(hidden_layer_size=32)


from matplotlib import pyplot as plt
import random
fig, axes = plt.subplots(7,5,figsize=(15,15))

random_img = random.sample(range(output_01.shape[0]),5)
outputs = [x_test, output_01, output_02, output_04,output_08
,output_16,output_32]

for row_num, row in enumerate(axes):
    for col_num, ax in enumerate(row):
        ax.imshow(outputs[row_num][random_img[col_num]].reshape(28,28),
        cmap = 'gray')
        ax.grid(False)
        ax.set_xticks([])
        ax.set_yticks([])

plt.show()