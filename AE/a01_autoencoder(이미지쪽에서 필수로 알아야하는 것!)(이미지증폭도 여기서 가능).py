"""
이미지 쪽에서 특성을 잡아내거나 잡음?을 제거하는 것!!!
특성중에서 약한 특성은 삭제되고 강한 특성만 살아남는것
GAN하고 비교하면 사진이 뿌옇게 나온다는 특징을 가진다.

이미지의 라벨은 필요없이 x값이 들어갔다가 x값이 나온느 것이다.
"""

#앞뒤가 똑같은 오토인코더 ~~~

import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.engine.input_layer import Input
#1. 데이터
(x_train, _),(x_test, _) = mnist.load_data()

x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255

#.2 모델
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense

input_img = Input(shape=(784,))
# encoded = Dense(64, activation='relu')(input_img)
encoded = Dense(1064, activation='relu')(input_img)
# 64라는 보다 작은 층으로 갓따가 다시 784로 돌아가는 것이 오토인코더

decoded = Dense(784, activation='sigmoid')(encoded)
# decoded = Dense(784, activation='relu')(encoded)
# decoded = Dense(784, activation='linear')(encoded)
# decoded = Dense(784, activation='tanh')(encoded)
#위에서 값을 0~1로 변환시켜주었기 때문에 sigmoid

autoencoder = Model(input_img, decoded)


# autoencoder.summary()

autoencoder.compile(optimizer='adam', loss= 'mse')
autoencoder.compile(optimizer='adam', loss= 'binary_crossentropy')


autoencoder.fit(x_train, x_train, epochs=30, batch_size=128, validation_split=0.2 )

decoded_img = autoencoder.predict(x_test)

import matplotlib.pyplot as plt

n = 10
plt.figure(figsize=(20 ,4))
for i in range(n) : 
    ax =plt.subplot(2, n, i+1)
    plt.imshow(x_test[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    ax = plt.subplot(2, n, i+1+n)
    plt.imshow(decoded_img[i].reshape(28,28))
    plt.gray()
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

plt.show()
