#conv로 만들어라!1


import numpy as np
from scipy.sparse.construct import random
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.engine.input_layer import Input
import tensorflow as tf
#1. 데이터

x= np.load('./_save/_npy/image/CAE_x.npy')

from sklearn.model_selection import train_test_split
x_train,x_test = train_test_split(
    x, random_state=66, shuffle=True
)

print(x_train.shape,x_test.shape)


# x_train = x_train.reshape(60000,784).astype('float')/255
# x_test = x_test.reshape(10000,784).astype('float')/255


x_train = x_train.reshape(2481,224,224,3).astype('float')
x_test = x_test.reshape(828,224,224,3).astype('float')
#255로 안나눠주는 이유는 이미지 제너레이터에서 이미 255로 나눠줬기 때문이다.


# x_train = tf.image.resize(x_train,(224,224))
# x_test = tf.image.resize(x_test,(224,224))

# print(x_train.shape,x_test.shape)


x_train_noised = x_train + np.random.normal(0, 0.1, size=x_train.shape)
#normal이 정규분포에 따른 랜덤값을 넣는건지 찾아보자!!
x_test_noised = x_test + np.random.normal(0,0.1, size=x_test.shape)

x_train_noised = np.clip(x_train_noised, a_min=0, a_max=1)
x_test_noised = np.clip(x_test_noised, a_min=0, a_max=1)
#clip은 최솟값을 벗어나는 놈을 0으로 최댓값을 벗어나는 놈을 1로 클립시켜버린다.




#.2 모델
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPool2D, UpSampling2D


def autoencoder(hidden_layer_size): 
    model = Sequential()
    model.add(Dense(units=hidden_layer_size, #dense의 아웃풋은 unit이다.
     input_shape=(784,), activation='relu'))
    model.add(Dense(units=784, activation='sigmoid'))
    return model

def autoencoder2(hidden_layer_size):
    input = Input(shape=(224, 224, 3))
    x=Conv2D(64,(2,2) #dense의 아웃풋은 unit이다.
    , activation='relu',padding='same')(input)
    print(x)
    x=MaxPool2D((2,2))(x)
    x = Conv2D(128,(2,2), activation='relu',padding='same')(x)
    x = Conv2D(128,(2,2), activation='relu',padding='same')(x)

    x= MaxPool2D((2,2))(x)
    x = Conv2D(128,(2,2), activation='relu',padding='same')(x)
    x = Conv2D(128,(2,2), activation='relu',padding='same')(x)
    


    
    
    x = Conv2D(128,(2,2), activation='relu',padding='same')(x)
    x= UpSampling2D((2,2))(x)
    x = Conv2D(64,(2,2), activation='relu',padding='same')(x)
    x = Conv2D(32,(2,2), activation='relu',padding='same')(x)
    x = Conv2D(16,(2,2), activation='relu',padding='same')(x)
    x = UpSampling2D((2,2))(x)
    x = Conv2D(3,(2,2),padding='same')(x)
    model = Model(inputs=input, outputs=x)

    model.summary() 

    return model

# model = autoencoder(hidden_layer_size=54) # pca 95%
model = autoencoder2(hidden_layer_size=1) # pca 95%


model.compile(optimizer='adam', loss='mse')

model.fit(x_train_noised, x_train, epochs =100)

output = model.predict(x_test_noised)

from matplotlib import pyplot as plt
import random

fig, ((ax1, ax2, ax3, ax4, ax5),(ax11,ax12, ax13, ax14, ax15),(ax6,ax7, ax8, ax9, ax10)) = \
    plt.subplots(3, 5, figsize=(20,7))


#이미지 5개를 무작위로 고른다.
random_images = random.sample(range(output.shape[0]),5)

#원본(입력) 이미지를 맨 위에 그린다.
for i, ax in enumerate([ax1, ax2, ax3, ax4, ax5]):
    ax.imshow((x_test[random_images[i]]))
    # ax.imshow(x_test[random_images[i]].reshape(224,224,3).astype(np.uint8), cmap='gray')
    if i ==0:
        ax.set_ylabel("INPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

for i, ax in enumerate([ax11, ax12, ax13, ax14, ax15]):
    ax.imshow((x_test_noised[random_images[i]]))
    # ax.imshow(x_test_noised[random_images[i]].reshape(224,224,3).astype(np.uint8), cmap='gray')
    if i ==0:
        ax.set_ylabel("NOISED", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])


#오토인코더가 출력한 이미지를 아래에 그린다.
for i, ax in enumerate([ax6, ax7, ax8, ax9, ax10]):
    ax.imshow((output[random_images[i]]))
    print(output[random_images[i]].shape)
    print(type(output[random_images[i]]))
    print(np.max(output[random_images[i]]))

    # ax.imshow(output[random_images[i]].reshape(224,224,3).astype(np.uint8), cmap='gray')
    if i ==0:
        ax.set_ylabel("OUTPUT", size=20)
    ax.grid(False)
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()

    
