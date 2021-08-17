from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train, y_train),(x_test,y_test) =fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator

print(x_train.shape)

#1. 데이터 제너레이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=False,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.5,
    fill_mode='nearest'
)





#1. ImageDataGenerator를 정의
#2. 파일에서 땡겨 올려면 -> flow_from_directory() // x,y가 튜플형태로 뭉쳐있음
#3. 데이터에서 땡겨 올려면  ->flow()              // x,y 가 나눠있어

agument_size=40000

randidx = np.random.randint(x_train.shape[0], size =agument_size)
print(x_train.shape[0])
print(randidx)
print(randidx.shape)

x_agmented = x_train[randidx].copy()
y_agmented = y_train[randidx].copy()

print(x_agmented.shape)

x_augmented = x_agmented.reshape(x_agmented.shape[0],28,28,1)

x_train = x_train.reshape(x_train.shape[0],28,28,1)
x_tset = x_test.reshape(x_test.shape[0],28,28,1)



x_agmented =train_datagen.flow( #flow는 4차원을 받아들이고 싶어!! 근데 너는 3차원을 넣었어!!
    x_train, y_train,
    batch_size=agument_size,shuffle=False
).next()[0] #이렇게 하면 x값만 빠지겠지?

# x_train = np.concatenate((x_train,x_agmented))
# y_train = np.concatenate((y_train,y_agmented))

print(x_train.shape) #(100000, 28, 28, 1)
print(y_train.shape) #(100000,)


#x_ agmented 10개와 원래 x_train 10개를 비교하는 이미지 출력할 것

import matplotlib.pyplot as plt
for i in range(10):
    plt.subplot(2,10,i+1)
    plt.axis('off')
    plt.imshow(x_train[i], cmap='gray')
    plt.subplot(2,10,i+11)
    plt.axis('off')
    plt.imshow(x_agmented[i], cmap='gray')


plt.show()
