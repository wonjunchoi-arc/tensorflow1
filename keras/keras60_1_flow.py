from tensorflow.keras.datasets import fashion_mnist
import numpy as np

(x_train, y_train),(x_test,y_test) =fashion_mnist.load_data()

from tensorflow.keras.preprocessing.image import ImageDataGenerator


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

# test_datagen = ImageDataGenerator(rescale=1./255)

# xy_train = train_datagen.flow_from_directory(
# '../data/men_women',
# target_size=(150,150),
# batch_size=3400,
# class_mode='binary'
# )







#1. ImageDataGenerator를 정의
#2. 파일에서 땡겨 올려면 -> flow_from_directory() // x,y가 튜플형태로 뭉쳐있음
#3. 데이터에서 땡겨 올려면  ->flow()              // x,y 가 나눠있어
augument_size=100
# 한장의 사진을 100장으로 바꾼것이다.  넣은 것이다. 
x_data = train_datagen.flow(
    np.tile(x_train[0].reshape(28*28), augument_size).reshape(-1,28,28,1),
    np.zeros(augument_size),
    batch_size=augument_size,
    shuffle=False
).next() #Iterator의 짝궁

'''
이미지들이 위를 통과하여 베치사이즈만큼 저장 된다. 그러므로 반대로 순서대로 하나 하나 끄내쓸 수 있는 
(NumpyArray)Iterator 형식으로 저장이 되는 것이다. 
'''

print(type(x_data)) #<class 'tensorflow.python.keras.preprocessing.image.NumpyArrayIterator'>
# Iterator 베치사이즈 만큼 하나 하나 반환한다.

# => <class 'tuple'>

print(type(x_data[0]))#<class 'tuple'>
# =>   <class 'numpy.ndarray'>

print(type(x_data[0][0]))#<class 'numpy.ndarray'>
print(x_data[0][0].shape)#(100, 28, 28, 1)
print(x_data[0][1].shape)#(100,)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,7))
for i in range(7):
    plt.subplot(1,7,1+i)  #앞에숫자가 행
    plt.axis('off')
    plt.imshow(x_data[0][i], cmap='gray')

plt.show()



