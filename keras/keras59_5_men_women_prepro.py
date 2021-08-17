# 실습 1. 
#men women데이터로 모델링

# 실습 2. 본인 사진으로 predict 하시오. 과제

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#데이터 넘파이로 변환해서 땡겨오는 연습하자

#1. 데이터 제너레이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=5,
    zoom_range=1.2,
    shear_range=0.7,
    fill_mode='nearest',
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
'../data/men_women',
target_size=(150,150),
batch_size=10000,
class_mode='binary',
shuffle=True,
)

y = xy_train[0][1]
 
print(y)
print(len(xy_train))
#Found 2482 images belonging to 2 classes.


# test = train_datagen.flow_from_directory(
#     '../data/men_women',
# target_size=(150,150),
# batch_size=100,
# class_mode='binary',
# shuffle=True,
# )
#Found 827 images belonging to 2 classes.

x = []
y = []
x_test = []
y_test = []

for i in range(len(xy_train)):
    x.append(xy_train[i][0])
    y.append(xy_train[i][1])

# for i in range(len(test)):
#     x_test.append(test[i][0])
#     y_test.append(test[i][1])



# print(x_train.shape)
# print(x_train)
# print(type(x_train))

np.save('./_save/_npy/image/k59_men_women_x_train.npy', arr=x)
np.save('./_save/_npy/image/k59_men_women_y_train.npy', arr=y)
# np.save('./_save/_npy/image/k59_men_women_x_test.npy', arr=x_test)
# np.save('./_save/_npy/image/k59_men_women_y_test.npy', arr=y_test)



