from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np

#데이터 넘파이로 변환해서 땡겨오는 연습하자

#1. 데이터 제너레이터
train_datagen = ImageDataGenerator(
    rescale=1./255,
    # horizontal_flip=True,
    # vertical_flip=True,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    # rotation_range=5,
    # zoom_range=1.2,
    # shear_range=0.7,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
'../data/men_women',
target_size=(224,224),
batch_size=3400,
class_mode='binary'
)

test = test_datagen.flow_from_directory(
    '../data/wonjun',
target_size=(224,224),
batch_size=1,
class_mode='binary'
)

# print(xy_train[0][1])

np.save('./_save/_npy/image/CAE_x.npy', arr=xy_train[0][0])
np.save('./_save/_npy/image/CAE_y.npy', arr=xy_train[0][1])
# np.save('./_save/_npy/image/k59_wonjun_y.npy', arr=test[0][0])
