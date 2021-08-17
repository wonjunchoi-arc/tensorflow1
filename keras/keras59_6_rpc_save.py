
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


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
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

full = train_datagen.flow_from_directory(
    '../data/rps',
    target_size=(150,150),
    batch_size=3000,
    class_mode='categorical',
    classes=['paper,','rock','scissors'],

)

print(full[0][0])
print(full[0][1])
print(full.class_indices)
#{'paper,': 0, 'rock': 1, 'scissors': 2}

np.save('./_save/_npy/image/k59_rps_x.npy', arr=full[0][0])
np.save('./_save/_npy/image/k59_rps_y.npy', arr=full[0][1])
# np.save('./_save/_npy/image/k59_wonjun_y.npy', arr=test[0][0])
