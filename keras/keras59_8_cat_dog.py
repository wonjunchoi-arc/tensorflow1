
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
    fill_mode='nearest',
    validation_split=0.25
    
)

test_datagen = ImageDataGenerator(rescale=1./255)

train = train_datagen.flow_from_directory(
    '../data/cat_dog/training_set/training_set',
    target_size=(224,224),
    batch_size=100,
    class_mode='binary',
    classes=['cats,','dogs'],
    subset = 'training'

)

val = test_datagen.flow_from_directory(
    '../data/cat_dog/training_set/training_set',
    target_size=(224,224),
    batch_size=100,
    class_mode='binary',
    classes=['cats,','dogs'],
    # subset = 'validation'

)


test = test_datagen.flow_from_directory(
    '../data/cat_dog/test_set/test_set',
    target_size=(224,224),
    batch_size=32,
    class_mode='binary',
    classes=['cats,','dogs'],

)



# print(full[0][0])
# print(full[0][1])
# print(full.class_indices)
#{'paper,': 0, 'rock': 1, 'scissors': 2}

# np.save('./_save/_npy/image/k59_rps_x.npy', arr=full[0][0])
# np.save('./_save/_npy/image/k59_rps_y.npy', arr=full[0][1])
# np.save('./_save/_npy/image/k59_wonjun_y.npy', arr=test[0][0])


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPool2D,GlobalAveragePooling2D

#2. 모델

model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(128, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))


model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation='sigmoid'))

model.summary()

#.3 모델

# #3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1,
                        restore_best_weights=True) #break 지점의 weight를 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                        filepath='./_save/men_women/image_mcp.hdf5')


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc',])
hist = model.fit(train, epochs=25,
steps_per_epoch=10, 
validation_data=val,
validation_steps=10,
callbacks=[es,])

model.save('./_save/men_women/image_model.h5')


acc = hist.history['acc']
val_acc = hist.history['val_acc']
val_loss = hist.history['val_loss']
loss = hist.history['loss']

loss =model.evaluate(test)
print('loss', loss[1])
print('val_loss', val_loss[-1])
print('val_acc', val_acc[-1])