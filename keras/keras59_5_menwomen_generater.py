import numpy as np

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
    validation_split=0.25

)

test_datagen = ImageDataGenerator(rescale=1./255)

xy_train = train_datagen.flow_from_directory(
'../data/men_women',
target_size=(150,150),
batch_size=100,
class_mode='binary',
shuffle=True,
)
print(len(xy_train))

val = train_datagen.flow_from_directory(
'../data/men_women',
target_size=(150,150),
batch_size=100,
class_mode='binary',
shuffle=True,
subset = 'validation'

)
'''
================================================

'''

# x= np.load('./_save/_npy/image/k59_men_women_x_train.npy')
# y = np.load('./_save/_npy/image/k59_men_women_y_train.npy')
predict=np.load('./_save/_npy/image/k59_wonjun_y.npy')

# from sklearn.model_selection import train_test_split
# x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7) 

# print(x_train.shape, x_test.shape)
# print(y_train.shape, y_test.shape)

# print(x_train.shape, x_test.shape) #(1400, 150, 150, 3) (600, 150, 150, 3)
# print(y_train.shape, y_test.shape) # (1400,) (600,)

# x_train = x_train.reshape(1400,150*150*3)
# x_test = x_test.reshape(600,150*150*3)
# predict = predict.reshape(1,150*150*3)


# from sklearn.preprocessing import StandardScaler,MinMaxScaler
# scaler =MinMaxScaler()
# # scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# predict = scaler.transform(predict)


# x_train = x_train.reshape(772*3,150,150,3)
# x_test = x_test.reshape(331*3,150,150,3)
# predict = predict.reshape(1,150,150,3)


#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPool2D,GlobalAveragePooling2D,MaxPooling2D


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 16, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation= 'sigmoid'))

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=50, mode='min', verbose=1,
                        restore_best_weights=True) #break 지점의 weight를 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                        filepath='./_save/men_women/image_mcp.hdf5')


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit_generator(xy_train, epochs=5,verbose=1,
steps_per_epoch=100,
validation_data=val,
validation_steps=100,
callbacks=[es])

model.save('./_save/men_women/image_model.h5')


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#4.
loss =model.evaluate_generator(val)
print('acc', acc[-1])
print('loss', loss[0])

y_predict = model.predict(predict)
y_predict = 1-y_predict *100
print('나는 ',y_predict,'%확률로 남자입니다.',)



