import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np


#1. 데이터 제너레이터
train_datagen1 = ImageDataGenerator(
    rescale=1./255,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
    rotation_range=1,
    zoom_range=1,
    
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

#지정한 이미지 제너레이터로 파일 불러오기

full1 = train_datagen1.flow_from_directory(
    '../data/disc/discs/',
    target_size=(150,150),
    batch_size=555,
    shuffle=True,
    class_mode='categorical',
    classes=['disc,','good','neck_disc'],  #3가지 항목 이름 명시
)

test = test_datagen.flow_from_directory(
     '../data/disc/discs/',
    target_size=(150,150),
    batch_size=1,
    shuffle=True,
    class_mode='categorical',
)

print(full1[0][0])
print(full1[0][1])
print(full1.class_indices)

x= full1[0][0]
y = full1[0][1]
predict = test[0][0]

# 맨 


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,shuffle=True,random_state=9) 

print(x_train.shape, x_test.shape) #(129, 150, 150, 3) (56, 150, 150, 3)
print(y_train.shape, y_test.shape) # (129, 3) (56, 3)


# # from sklearn.preprocessing import StandardScaler,MinMaxScaler
# # scaler =MinMaxScaler()
# # # scaler = StandardScaler()
# # # scaler.fit(x_train)
# # # x_train = scaler.transform(x_train)
# # # x_test = scaler.transform(x_test)
# # # predict = scaler.transform(predict)


# # x_train = x_train.reshape(1176,150,150,3)
# # x_test = x_test.reshape(504,150,150,3)
# # # predict = predict.reshape(1,150,150,3)


# # #2. 모델
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPool2D,GlobalAveragePooling2D

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150, 150, 3)))
model.add(Conv2D(16, (2,2)))
model.add(MaxPool2D(2,2))

model.add(Conv2D(8, (2,2), activation='relu'))
model.add(Conv2D(4, (2,2), activation='relu'))

model.add(GlobalAveragePooling2D())
model.add(Dense(3, activation='softmax'))

model.summary()

# #3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=20, mode='min', verbose=1,
                        restore_best_weights=True) #break 지점의 weight를 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                        filepath='./_save/men_women/image_mcp.hdf5')


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=100, batch_size=50
, 
validation_split=0.3,
# steps_per_epoch=4,
# validation_steps=4,
callbacks=[es,])

# model.save('./_save/men_women/image_model.h5')


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

model = load_model('./_save/men_women/image_mcp.hdf5')


# #4.  모델 평가 및 예측
loss =model.evaluate(x_test,y_test)
print('loss', loss[1])
print('val_loss', val_loss[-1])
print('val_acc', val_acc[-1])

pose = {0:'disc', 1:'Correct_pose',2:'neck_disc'}

y_predict = model.predict(predict)
y_predict =np.argmax(y_predict)
print(pose[y_predict])
