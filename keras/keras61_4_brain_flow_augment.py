#훈련 데이터를 기존의 20%증가 
#성과 비교
#save dir할 것

import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 제너레이터는 1. 데이터를 수치화, 2. 데이터를 증폭하는 기능을 가짐


#1. 데이터 제너레이터를 정의하는 과정
train_datagen= ImageDataGenerator( #train은 증폭시켜서 훈련을 해야겠죠!! 훈련하는 양이 많을 수록 좋자나여!!
    rescale=1./255,
    horizontal_flip=True, #수평으로 이동하겠냐
    vertical_flip=True,
    width_shift_range=0.1, #좌우로 0.1만큼 움직여서 증폭시키겠냐
    height_shift_range=0.1, # 상하로
    rotation_range=5, # 반전시켜서
    zoom_range=1.2, #원래 이미지에서 20정도 더 크게 
    shear_range=0.7,
    fill_mode='nearest', # 내사진을 옆으로 조금 옮기면 공백이 생기는 데 그것을 비슷한 애들로 채우겠당
)

test_datagen = ImageDataGenerator(rescale= 1./255 #검사하려는 애를 바꾸면 안되겠죵
)

# 데이터를 불러오는 과정 
xy_train = train_datagen.flow_from_directory(
'../data/brain/train', 
target_size=(150,150), 
batch_size= 160, #y값
class_mode='binary',
save_to_dir='d:/temp/'

)
#Found 160 images belonging to 2 classes.


batch_size=160*0.2
# 추가 데이터 만들기

xy_aug = train_datagen.flow_from_directory(
    '../data/brain/train',
    target_size=(150,150), 
    batch_size=32,
class_mode='binary', 
shuffle=True
,    save_to_dir='d:/temp/'
)

# 데이터 합치기
x_train = np.concatenate((xy_train[0][0],xy_aug[0][0])) 
y_train = np.concatenate((xy_train[0][1],xy_aug[0][1]))

#테스트 데이터 만들기

xy_test = test_datagen.flow_from_directory(
    '../data/brain/train',
    target_size=(150,150), 
    batch_size=120,
class_mode='binary', 
shuffle=True,

)

x_test = xy_test[0][0]
y_test = xy_test[0][1]

print(x_train.shape)

from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

# y_test = y_test.reshape(y_test[0],1)


print(y_train.shape)


from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPool2D,GlobalAveragePooling2D


model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(150, 150, 3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))



model.add(GlobalAveragePooling2D())
model.add(Dense(2, activation='softmax'))


#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)

import time
start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=100, batch_size=100, validation_split=0.2, verbose=1,callbacks=[es])

end_time = time.time() - start_time


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[-1])
print('accuracy', loss[-1])
print("걸린시간 : ", end_time)
print('val_loss:', val_loss)
print('val_acc:', val_acc)
