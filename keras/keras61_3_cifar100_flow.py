# 훈련데이터를 10만개로 증폭
#완료후 기존 모델과 비교
#save_dir도 temp에 넣을 것

from tensorflow.keras.datasets import cifar10
import numpy as np

(x_train, y_train),(x_test,y_test) =cifar10.load_data()

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

print(x_train.shape, x_test.shape)



#1. ImageDataGenerator를 정의
#2. 파일에서 땡겨 올려면 -> flow_from_directory() // x,y가 튜플형태로 뭉쳐있음
#3. 데이터에서 땡겨 올려면  ->flow()              // x,y 가 나눠있어

agument_size=40000

randidx = np.random.randint(x_train.shape[0], size =agument_size)
print(x_train.shape[0])
print(randidx)
print(randidx.shape)

x_agmented = x_train[randidx].copy()
y_agmented = y_train[randidx].copy() # 동일한 메모리 안쓸려고 넣는거



print(x_agmented.shape)

x_agmented = x_agmented.reshape(x_agmented.shape[0],32,32,3)

x_train = x_train.reshape(x_train.shape[0],32,32,3)
x_tset = x_test.reshape(x_test.shape[0],32,32,3)



x_agmented =train_datagen.flow( #flow는 4차원을 받아들이고 싶어!! 근데 너는 3차원을 넣었어!!
    x_agmented, y_agmented,
    batch_size=agument_size,shuffle=False,
    # save_to_dir='d:/temp/'
).next()[0] #이렇게 하면 x값만 빠지겠지?

x_train = np.concatenate((x_train,x_agmented)) 
y_train = np.concatenate((y_train,y_agmented))

print(x_train.shape) #(100000, 28, 28, 1)
print(x_agmented[0][0].shape) #(28, 1)
#이렇게 4번을 프린트하면 위의 



from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPool2D,GlobalAveragePooling2D


model = Sequential()
model.add(Conv2D(64, (3,3), activation='relu', input_shape=(32, 32, 3)))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))

model.add(Conv2D(64, (3,3), activation='relu'))
model.add(Conv2D(64, (3,3), activation='relu'))
model.add(MaxPool2D(2,2))



model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))


#3. 컴파일 훈련
from tensorflow.keras.callbacks import EarlyStopping
es= EarlyStopping(monitor='val_loss', patience=100, mode='min', verbose=1)

import time
start_time = time.time()

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['acc'])
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
print('accuracy', loss[-11])
print("걸린시간 : ", end_time)
print('val_loss:', val_loss)
print('val_acc:', val_acc)

#800/800 [==============================] - 6s 7ms/step - loss: 0.0965 - acc: 0.9679 - val_loss: 1.6071 - val_acc: 0.6877