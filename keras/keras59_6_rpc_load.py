from sys import prefix
import numpy as np

x= np.load('./_save/_npy/image/k59_rps_x.npy')
y = np.load('./_save/_npy/image/k59_rps_y.npy')
# predict=np.load('./_save/_npy/image/k59_wonjun_y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7,shuffle=True,random_state=9) 

print(x_train.shape, x_test.shape) #(1176, 150, 150, 3) (504, 150, 150, 3)
print(y_train.shape, y_test.shape) # (588, 3) (252, 3)

x_train = x_train.reshape(1176,150*150*3)
x_test = x_test.reshape(504,150*150*3)
# predict = predict.reshape(1,150*150*3)


from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler =MinMaxScaler()
# scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)
# predict = scaler.transform(predict)


x_train = x_train.reshape(1176,150,150,3)
x_test = x_test.reshape(504,150,150,3)
# predict = predict.reshape(1,150,150,3)


# #2. 모델
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPool2D,GlobalAveragePooling2D

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=(150, 150, 3)))
model.add(Conv2D(8, (3,3)))
model.add(MaxPool2D(2,2))

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
hist = model.fit(x_train,y_train, epochs=500, batch_size=50
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


# #4.
loss =model.evaluate(x_test,y_test)
print('loss', loss[1])
print('val_loss', val_loss[-1])
print('val_acc', val_acc[-1])

# y_predict = model.predict(predict)
# y_predict = 1 - y_predict
# print('나는 ${y_predict}%확률로 남자입니다.',)



