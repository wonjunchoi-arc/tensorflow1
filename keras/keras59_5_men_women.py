from sys import prefix
import numpy as np

# x1_train =np.load('./_save/_npy/image/k59_men_women_x_train.npy')
# y1_train =np.load('./_save/_npy/image/k59_men_women_y_train.npy')
# x1_test =np.load('./_save/_npy/image/k59_men_women_x_test.npy')
# y1_test =np.load('./_save/_npy/image/k59_men_women_y_test.npy')

x= np.load('./_save/_npy/image/k59_men_women_x.npy')
y = np.load('./_save/_npy/image/k59_men_women_y.npy')
predict=np.load('./_save/_npy/image/k59_wonjun_y.npy')

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, train_size=0.7) 

print(x_train.shape, x_test.shape)
print(y_train.shape, y_test.shape)

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

print(y_train.shape)

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout,MaxPool2D,GlobalAveragePooling2D,MaxPooling2D


model = Sequential()

model.add(Conv2D(filters = 32, kernel_size=(3,3), input_shape =(150,150,3), activation= 'relu'))
model.add(Conv2D(filters = 32, kernel_size=(3,3),  activation= 'relu'))
model.add(MaxPooling2D(2,2))
model.add(GlobalAveragePooling2D())
model.add(Dense(1, activation= 'sigmoid'))

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=15, mode='min', verbose=1,
                        restore_best_weights=True) #break 지점의 weight를 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                        filepath='./_save/men_women/image_mcp.hdf5')


model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=5
,
 validation_split=0.2,
 callbacks=[es])

model.save('./_save/men_women/image_model.h5')


acc = hist.history['acc']
val_acc = hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#4.
loss =model.evaluate(x_test,y_test)
print('acc', acc[-1])
print('loss', loss[0])

y_predict = model.predict(predict)
y_predict = (1-y_predict)*100
print('나는',y_predict,'%확률로 남자입니다.',)



