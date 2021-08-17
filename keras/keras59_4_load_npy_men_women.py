import numpy as np

x_train = np.load('./_save/_npy/image/k59_3_train_x.npy')
y_train = np.load('./_save/_npy/image/k59_3_train_y.npy')
x_test = np.load('./_save/_npy/image/k59_3_test_x.npy')
y_test = np.load('./_save/_npy/image/k59_3_test_y.npy')




#2.모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

model = Sequential()
model.add(Conv2D(32, (2,2), input_shape=(150,150,3)))
model.add(Flatten())
model.add(Dense(1, activation='sigmoid'))


#3. 컴파일, 훈련
model.compile(loss = 'binary_crossentropy',optimizer='adam',)

# model.fit(x_train, y_train)
hist = model.fit(x_train,y_train, #xy가 뭉텅이로 되어있는 데이터 훈련용
epochs=50, steps_per_epoch=32, #160/5 =32 한 에포당 베치사이즈로 나눠서 돌아가는 숫자 명시
validation_data=([x_train,y_train]),
validation_steps=4
) 

acc = hist.history['acc']
val_acc= hist.history['val_acc']
loss = hist.history['loss']
val_loss = hist.history['val_loss']

#위에거로 시각화 할 것

print('acc:', acc[-1])
print('val_acc:', val_acc[-1])
