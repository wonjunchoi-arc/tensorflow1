from re import T
from tensorflow.keras.datasets import imdb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test)  = imdb.load_data(num_words=2000,
)

print(len(x_train), len(x_test))#25000, 25000
print(x_train.shape)



print(x_train.shape, x_test.shape) # 
print(y_train.shape, y_test.shape) #

print(type(x_train)) #<class 'numpy.ndarray'>


print("최대길이 :", max(len(i) for i in x_train)) #2494
print("평균길이", sum(map(len, x_train)) / len(x_train)) # 238.71364
print(max(max(x_train)))




from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical


x_train = pad_sequences(x_train, maxlen=239, padding='pre')
x_test = pad_sequences(x_test, maxlen=239, padding='pre')

print(x_train.shape, x_test.shape) #(25000, 239) (25000, 239)


#y 확인
print(np.unique(y_train))

y_train = to_categorical(y_train)
y_test = to_categorical(y_test)
#카테고리칼 하면 원핫 인코딩 해줘야한다. 
print(y_train.shape)



#2-1. 모델구성(Dense)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D,Flatten

# model = Sequential()
# model.add(Embedding(10000, 20, input_length=239))
# model.add(Flatten())
# model.add(Dense(64, activation='relu'))
# model.add(Dense(32, activation='relu'))
# model.add(Dense(16, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(4, activation='relu'))
# model.add(Dense(2, activation='softmax'))


#2-2. 모델구성(LSTM)

model = Sequential()
model.add(Embedding(2000, 20, input_length=239))
model.add(LSTM(16, activation='relu'))
model.add(Dense(64, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(8, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dense(2, activation='softmax'))

#3. 컴파일, 훈련
from keras.callbacks import EarlyStopping, ModelCheckpoint
es =EarlyStopping(monitor='val_loss', patience=30, mode='min', verbose=1,
                                restore_best_weights=True)

mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                                            filepath='./_save/keras/imdb_MCP.hdf5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train, y_train, epochs=10, batch_size=300,validation_split=0.25
                                    , callbacks=[es,mcp])

model.save('./_save/keras/imdb_model.h5')
model.save_weights('./_save/keras/imdb_weight.h5')


#4. 평가
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])


import matplotlib.pyplot as plt

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.title('loss of Bidirectional LSTM (model3) ', fontsize= 15)
plt.plot(hist.history['loss'], 'b-', label='loss')
plt.plot(hist.history['val_loss'],'r--', label='val_loss')
plt.xlabel('Epoch')
plt.legend()

plt.subplot(1, 2, 2)
plt.title('accuracy of Bidirectional LSTM (model3) ', fontsize= 15)
plt.plot(hist.history['acc'], 'g-', label='acc')
plt.plot(hist.history['acc'],'k--', label='val_acc')
plt.xlabel('Epoch')
plt.legend()
plt.show()



'''
1. Dense
loss 0.31724026799201965
accuracy 0.8637999892234802






'''



#실습시작!! 확인하시오!
