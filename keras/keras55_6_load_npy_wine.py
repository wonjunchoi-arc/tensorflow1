import numpy as np



from re import X
import numpy as np
import pandas as pd
from pandas.core.arrays import categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Conv1D,MaxPool1D,GlobalAveragePooling1D,Dropout,LSTM
from tensorflow.python.keras.layers.core import Flatten


x = np.load('./_save/_npy/k55_x_data_wine.npy')
y = np.load('./_save/_npy/k55_y_data_wine.npy')



print(np.unique(y))

# from sklearn.preprocessing import OneHotEncoder
# en = OneHotEncoder()
# y = en.fit_transform(y).toarray()

from tensorflow.keras.utils import to_categorical
y = to_categorical(y)

print(y.shape)
print(y[:5])


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.1, random_state=66)


from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer, RobustScaler
# scaler =MinMaxScaler()
# scaler = StandardScaler()
# scaler =QuantileTransformer()
scaler = RobustScaler()
# scaler = QuantileTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

print(x_train.shape)




x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 

#2. 모델 구성
model = Sequential()
# model.add(SimpleRNN(units=10, activation='relu', input_shape=(3,1)))
model.add(Conv1D(64,2, input_shape=(x_train.shape[1],1)))
model.add(Flatten())
model.add(Dense(32, activation='relu'))
model.add(Dense(16, activation='relu'))
model.add(Dense(4, activation='relu'))
# model.add(Dense(8, activation='relu'))
# model.add(Dense(8, activation='relu'))
model.add(Dense(3, activation="softmax"))


#3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 30, mode= 'min', verbose=3)
cp = ModelCheckpoint(monitor='acc', save_best_only=True, mode='auto' ,
                            filepath='./_save/ModelCheckPoint/keras48_5_MCP.hdf5')

import time
start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
hist = model.fit(x_train,y_train, epochs=5,
batch_size=30, validation_split=0.3,verbose=1,callbacks=[es,cp])
end_time = time.time() - start_time

# model.save('./_save/ModelCheckPoint/keras48_5_model.h5')
from tensorflow.keras.models import load_model

model = load_model('./_save/ModelCheckPoint/keras48_5_MCP.hdf5')


#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print('accuracy', loss[1])
print("걸린시간 : ", end_time)

