import numpy as np

x_data = np.load('./_save/_npy/k55_x_data_boston.npy')
y_data = np.load('./_save/_npy/k55_y_data_boston.npy')


print(x_data)
print(y_data)
print(x_data.shape, y_data.shape)


from re import M
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM ,Conv1D,Flatten
import numpy as np
import matplotlib.pyplot as plt
import random

from tensorflow.python.keras.saving.save import load_model
import numpy as np

x = np.load('./_save/_npy/k55_x_data_boston.npy')
y = np.load('./_save/_npy/k55_y_data_boston.npy')




print(np.min(x), np.max(x)) #0.0 711.0


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, 
train_size=0.7, random_state=66)


from sklearn.preprocessing import PowerTransformer
 
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

#이는 test 데이터는 train데이터에 관여하면 안된다는거

# print(np.min(x_scale), np.max(x_scale))



# print(x.shape) #(506, 13)
# print(y.shape) #(506,)
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], 1) 



#2. 모델 구성

model = Sequential()

# model.add(LSTM(64, input_shape=(5,1)))
model.add(Conv1D(64, 2, input_shape=(13,1)))
model.add(Flatten())
model.add(Dense(10, activation='relu'))
model.add(Dense(1))



# 3. 컴파일, 훈련 , metrics=['acc']
from tensorflow.keras.models import load_model, load_w
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience= 100, mode= 'min', verbose=3)
cp = ModelCheckpoint(monitor='val_loss', save_best_only=True, mode='auto'
                    ,filepath='./_save/ModelCheckPoint/keras48_1_MCP.hdf5')

import time
start_time = time.time()

model.compile(loss='mse', optimizer='adam')
hist = model.fit(x_train,y_train, epochs=100,
batch_size=30, validation_split=0.3,verbose=1,callbacks=[es, cp])
end_time = time.time() - start_time

model.save('./_save/ModelCheckPoint/keras48_1_model_save.h5')

# model = load_model('./_save/ModelCheckPoint/keras48_1_MCP.hdf5')
model.save_weights('./_save/ModelCheckPoint/keras48_1_model_save.h5')
# model.load_weights('./_save/ModelCheckPoint/keras48_1_model_save.h5')



#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss)
print("걸린시간 : ", end_time)


y_predict = model.predict(x_test)
print('y_predict', )


from sklearn.metrics import r2_score
r2 = r2_score(y_test,y_predict)
print('r2스코어',r2)


