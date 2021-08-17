from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Input
import numpy as np
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


#1. 데이터

datasets = load_diabetes()
x = datasets.data
y = datasets.target

print(x.shape , y.shape)

# print(x[:,1])


# print(datasets.feature_names)
# #['age', 'sex', 'bmi', 'bp', 's1', 's2', 's3', 's4', 's5', 's6']        
# print(datasets.DESCR)
# print(y[:30])
# print(np.min(y), np.max(y))

x_train, x_test, y_train, y_test =train_test_split(
    x,y, train_size=0.90,
)


# print(x_train.shape)

# #2. 모델 구성
# model = Sequential()
# model.add(Dense(10, input_dim=10, activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(10,activation='relu'))
# model.add(Dense(1))

input1 = Input(shape=(10,))
dense1 = Dense(10, activation='relu')(input1)
dense2 = Dense(10,activation='relu')(dense1)
dense3 = Dense(10,activation='relu')(dense2)
dense4 = Dense(10,activation='relu')(dense3)
dense5 = Dense(10,activation='relu')(dense4)
dense6 = Dense(10,activation='relu')(dense5)
dense7 = Dense(10,activation='relu')(dense6)
output1 = Dense(1)(dense7)


model = Model(inputs = input1 , outputs = output1)

# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=10, validation_split=0.3
, verbose=0)



# #4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss' ,loss)

y_predict = model.predict(x_test)

'''
2021-07-12 17:59:14.284994: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublas64_11.dll
2021-07-12 17:59:14.635797: I tensorflow/stream_executor/platform/default/dso_loader.cc:49] Successfully opened dynamic library cublasLt64_11.dll
2/2 [==============================] - 0s 3ms/step - loss: 1764.0861
loss 1764.0860595703125
r2 0.7220125911640178
PS D:\study> 
'''


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2', r2)



# plt.scatter(x[:,1],y)
# plt.show()

#과제 0.62까지 올리기