from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
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
    x,y, train_size=0.8,
)


# print(x_train.shape)

# #2. 모델 구성
model = Sequential()
model.add(Dense(10, input_dim=10, activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(30,activation='relu'))
model.add(Dense(10,activation='relu'))
model.add(Dense(1))


#활성함수 

# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=500, batch_size=40, validation_split=0.4,verbose=3)



# #4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss' ,loss)

y_predict = model.predict(x_test)

'''
loss 3182.093994140625
r2 0.5366601686095063
'''


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2', r2)



# plt.scatter(x[:,1],y)
# plt.show()

#과제 0.62까지 올리기