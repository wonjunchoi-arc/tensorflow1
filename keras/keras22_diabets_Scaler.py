# 실습 diabets
#1.  loss와 R2로 평가함
#2. MinMax와 Stanard 결과들 표시

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

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, QuantileTransformer, MaxAbsScaler, PowerTransformer
# scaler = MinMaxScaler()
# scaler = StandardScaler()
# scaler = RobustScaler()
# scaler = QuantileTransformer()
# scaler = MaxAbsScaler()
scaler = PowerTransformer()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)



# print(x_train.shape)

# #2. 모델 구성
model = Sequential()
model.add(Dense(128, input_dim=10, activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(32,activation='relu'))
model.add(Dense(1))


# #3. 컴파일, 훈련
model.compile(loss='mse', optimizer='adam')
model.fit(x_train, y_train, epochs=100, batch_size=8,verbose=3)



# #4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss' ,loss)

y_predict = model.predict(x_test)

'''
1. Standard
loss 5788.34912109375
r2 -0.026863795789244582

2. MinMax 
loss 3219.92431640625
r2 0.3976042539059982  여기가 높게 나오는걸 봐서는 데이터가 넓게 골고루 퍼져있지는 않은것으로 보인다. 

3.Robust
loss 6044.29638671875
r2 0.03784352539294633

4. Quantile
loss 3862.392333984375
r2 0.24146390352885094

5. Maxabs
loss 5691.2021484375
r2 -0.20928538029424648

6 Power
loss 4645.39599609375
r2 0.12970579306114272


'''


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2', r2)



plt.scatter(x[:,7],y)
plt.show()
