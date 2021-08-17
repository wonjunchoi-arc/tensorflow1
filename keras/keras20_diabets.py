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
    x,y, train_size=0.90,
)

from sklearn.preprocessing import MinMaxScaler, StandardScaler
# scaler = MinMaxScaler()
scaler = StandardScaler()
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
model.fit(x_train, y_train, epochs=100, batch_size=10,verbose=3)



# #4. 평가, 예측

loss = model.evaluate(x_test, y_test)
print('loss' ,loss)

y_predict = model.predict(x_test)

'''
1. scaler x , validation x
loss 3187.416015625
r2 0.3231250571986696
loss 2792.3330078125
r2 0.3744709073749548

2. scaler x
loss 3034.58935546875
r2 0.4359843969425009

loss 4244.03662109375
r2 0.24465147333333837

3. minmax ,,validation o

loss 3679.414306640625
r2 0.4994526913796794

loss 2850.285400390625
r2 0.522830359130503

4. minmax ,,validation x
loss 3571.372802734375
r2 0.5471933182202292

loss 2861.36181640625
r2 0.6076071454259189

5 Standardization ,, validation o
loss 4891.14794921875
r2 0.13273420603432196

loss 7424.02099609375
r2 0.04397191074675366

6. Standardization ,, validation o
loss 4849.90576171875
r2 0.1173964682783668

loss 4094.07080078125
r2 0.3738214866010868

'''


from sklearn.metrics import r2_score
r2 = r2_score(y_test, y_predict)
print('r2', r2)



plt.scatter(x[:,7],y)
plt.show()
