import numpy as np
#1. 데이터
x = np.array([1,2,3,4,5,6,7,8,9,10])
y = np.array([1,2,3,4,5,6,7,8,9,10])

#2. 모델
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

model = Sequential()
model.add(Dense(100, input_dim=1))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1000))
model.add(Dense(1))

#3. 컴파일, 훈련
from tensorflow.keras.optimizers import Adam,Adagrad,Adamax,Adadelta
from tensorflow.keras.optimizers import SGD, RMSprop, Nadam

# optimizer = Adam(lr=0.001) /default =0.01
# just adam :loss: 1.4210854715202004e-13 결과물: [[10.999999]]
# lr 0.1 : loss: 0.001636727130971849 결과물: [[10.926974]]
# lr 0.01 :loss: 2.688507905190818e-08 결과물: [[11.0003]]
# lor 0.001 loss: 0.0 결과물: [[11.]]

optimizer = Adagrad(lr=0.01) #Defaults to 0.001.
# just :loss: 1.2458856872399338e-05 결과물: [[11.002022]]
# lr 0.1 : loss: 3.87214204238262e-05 결과물: [[10.997539]]
# lr 0.01 : loss: 184.72964477539062 결과물: [[0.17814872]]

optimizer = Adamax(lr=0.01) #Defaults to 0.001.
# just : loss: 2.728413892327808e-05 결과물: [[11.009374]]
# 0.1 : loss: 202648.21875 결과물: [[488.10425]]
# 0.01 loss: 5.890460030855138e-08 결과물: [[11.000213]]

optimizer = Adadelta(lr=0.01) #default : 0.001
#just : loss: 6.574883445864543e-05 결과물: [[10.9861145]]
#lr 0.1 : loss: 0.600075900554657 결과물: [[11.913956]]
#lr 0.01 : loss: 0.08716388046741486 결과물: [[11.352859]]

optimizer = SGD(lr=0.1)
#just : loss: nan 결과물: [[nan]]

optimizer = RMSprop(lr=0.01) #default : 0.001
#just :loss: 0.18878138065338135 결과물: [[11.935809]]

optimizer = Adamax(lr=0.01)

model.compile(loss='mse', optimizer='RMSprop', metrics=['mse'])
model.fit(x,y, epochs=100,batch_size=1 )

#4. 평가 예측
loss, mse = model.evaluate(x,y,batch_size=1)
y_pred =model.predict([11])

print('loss:', loss, '결과물:',y_pred)
