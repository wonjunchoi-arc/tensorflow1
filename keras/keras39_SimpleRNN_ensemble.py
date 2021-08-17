import numpy as np
from numpy import array
from tensorflow.keras.models import Sequential,Model
from tensorflow.keras.layers import Dense, SimpleRNN, LSTM, GRU,Input


#1. 데이터
x1 = np.array([[1,2,3],[2,3,4],[3,4,5],[4,5,6],
[5,6,7],[6,7,8],[7,8,9],[8,9,10],
[9,10,11],[10,11,12],
[20,30,40],[30,40,50,],[40,50,60,]])


x2 = np.array([[10,20,30],[20,30,40],[30,40,50],[40,50,60],
[50,60,70],[60,70,80],[70,80,90],[80,90,100],
[90,100,110],[100,110,120],
[2,3,4],[3,4,5,],[4,5,6,]])
y = np.array([4,5,6,7,8,9,10,11,12,13,50,60,70])

x1_predict = array([55,65,75])
x2_predict = array([65,75,85])






x1= x1.reshape(x1.shape[0], x1.shape[1], 1)     #(batch_size, timesteps, features)
x2= x2.reshape(x2.shape[0], x2.shape[1], 1)




#2-1. 모델 1

input1 = Input(shape=(3,1))
dense1 = SimpleRNN(units=64, activation='relu')(input1)
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(10, activation='relu',name='dense2')(dense1)
dense3 = Dense(10, activation='relu',name='dense3')(dense2)
dense4 = Dense(10, activation='relu',name='dense4')(dense3)
output1 = Dense(1,name='output1')(dense4)

#2-2. 모델 2

input2 = Input(shape=(3,1))
dense1 = SimpleRNN(units=64, activation='relu')(input1)
dense11 = Dense(10, activation='relu',name='dense11')(input2)
dense12 = Dense(10, activation='relu',name='dense12')(dense11)
dense13 = Dense(10, activation='relu',name='dense13')(dense12)
dense14 = Dense(10, activation='relu',name='dense14')(dense13)
dense15 = Dense(10, activation='relu',name='dense15')(dense14)
output2 = Dense(1,name='output2')(dense15)



from tensorflow.keras.layers import concatenate, Concatenate
# 소문자는 메소드, 대문자는 클래스를 불러오는 것이다. 그러나 둘중 하나는 예전에 쓰던 것일 수 도 있기에 새로운 기능을 사용하지 못할 수 도 있다. 

merge1 = concatenate([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)
last_output = Dense(1)(merge3)


model = Model(inputs=[input1, input2], outputs=last_output)

model.summary()

#3. 컴파일, 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', patience= 100, mode= 'min', verbose=3)

model.compile(loss='mse', optimizer='adam', metrics=['mse'])
hist = model.fit([x1,x2],y, epochs=300, batch_size=1,callbacks=[es])


#4. 평가, 예측
x1_input = x1_predict.reshape(1,3,1)
x2_input = x2_predict.reshape(1,3,1)

results = model.predict([x1_input,x2_input])
print(results)

#5 그리기

# import matplotlib.pyplot as plt
# plt.figure(figsize=(9,5))

# plt.plot(hist.history["loss"])
# plt.grid()
# plt.title('loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['loss'])
# plt.show()


# 결과값이 80 근접하게

'''
[[[57.106964]
  [64.38043 ]
  [71.65389 ]]]
'''


'''
480
output*output*4 +(input+bias*output*4) 
1.0068543
'''