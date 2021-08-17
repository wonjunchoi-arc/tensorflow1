import imp
import numpy as np
from sklearn.model_selection import train_test_split
x1 = np.array([range(100), range(301,401), range(1, 101)])
# x2 = np.array([range(101, 201), range(411,511), range(100,200)])
x3 = np.transpose(x1)
# x4 = np.transpose(x2)
# y1= np.array([range(1001, 1101)]) 
# y1=
y1 = np.array(range(1001,1101))
y2 = np.array(range(1901,2001))

print(x1.shape, y1.shape) #(100, 3) (100, 3) (100,) (100,)

x3_train, x3_test, y1_train, y1_test, y2_train, y2_test = train_test_split(
    x3, y1,y2, train_size=0.90,
)


print(x3_train.shape,
y1_train.shape, y2_train.shape, 
    y2_test.shape)

#2. 모델 구성
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델 1

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(10, activation='relu',name='dense2')(dense1)
dense3 = Dense(10, activation='relu',name='dense3')(dense2)
dense4 = Dense(10, activation='relu',name='dense4')(dense3)
output1 = Dense(5,name='output1')(dense4)

#2-2. 모델 2

# input2 = Input(shape=(3,))
# dense11 = Dense(10, activation='relu',name='dense11')(input2)
# dense12 = Dense(10, activation='relu',name='dense12')(dense11)
# dense13 = Dense(10, activation='relu',name='dense13')(dense12)
# dense14 = Dense(10, activation='relu',name='dense14')(dense13)
# dense15 = Dense(10, activation='relu',name='dense15')(dense14)
# output2 = Dense(7,name='output2')(dense15)

from tensorflow.keras.layers import concatenate, Concatenate
# 소문자는 메소드, 대문자는 클래스를 불러오는 것이다. 그러나 둘중 하나는 예전에 쓰던 것일 수 도 있기에 새로운 기능을 사용하지 못할 수 도 있다. 

# merge1 = concatenate([output1, output2])
# merge1 = Concatenate(axis=1)([output1, output2])

merge2 = Dense(10)(output1)
merge3 = Dense(5, activation='relu')(merge2)


output21 = Dense(7)(merge3)
last_output1 = Dense(1)(output21)


output22 = Dense(8)(merge3)
last_output2 = Dense(1)(output22)




model = Model(inputs=input1, outputs=[last_output1, last_output2])
#두개 이상은 리스트이다. 

model.summary()

#3. 컴파일, 훈련
model.compile(loss = 'mse', optimizer='adam', metrics=['mae'] )
model.fit(x3,[y1, y2], epochs=100, batch_size=8, verbose=1  )#x가 두개면 그냥 리스트로 주면 되는 구나!!

#4. 평가 예측
loss = model.evaluate(x3_test,[y1_test, y2_test])
# print('loss' ,loss[0])
print('loss' ,loss[0])
print('matrics[mae]' ,loss[1])


# y_predict = model.predict([x3_test, x4_test])