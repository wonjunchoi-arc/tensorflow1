import imp
from operator import mod
import numpy as np
from sklearn.model_selection import train_test_split
x1 = np.array([range(100), range(301,401), range(1, 101)])
x2 = np.array([range(101, 201), range(411,511), range(100,200)])
x3 = np.transpose(x1)
x4 = np.transpose(x2)
# y1= np.array([range(1001, 1101)]) 
# y1=
y1 = np.array(range(1001,1101))

print(x1.shape, x2.shape, y1.shape) #(100, 3) (100, 3) (100,)

x3_train, x3_test,x4_train, x4_test, y_train, y_test = train_test_split(
    x3,x4, y1, train_size=0.90,
)


print(x3_test)

#2. 모델 구성
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input

#2-1. 모델 1

input1 = Input(shape=(3,))
dense1 = Dense(10, activation='relu', name='dense1')(input1)
dense2 = Dense(10, activation='relu',name='dense2')(dense1)
dense3 = Dense(10, activation='relu',name='dense3')(dense2)
dense4 = Dense(10, activation='relu',name='dense4')(dense3)
output1 = Dense(5,name='output1')(dense4)

#2-2. 모델 2

input2 = Input(shape=(3,))
dense11 = Dense(10, activation='relu',name='dense11')(input2)
dense12 = Dense(10, activation='relu',name='dense12')(dense11)
dense13 = Dense(10, activation='relu',name='dense13')(dense12)
dense14 = Dense(10, activation='relu',name='dense14')(dense13)
dense15 = Dense(10, activation='relu',name='dense15')(dense14)
output2 = Dense(5,name='output2')(dense15)

from tensorflow.keras.layers import concatenate, Concatenate
# 소문자는 메소드, 대문자는 클래스를 불러오는 것이다. 그러나 둘중 하나는 예전에 쓰던 것일 수 도 있기에 새로운 기능을 사용하지 못할 수 도 있다. 

merge1 = concatenate([output1, output2])
merge2 = Dense(10)(merge1)
merge3 = Dense(5, activation='relu')(merge2)

last_output = Dense(1)(merge3)


model = Model(inputs=[input1, input2], outputs=last_output)
#두개 이상은 리스트이다. 

model.summary()

#3. 컴파일, 훈련

from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='loss', patience=3, mode='min', verbose=1,
                        restore_best_weights=False) #break 지점의 weight를 저장
mcp = ModelCheckpoint(monitor='val_loss', mode='auto', verbose=1, save_best_only=True,
                        filepath='./_save/ModelCheckPoint/keras49_mcp.hdf5')


model.compile(loss = 'mse', optimizer='adam', metrics=['mae'] )
model.fit([x3, x4],y1, epochs=100, batch_size=8, verbose=1,validation_split=0.2, 
            callbacks=[es,mcp]  )#x가 두개면 그냥 리스트로 주면 되는 구나!!

model.save('./_save/ModelCheckPoint/keras49_model.h5')


from sklearn.metrics import r2_score
print('===================1. 기본 출력 ================')

#4. 평가 예측
results = model.evaluate([x3_test,x4_test], y_test)
# print('loss' ,loss[0])
print('loss' ,results[0])


y_predict = model.predict([x3_test, x4_test])

r2 = r2_score(y_test,y_predict)
print('r2스코어',r2)

print('===================2. load model ================')
model2 =load_model('./_save/ModelCheckPoint/keras49_model.h5')

results = model.evaluate([x3_test,x4_test], y_test)
# print('loss' ,loss[0])
print('loss' ,results[0])

y_predict = model.predict([x3_test, x4_test])

r2 = r2_score(y_test,y_predict)
print('r2스코어',r2)

print('===================3. ModelCheckPoint ================')
model3 =load_model('./_save/ModelCheckPoint/keras49_mcp.hdf5')

results = model.evaluate([x3_test,x4_test], y_test)
# print('loss' ,loss[0])
print('loss' ,results[0])

y_predict = model.predict([x3_test, x4_test])

r2 = r2_score(y_test,y_predict)
print('r2스코어',r2)

