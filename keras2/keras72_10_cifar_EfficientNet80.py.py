#cifar 10 cifar 100 으로 모델 만들것
# trainable = True, False
# Fc로 만들것, GAP로 만들것 비교

#결과 출력
# 1. cifar 10
# trainable = True, FC: loss ? , acc ?
# trainable = True, Gap: loss ? , acc ?
# trainable = False, FC: loss ? , acc ?
# trainable = False, Gap: loss ? , acc ?

# 2. cifar 100
# trainable = True, FC: loss ? , acc ?
# trainable = True, Gap: loss ? , acc ?
# trainable = False, FC: loss ? , acc ?
# trainable = False, Gap: loss ? , acc ?

# cifar 10 완성
from tensorflow.keras.layers import Dense, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.python.keras.backend import softmax


#1. 데이터 구성
(x_train, y_train), (x_test, y_test)= cifar100.load_data()

print(x_train.shape, y_train.shape) #(50000, 32, 32, 3) (50000, 1)
print(x_test.shape, y_test.shape)#  (10000, 32, 32, 3) (10000, 1)


from sklearn.preprocessing import OneHotEncoder
one = OneHotEncoder()
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)
one.fit(y_train)
y_train = one.transform(y_train).toarray() # (50000, 100)
y_test = one.transform(y_test).toarray() # (10000, 100)

print(y_train.shape)

#2. 모델 구성
efficientNetB0 =EfficientNetB0(weights='imagenet',
include_top=False, input_shape=(32,32,3))
 
efficientNetB0.trainable=True
#model.trainable=False 이거쓰면 모델을 훈련을 안시키는 거니깐 아래 모델 전체로 맛이감

# model.summary()

model = Sequential()
model.add(efficientNetB0)
model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))

# model.trainable=False

model.summary()

#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode="auto", verbose=1)


import time
start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
model.fit(x_train, y_train, epochs=100, batch_size=100, validation_split=0.3, 
verbose=1,callbacks=[es])

end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print('accuracy', loss[1])
print("걸린시간 : ", end_time)


'''
!!!train true
loss 0.9063402414321899
accuracy 0.7184000015258789
걸린시간 :  119.85317492485046


!!!train False 훈련을 하지 않고 원래 저장된모델을 쓰겠다.
loss 1.3785994052886963
accuracy 0.5837000012397766


-------------------G A P -------------
loss 2.3025918006896973
accuracy 0.10000000149011612
걸린시간 :  50.49440550804138


!!!train False 훈련을 하지 않고 원래 저장된모델을 쓰겠다.
=============================================
loss 1.4440944194793701
accuracy 0.5530999898910522
걸린시간 :  46.06049036979675

'''