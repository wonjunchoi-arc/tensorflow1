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
from tensorflow.keras.applications import VGG16,VGG19, Xception
from tensorflow.keras.datasets import cifar10, cifar100
from tensorflow.python.keras.backend import softmax
import tensorflow as tf


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
x_train = tf.image.resize(x_train,(71,71))
x_test = tf.image.resize(x_test,(71,71))


xception =Xception(weights='imagenet',
include_top=False, input_shape=(71,71,3))
 
xception.trainable=False
#model.trainable=False 이거쓰면 모델을 훈련을 안시키는 거니깐 아래 모델 전체로 맛이감

# model.summary()

model = Sequential()
model.add(xception)
# model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
# model.add(Dense(402, activation='relu'))


model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))

# model.trainable=False

model.summary()

#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=10, mode="auto", verbose=1)


import time
start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=3, batch_size=30, validation_split=0.3, 
verbose=1,callbacks=[es])

end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print('accuracy', loss[1])
print('val_loss는',hist.history['val_loss'][-1:])
print('val_acc',hist.history['val_acc'][-1:])
print("걸린시간 : ", end_time)


'''
!!!train true ====>FC
loss 0.5537300705909729
accuracy 0.8640000224113464
걸린시간 :  831.5912144184113

!!!train true ====> GAP  
loss는 [0.6726522445678711, 0.36448660492897034, 0.2703045904636383]
acc는 [0.7781714200973511, 0.880142867565155, 0.9100000262260437]
val_loss는 [0.6263164281845093, 0.5145093202590942, 0.5226519703865051]
val_acc [0.8023999929428101, 0.8267999887466431, 0.8429333567619324]
걸린시간 :  247.4694049358368
PS D:\study> 



!!!train False ===> FC
loss 2.3026139736175537
accuracy 0.10000000149011612
val_loss는 [2.302699327468872]
val_acc [0.09833333641290665]
걸린시간 :  102.34308075904846

!!!train False ===> GAP
loss 3.29491925239563
accuracy 0.32359999418258667
val_loss는 [3.3184471130371094]
val_acc [0.32519999146461487]
걸린시간 :  98.15999269485474

=========================================================

CIFAR 100
!!!train true ====>FC

loss 2.1943178176879883
accuracy 0.43540000915527344
val_loss는 [2.2116358280181885]
val_acc [0.430333346


!!!train true ====> GAP  
loss 1.5812097787857056
accuracy 0.5852000117301941
val_loss는 [1.6143648624420166]
val_acc [0.5750666856765747]
걸린시간 :  253.12861275672913


!!!train False ===> FC
loss 3.8390960693359375
accuracy 0.2964000105857849
걸린시간 :  74.54202342033386

!!!train False ===> GAP

accuracy 0.09570000320672989
val_loss는 [13.215529441833496]
val_acc [0.09273333102464676]
걸린시간 :  96.67868065834045



'''