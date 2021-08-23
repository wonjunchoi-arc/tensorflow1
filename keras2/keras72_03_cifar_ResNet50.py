'''
ResNet50
레이어 층이 깊을 수록 좋다는 인식을 벗어나 
층이 깊어질 수록 입력값이 희미해져가는 것을 방지하기 위해 
입력값이 출력으로 그대로 더해지게 만든 방법을 사용합니다.
https://bskyvision.com/644
그림 참조


'''

# cifar 10 완성
from tensorflow.keras.layers import Dense, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16,VGG19, Xception, ResNet50
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
resNet50 =ResNet50(weights='imagenet',
include_top=False, input_shape=(32,32,3))
 
resNet50.trainable=False
#model.trainable=False 이거쓰면 모델을 훈련을 안시키는 거니깐 아래 모델 전체로 맛이감

# model.summary()

model = Sequential()
model.add(resNet50)
model.add(Flatten())
# model.add(Dense(1024, activation='relu'))
model.add(Dense(400, activation='relu'))
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
model.fit(x_train, y_train, epochs=3, batch_size=100, validation_split=0.3, 
verbose=1,callbacks=[es])

end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss', loss[0])
print('accuracy', loss[1])
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
loss 3.1131629943847656
accuracy 0.32899999618530273
걸린시간 :  31.903987407684326

=========================================================

CIFAR 100
!!!train true ====>FC

loss 2.702850580215454
accuracy 0.41659998893737793
걸린시간 :  238.14766144752502



!!!train true ====> GAP  
loss 2.8949389457702637
accuracy 0.3865000009536743
걸린시간 :  237.08471202850342


!!!train False ===> FC
loss 2.7380387783050537
accuracy 0.3321000039577484
걸린시간 :  33.33426070213318

!!!train False ===> GAP

loss 3.1131629943847656
accuracy 0.32899999618530273
걸린시간 :  31.903987407684326

'''