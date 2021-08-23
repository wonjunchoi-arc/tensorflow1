##모델을 돌리다가 어느때는 되고 어느때는 안될 때 모델의 레이어를 조절해보자!!! 너무 과하거나 부족한것이 문제일 것이다!!



from tensorflow.keras.layers import Dense, Flatten,GlobalAveragePooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16,VGG19
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

vgg19 =VGG19(weights='imagenet',
include_top=False, input_shape=(32,32,3))
 
vgg19.trainable=False
#model.trainable=False 이거쓰면 모델을 훈련을 안시키는 거니깐 아래 모델 전체로 맛이감

# model.summary()

model = Sequential()
model.add(vgg19)
model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dense(402, activation='relu'))
model.add(Dense(200, activation='relu'))
model.add(Dense(100, activation='relu'))
# model.add(Dense(64, activation='relu'))
model.add(Dense(64, activation='relu'))

# model.add(GlobalAveragePooling2D())
model.add(Dense(100, activation='softmax'))

# model.trainable=False

model.summary()

#3. 컴파일 및 훈련
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_acc', patience=5, mode="auto", verbose=1)


import time
start_time = time.time()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc']) 
hist = model.fit(x_train, y_train, epochs=1, batch_size=100, validation_split=0.3, 
verbose=1,callbacks=[es])

end_time = time.time() - start_time

#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('=============================================')
print('loss는',hist.history['loss'][-1:])
print('acc는',hist.history['acc'][-1:])
print('val_loss는',hist.history['val_loss'][-1:])
print('val_acc',hist.history['val_acc'][-1:])
print("걸린시간 : ", end_time)


'''
!!!train true ====>FC
loss 0.8004138469696045
accuracy 0.736299991607666
걸린시간 :  179.6866216659546

!!!train true ====> GAP  
모델의 연산값이 너무 많아 값을 제대로 못받는 경우가 발생



!!!train False ===> FC
loss 1.2606436014175415
accuracy 0.5827999711036682
걸린시간 :  68.77205109596252

!!!train False ===> GAP
loss 1.5419098138809204
accuracy 0.5419999957084656
걸린시간 :  64.4881284236908

=========================================================

CIFAR 100
!!!train true ====>FC

loss 3.710245370864868
accuracy 0.08810000121593475
걸린시간 :  168.88012647628784



!!!train true ====> GAP  
모델의 연산값이 너무 많아 값을 제대로 못받는 경우가 발생



!!!train False ===> FC
loss 3.8390960693359375
accuracy 0.2964000105857849
걸린시간 :  74.54202342033386

!!!train False ===> GAP
loss 5.8238525390625
accuracy 0.26989999413490295
걸린시간 :  67.29795002937317



'''