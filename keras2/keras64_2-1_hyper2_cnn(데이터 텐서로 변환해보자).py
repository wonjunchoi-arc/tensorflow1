# 실습 : CNN으로 변경
#파라미터 변경해보고(노드의 갯수 , activation도 추가)
#epoch[1,2,3]
#learning rate


from os import name
import numpy as np
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout,Input, Conv2D,MaxPooling2D,GlobalAveragePooling2D
from tensorflow.python.keras.backend import dropout, learning_phase
import tensorflow as tf

#1. 데이터
(x_train, y_train),(x_test, y_test)= mnist.load_data()



from tensorflow.keras.utils import to_categorical
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


def single_conv(x, units):
    fc = Conv2D(units,kernel_size=(2,2), activation ='relu',)(x)
    dr = Dropout(0.2)(fc)
    return dr

def Max(x):
    Pool = MaxPooling2D()(x)
    
    return Pool

#2. 모델
def build_model(drop=0.5, optimizer='adam'):
    inputs = Input(shape=(28,28,1), name = 'input')
    x=single_conv(inputs,64)
    x=single_conv(x,32)
    x=single_conv(x,32)
    x=Max(x)
    x=single_conv(x,16)
    x=single_conv(x,16)
    x=GlobalAveragePooling2D()(x)
    outputs = Dense(10, activation='softmax', name='outputs')(x)
    model =Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer = optimizer, metrics=['acc'], loss='categorical_crossentropy')
    return model

def create_hyperparameters():
    batches = [100,200,300,400,500]
    optimizer = ['rmsprop','adam','adadelta']
    dropout = [0.3, 0.4, 0.5]
    # learning_rate =[0.01, 0.001,0.5,0.05]
    return{'batch_size':batches, "optimizer": optimizer,
     "drop":dropout,} #"learning_rate":learning_rate}


hyperparameters= create_hyperparameters()
print(hyperparameters)


# model2 = build_model()
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
model2 = KerasClassifier(build_fn=build_model, verbose=1,)# epochs=2)

'''
RandomizedSearchCV가 텐서플로우 모델을 받아들일수 있을까??
정답은 아니다 
그러므로 텐서플로우 형식을 사이킷런 형식으로 바꿔줘야한다


'''

from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

model = RandomizedSearchCV(model2, hyperparameters, cv=2)

model.fit(x_train,y_train, verbose=1,epochs=3, validation_split=0.2) 
#우리가 쓸 수 있는 지표는 다 쓸수 있다리!!!

print(model.best_params_)
print(model.best_estimator_)
print(model.best_score_)
acc = model.score(x_test,y_test)
print("최종스코어:", acc)

