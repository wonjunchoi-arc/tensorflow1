from tensorflow.keras.datasets import reuters
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = reuters.load_data(num_words=10000,
 test_split=0.2)

print(x_train[0], type(x_train[0]))
print(y_train[0])

print(len(x_train[0]), len(x_train[1]))

print(x_train.shape, x_test.shape) # (8982,) (2246,)
print(y_train.shape, y_test.shape) #(8982,) (2246,)


print(type(x_train)) #<class 'numpy.ndarray'> array안에 list로 들어가있다. 


print("뉴스기사의 최대길이:" ,max(len(i) for i in x_train)) #2376
#print("뉴스기사의 최대길이:" ,max(len(x_train))) 이건 안되!!!

print(sum(map(len, x_train)))# map함수를 써서 x_train의 길이를 순서별로 더해준다. 그리고 아래에서 전체 숫자로 나누어 주는것 그럼 평균 길이 나오겠지
print("뉴스기사의 평균길이:", sum(map(len, x_train)) /len(x_train)) # 145.5


# plt.hist([len(s) for s in x_train], bins=50)
# plt.show()

# 전처리

from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

x_train = pad_sequences(x_train, maxlen=100, padding='pre')
x_test = pad_sequences(x_test, maxlen=100, padding='pre')
# x_train = pad_sequences(x_train, maxlen=100, padding='pre')
# x_train = pad_sequences(x_train, maxlen=100, padding='pre')

print(x_train.shape, x_test.shape) #(8982, 100) (2246, 100)
print(type(x_train),type(x_train[0])) #<class 'numpy.ndarray'> <class 'numpy.ndarray'>

print(x_train[0])

#y확인
print(np.unique(y_train))

y_train = to_categorical(y_train) #(8982, 46)
y_test = to_categorical(y_test)

print(y_train.shape)


#2. 모델구성

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding

model = Sequential( )
model.add(Embedding(10000,20, input_length=100))
model.add(LSTM(8))
model.add(Dense(46, activation='softmax'))


#3. 컴파일, 훈련
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(x_train, y_train, epochs=100, batch_size=300)

#4. 평가
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])