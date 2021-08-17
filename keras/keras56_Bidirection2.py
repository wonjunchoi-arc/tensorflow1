from tensorflow.keras.preprocessing.text import Tokenizer
import numpy as np

#1. 데이터

docs = ['너무 재밋어요', '참 최고에요', '참 잘 만든 영화예요', 
        '추천하고 싶은 영화입니다', '한 번 더 보고 싶네요', '글세요', 
        '별로에요', '생각보다 지루해요', '연기가 어색해요', '재미없어요', 
        '너무 재미없다', '참 재밋네요', '청순이가 잘 생기긴 했어요']


#1. 긍정, 2. 부정
labels = np.array([1,1,1,1,1,0,0,0,0,0,0,1,1])

token = Tokenizer()
token.fit_on_texts(docs)
print(token.word_index)

'''
{'참': 1, '너무': 2, '잘': 3, '재밋어요': 4, '최고에요': 5, '만든': 6, '영화예요': 7, '
추천하고': 8, '싶은': 9, '영화입니다': 10, '한': 11, '번': 12, '더': 13, '보고': 14, '싶
네요': 15, '글세요': 16, '별로에요': 17, '생각보다': 18, '지루해요': 19, '연기가': 20, '어색해요': 21, '재미없어요': 22, '재미없다': 23, '재밋네요': 24, '청순이가': 25, '생기긴
': 26, '했어요': 27}
'''

x = token.texts_to_sequences(docs)
print(x)

'''
[[2, 4], [1, 5], [1, 3, 6, 7], [8, 9, 10], [11, 12, 13, 14, 15], [16], [17],
 [18, 19], [20, 21], [22], [2, 23], [1, 24], [25, 3, 26, 27]]

데이터의 크기가 모두 다르기 때문에 0을 채워줌으로써 같은 크기로 만들어준다. 
안그러면 모델에 안들어 가자너?
0을 채워줄 땐 맨 앞에 채워준다. 그 이유는 ?

시계열 데이터에서 가중치에 가장 큰 영향을 주는 것은 마지막에 나오는 데이터다

근데 마지막에 0을 채워넣어 버리면 가중치가 전부 0으로 동일해지므로 
맨 앞에 넣는 것이다. !

'''

from tensorflow.keras.preprocessing.sequence import pad_sequences

pad_x = pad_sequences(x, padding='pre', maxlen=5) #0을 앞에넣고 5개짜리에 맞춰서 하겠따
print(pad_x)
print(pad_x.shape) # (13, 5)

'''
너무 긴 문장에 대해 위와 같은 방식으로 진행을 하게 되면 
수없이 많은 0이 들어가기 때문에 적당한 크기에서 잘라야한다.
tip = maxlen으로 자르면 패딩값이 어디에 들어가든 앞의 숫자를 자른다. 

'''

from tensorflow.keras.utils import to_categorical

# x = to_categorical(pad_x)
# print(x)#
# print(x.shape) #(13, 5, 28)
#옥스포드 사전 (13,5,100000) => 6500만개
'''
이처럼 자연어 처리에서 원핫인코더를 쓰면 데이터의 수가 너무나 크게 늘어날 수 있다.
조금 더 쉽게 예를 들어보면 1만개의 단어를 가진 문장을 원핫 인코딩 한다고 치면 각 인코더마다 쓸모없는 9999개의 0이 들어간다. 
그렇다면 어떻게 해야할까?

'''
word_size = len(token.word_index)
print(word_size) #27

print(np.unique(pad_x))

#[ 0  1  2  3  4  5  6  7  8  9 10 11 12 13 14 15 16 17 18 19 20 21 22 23
# 24 25 26 27]


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Embedding, LSTM,Bidirectional



# model = Sequential()
#                 # 단어사전의 개수 , #아웃풋 노드의 개수  #문장의 길이 , 단어 수  
# model.add(Embedding(input_dim=27, output_dim=10, input_length=5)) # (None, 5, 11) 
# #인풋은 (13,5)인데  다음 파라미터에 반영 되지 않는 이유는 
# #Embedding은 벡터화만 시키기 때문에 들어오는 input이 중요하지 않다. 
#  단 단어사전의 수 보다는 많아야 한다. 

# model.add(LSTM(32))
# model.add(Dense(1, activation='sigmoid'))

# model.summary()

'''
_________________________________________________________________
Layer (type)                 Output Shape              Param #
=================================================================
embedding (Embedding)        (None, 5 10,)             270
_________________________________________________________________
lstm (LSTM)                  (None, 32)                5504
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
'''



model = Sequential()
                # 단어사전의 개수 , #아웃풋 노드의 개수  #문장의 길이 , 단어 수  
model.add(Embedding(28, 77)) # (None, 5, 11) 
#인풋은 (13,5)인데  다음 파라미터에 반영 되지 않는 이유는 
#Embedding은 벡터화만 시키기 때문에 들어오는 input이 중요하지 않다. 
# input length를 따로 입력하지 않으면 자동으로 가장 긴 값이 적용된다. 


# model.add(LSTM(32))
model.add(Bidirectional(LSTM(32)))
model.add(Dense(1, activation='sigmoid'))

model.summary()

'''
=================================================================
embedding (Embedding)        (None, None, 77)          2156
_________________________________________________________________
lstm (LSTM)                  (None, 32)                14080
_________________________________________________________________
dense (Dense)                (None, 1)                 33
=================================================================
'''
#3. 컴파일, 훈련
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
model.fit(pad_x, labels, epochs=100, batch_size=1)

#4. 평가
acc = model.evaluate(pad_x, labels)[1]
print('acc:', acc)