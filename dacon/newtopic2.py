from re import X
import numpy as np
import pandas as pd
from pandas.core.arrays import categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.convolutional import Conv1D
import re
from icecream import ic 

import warnings 
warnings.filterwarnings(action='ignore')




train = pd.read_csv('../data/train_data.csv',index_col=None,  
                         header=0,)#usecols=원하는 컬럼 가져오기
test = pd.read_csv('../data/test_data.csv',index_col=None,
                         header=0)

submission = pd.read_csv('../data/sample_submission.csv', header=0)

def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean
train["cleaned_title"] = train["title"].apply(lambda x : clean_text(x))
test["cleaned_title"]  = test["title"].apply(lambda x : clean_text(x))



x_train = train["cleaned_title"].tolist()
x_test =test["cleaned_title"].tolist()
Y_train = np.array(train.topic_idx)


# print(test.shape)
# print(y.shape)

# Tokenizer
from keras.preprocessing.text import Tokenizer
vocab_size = 10000  

tokenizer = Tokenizer()  
  # Tokenizer 는 데이터에 출현하는 모든 단어의 개수를 세고 빈도 수로 정렬해서 
  # num_words 에 지정된 만큼만 숫자로 반환하고, 나머지는 0 으로 반환합니다                 
tokenizer.fit_on_texts(x_train) # Tokenizer 에 데이터 실제로 입력

print(tokenizer.word_index)

sequences_train = tokenizer.texts_to_sequences(x_train)    # 문장 내 모든 단어를 시퀀스 번호로 변환
sequences_test = tokenizer.texts_to_sequences(x_test)      # 문장 내 모든 단어를 시퀀스 번호로 변환

print(len(sequences_train),)





# # #1. 데이터 프레임 데이터 토큰화

# # from tensorflow.keras.preprocessing.text import Tokenizer

# # token = Tokenizer(num_words=5000)
# # token.fit_on_texts(x)
# # token.fit_on_texts(predict)

# # # print(token.word_index)

# # x =token.texts_to_sequences(x)
# # predict =token.texts_to_sequences(predict)
# # # print(x)

# # #1-1. 테스트셋 분리

# # from sklearn.model_selection import train_test_split
# # x_train, x_test, y_train, y_test = train_test_split(
# #     x,y,train_size=0.7
# # )

# # print("뉴스기사의 최대길이:" ,max(len(i) for i in x_train)) #13
# # print("뉴스기사의 평균길이:", sum(map(len, x_train)) /len(x_train)) # 6.6
# # print("단어의 개수는:" ,max(x_train)) #13



# # # from sklearn.preprocessing import MinMaxScaler, RobustScaler, StandardScaler, QuantileTransformer,PowerTransformer
# # # # scaler =StandardScaler()
# # # scaler =MinMaxScaler()
# # # # scaler =RobustScaler()
# # # # scaler = QuantileTransformer()
# # # # scaler = PowerTransformer()
# # # scaler.fit(x_train)
# # # x_train = scaler.transform(x_train)
# # # x_test = scaler.transform(x_test)
# # # predict = scaler.transform(predict)



# # # 2. 길이 맞춰주기

# # # 2. 길이 맞춰주기

from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(sequences_train, maxlen=12, padding='pre',)
x_test = pad_sequences(sequences_test, maxlen=12, padding='pre')

print(x_train.shape, x_test.shape)
# # print(x_train.shape, x_test.shape)  #(31957, 7) (13697, 7)

# # # 2-1. y 원핫 인코딩

from tensorflow.keras.utils import to_categorical
# # print(np.unique(y)) #[0 1 2 3 4 5 6]

# y_train = to_categorical(Y_train) #(31957, 7)
# # y_test = to_categorical(y_test)

# # print(y_train.shape)

# np.save('./_save/_npy/newstopic_x_train.npy', arr=x_train)
# np.save('./_save/_npy/newstopic_x_test.npy', arr=x_test)
# np.save('./_save/_npy/newstopic_y_train.npy', arr=y_train)




# # #3-1. 모델구성(LSTM)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D,Flatten,Dropout,MaxPool1D,GlobalAveragePooling1D, GRU,Bidirectional



# # 양방향

model = Sequential( )
model.add(Embedding(10000,200, input_length=14))
model.add(Bidirectional(GRU(64 , return_sequences=True)))
model.add(Bidirectional(GRU(64, return_sequences=True)))
model.add(Bidirectional(GRU(32,)))
model.add(Dense(7, activation='softmax'))



# #3. 컴파일, 훈련


from keras.callbacks import EarlyStopping, ModelCheckpoint
es = EarlyStopping(monitor='val_loss', patience=2, mode='min', verbose=1,
                        restore_best_weights=True,min_delta=0.001) #break 지점의 weight를 저장

mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True,
                        filepath='./_save/dacon/news_mcp.hdf5')

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

from sklearn.model_selection import StratifiedKFold
n_fold = 5  
seed = 42
cv = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state=seed)

for i, (i_trn, i_val) in enumerate(cv.split(x_train, Y_train), 1):
    print(f'training model for CV #{i}')
    hist = model.fit(x_train[i_trn],to_categorical(Y_train[i_trn]), validation_data=(x_train[i_val],
    to_categorical(Y_train[i_val])),epochs=10, batch_size=256, verbose=1, 
            callbacks=[es,mcp])#x가 두개면 그냥 리스트로 주면 되는 구나!!





model.save('./_save/dacon/news_model.h5')


# model =load_model('./_save/dacon/news_mcp.hdf5')


# # #4. 평가, 예측
loss=model.evaluate(x_test,)

y_predict = model.predict(x_test)/ n_fold
print('예상값은:',y_predict)



topic = []
for i in range(len(y_predict)):
    topic.append(np.argmax(y_predict[i]))

submission['topic_idx'] = topic

print(submission)
submission.to_csv('./_save/dacon/submission.csv', index=False)

