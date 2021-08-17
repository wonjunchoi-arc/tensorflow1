from re import X
import re
import numpy as np
import pandas as pd
from pandas.core.arrays import categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.convolutional import Conv1D
import kss

import warnings 
warnings.filterwarnings(action='ignore')


Y = pd.read_csv('../data/train_data.csv',index_col=None,  
                         header=0)#usecols=원하는 컬럼 가져오기

train = pd.read_csv('../data/dacon/train.csv',index_col=None,  
                         header=0)#usecols=원하는 컬럼 가져오기
test = pd.read_csv('../data/dacon/test.csv',index_col=None,
                         header=0)

submission = pd.read_csv('../data/sample_submission.csv', header=0)

train=train.iloc[:,0]  #숫자로 자를 땐 iloc 0, =train.iloc[:,0] 시리즈 추출
test=test.iloc[:,0] # test.iloc[:,[0]] dataframe으로 추출
print(type(train))


def clean_text(sent):
    sent_clean = re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean
train["cleaned_title"] = train.apply(lambda x : clean_text(x))
test["cleaned_title"]  = test.apply(lambda x : clean_text(x))



x_train = train["cleaned_title"].tolist()
x_test =test["cleaned_title"].tolist()
Y_train = np.array(Y.topic_idx)


# Tokenizer
from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer( )  
  # Tokenizer 는 데이터에 출현하는 모든 단어의 개수를 세고 빈도 수로 정렬해서 
  # num_words 에 지정된 만큼만 숫자로 반환하고, 나머지는 0 으로 반환합니다                 
tokenizer.fit_on_texts(x_train) # Tokenizer 에 데이터 실제로 입력

print(tokenizer.word_index)

vocab_size = len(tokenizer.word_index) + 1

print(vocab_size)

sequences_train = tokenizer.texts_to_sequences(x_train)    # 문장 내 모든 단어를 시퀀스 번호로 변환
sequences_test = tokenizer.texts_to_sequences(x_test)      # 문장 내 모든 단어를 시퀀스 번호로 변환

# print(len(sequences_train), len(sequences_test))





# # # #1. 데이터 프레임 데이터 토큰화

# # # #1-1. 테스트셋 분리


print("뉴스기사의 최대길이:" ,max(len(i) for i in sequences_train)) #13
print("뉴스기사의 평균길이:", sum(map(len, sequences_train)) /len(sequences_train)) # 6.6
print("단어의 개수는:" ,max(sequences_train)) #13



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

from tensorflow.keras.preprocessing.sequence import pad_sequences

x_train = pad_sequences(sequences_train, maxlen=14, padding='pre',)
x_test = pad_sequences(sequences_test, maxlen=14, padding='pre')

print(x_train.shape, x_test.shape)  #(45654, 14) (9131, 14)
print(x_train.shape, x_test.shape)  #(45654, 14) (9131, 14)

# # # 2-1. y 원핫 인코딩

from tensorflow.keras.utils import to_categorical


# # #3-1. 모델구성(LSTM)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Embedding, Conv1D,Flatten,Dropout,MaxPool1D,GlobalAveragePooling1D, GRU,Bidirectional

# # #3-2. 모델구성(Dense)

# # # model = Sequential()

# # # 3-3 모델구성 (COnv)
# # model =Sequential()
# # model.add(Embedding(2000,200, input_length=14))
# # model.add(Conv1D(256, 2, activation='relu' ))
# # model.add(Dropout(0.4)) # 20%만큼 노드를 랜덤으로 탈락시켜준다. 
# # model.add(Conv1D(128, 2, activation='relu'))
# # model.add(MaxPool1D())

# # model.add(Conv1D(64, 2, activation='relu'))
# # model.add(Dropout(0.4))
# # model.add(Conv1D(32, 1, activation='relu'))
# # model.add(GlobalAveragePooling1D())
# # model.add(Dense(7, activation='softmax'))


# # # model.summary()

# # #3-4 모뎅구성(GRU)
# model = Sequential( )
# model.add(Embedding(vocab_size,50, input_length=7))
# model.add(LSTM(8, activation='relu',return_sequences=True))
# model.add(LSTM(4, activation='relu'))
# # model.add(Flatten())
# # model.add(Dense(256, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(128, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(64, activation='relu'))
# # model.add(Dropout(0.2))
# # model.add(Dense(32, activation='relu'))
# # model.add(Dense(16, activation='relu'))
# model.add(Dense(7, activation='softmax'))

# # # #3-5 DNN
# # # model = Sequential()
# # # model.add(Dense(128, input_dim = 150000, activation = "relu"))
# # # model.add(Dropout(0.8))
# # # model.add(Dense(7, activation = "softmax"))


# # # 양방향

# model = Sequential( )
# model.add(Embedding(vocab_size,200, input_length=14))
# model.add(Bidirectional(GRU(64 , return_sequences=True)))
# model.add(Bidirectional(GRU(64, return_sequences=True)))
# model.add(Bidirectional(GRU(32,)))
# model.add(Dense(7, activation='softmax'))



# # #3. 컴파일, 훈련


# from keras.callbacks import EarlyStopping, ModelCheckpoint
# es = EarlyStopping(monitor='val_loss', patience=2, mode='min', verbose=1,
#                         restore_best_weights=True,min_delta=0.001) #break 지점의 weight를 저장

# mcp = ModelCheckpoint(monitor='val_loss', mode='min', verbose=1, save_best_only=True,
#                         filepath='./_save/dacon/news_mcp.hdf5')

# model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['acc'])

# from sklearn.model_selection import StratifiedKFold
# n_fold = 5  
# seed = 42
# cv = StratifiedKFold(n_splits = n_fold, shuffle=True, random_state=seed)

# test_y = np.zeros((x_test.shape[0], 7))


# for i, (i_trn, i_val) in enumerate(cv.split(x_train, Y_train), 1):
#     print(f'training model for CV #{i}')
#     print(i_trn,i_val)
#     hist = model.fit(x_train[i_trn],to_categorical(Y_train[i_trn]), validation_data=(x_train[i_val],
#     to_categorical(Y_train[i_val])),epochs=10, batch_size=256, verbose=1, 
#             callbacks=[es,mcp])#x가 두개면 그냥 리스트로 주면 되는 구나!!

#     test_y += model.predict(x_test)/ n_fold



# model.save('./_save/dacon/news_model.h5')


model =load_model('./_save/dacon/news_mcp1.hdf5')


# # #4. 평가, 예측
# loss=model.evaluate(x_test,)


y_predict = model.predict(x_test)
print('예상값은:',y_predict)

print(type(y_predict))

topic = []
for i in range(9131):
    topic.append(np.argmax(y_predict[i]))

submission['topic_idx'] = topic

print(submission)
submission.to_csv('./_save/dacon/submission.csv', index=False)


# # import matplotlib.pyplot as plt

# # plt.figure(figsize=(12, 4))

# # plt.subplot(1, 2, 1)
# # plt.title('loss of Bidirectional LSTM (model3) ', fontsize= 15)
# # plt.plot(hist.history['loss'], 'b-', label='loss')
# # plt.plot(hist.history['val_loss'],'r--', label='val_loss')
# # plt.xlabel('Epoch')
# # plt.legend()

# # plt.subplot(1, 2, 2)
# # plt.title('accuracy of Bidirectional LSTM (model3) ', fontsize= 15)
# # plt.plot(hist.history['acc'], 'g-', label='acc')
# # plt.plot(hist.history['val_acc'],'k--', label='val_acc')
# # plt.xlabel('Epoch')
# # plt.legend()
# # plt.show()