# Data Loading & Setting Library
import pandas as pd
import numpy as np
import re
import os
from konlpy.tag import Okt

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rc('font', family='Malgun Gothic')


from tensorflow.keras.preprocessing.text import Tokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, f_classif

import pickle

# etc..
import warnings
warnings.simplefilter('ignore')
warnings.filterwarnings('ignore')

def data_loading(path,target_columns):
    train_path = os.path.join(path, 'train.csv')
    test_path = os.path.join(path,'test.csv')

    train = pd.read_csv(train_path)
    train_texts = train[target_columns + ['label']]
    test_texts = pd.read_csv(test_path)[target_columns]
    
    print('DATA LOADING DONE')
    
    return train_texts, test_texts
def clean_text(sent):
    sent_clean=re.sub("[^가-힣ㄱ-ㅎㅏ-ㅣa-zA-Z]", " ", sent) #특수문자 및 기타 제거
    sent_clean = re.sub(' +', ' ', sent_clean) # 다중 공백 제거
    return sent_clean

def data_preprocessing(data,target_columns):
    data = data.fillna('NONE')
    data['요약문_연구목표'] = data.apply(lambda x : x['과제명'] if x['요약문_연구목표'] == 'NONE' else x['요약문_연구목표'], axis=1)
    
    okt = Okt()
    data['요약문_한글키워드'] = data.apply(lambda x : ','.join(okt.nouns(x['과제명'])) if x['요약문_한글키워드'] == 'NONE' else x['요약문_한글키워드'], axis = 1)
    
    data.loc[:,target_columns] = data[target_columns].applymap(lambda x : clean_text(x))
    
    return data

def drop_short_texts(train, target_columns):
    train_index = set(train.index)
    for column in target_columns:
        train_index -= set(train[train[column].str.len() < 10].index)

    train = train.loc[list(train_index)]
    
    print('SHORT TEXTS DROPED')
    
    return train

def visualize_train_length(train, target_columns) :
    plt.figure(figsize = (12,8)) 
    for i, column in enumerate(target_columns):
        text_len = [len(s.split()) for s in train[column]]

        print('='*50)
        print(f'{column} 최소 길이 : {np.min(text_len)}')
        print(f'{column} 25% 길이 : {np.quantile(text_len,0.25)}')
        print(f'{column} 최대 길이 : {np.max(text_len)}')
        print(f'{column} 75% 길이 : {np.quantile(text_len,0.75)}')
        print(f'{column} 평균 길이 : {np.mean(text_len)}')

        plt.subplot(2,3,i+1)
        plt.boxplot(text_len)
        plt.title(column)


        plt.title(column)

        plt.subplot(2,3,i+4)
        plt.hist(text_len, bins=100)
        plt.xlabel('샘플의 길이')
        plt.ylabel('샘플의 수')

    plt.show()


def sampling_data(train, target_columns):
    pj_name_len = 18
    summ_goal_len = 210
    summ_key_len = 18

    max_lens = [pj_name_len, summ_goal_len, summ_key_len]
    total_index = set(train.index)
    for column, max_len in zip(target_columns, max_lens) : 
        temp = train[column].apply(lambda x : len(x.split()) < max_len)
        explained_ratio = temp.values.sum() / train.shape[0]
        print(column)
        print(f'전체 샘플 중 길이가 {max_len}이하인 샘플의 비율 : {round(explained_ratio * 100,2)}%')

        total_index -= set(train[temp == False].index)
    train = train.loc[list(total_index)].reset_index(drop = True)
    
    return train

def oversampling_minor_classes(train, target_columns) : 
    temp = train.copy()
    temp.loc[:,target_columns] = temp.loc[:,target_columns].applymap(lambda x : len(x.split()))
    pj_range = (6,11)
    summ_goal_range = (30,89)
    summ_key_range = (5,9)
    temp = temp.query('label != 0')
    temp = pd.DataFrame(list(train.loc[temp.query('과제명 in @pj_range or 요약문_연구목표 in @summ_goal_range or 요약문_한글키워드 in @summ_key_range').index].values)*1, columns = temp.columns)
    train = pd.concat([train, temp], axis = 0).reset_index(drop = True)
    train.groupby(['label'])['과제명'].agg('count').plot.bar()
    plt.show()
    train.groupby(['label'])['과제명'].agg('count')[1:].plot.bar()
    plt.show()
    
    return train

def get_topk_nums(train, target_columns):
    top_ks = []
    for column in target_columns:
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(train[column])


        threshold = 5 #기준치
        total_cnt = len(tokenizer.word_index) # 단어의 수
        rare_cnt = 0 # 등장 빈도수가 threshold보다 작은 단어의 개수를 카운트
        total_freq = 0 # 훈련 데이터의 전체 단어 빈도수 총 합
        rare_freq = 0 # 등장 빈도수가 threshold보다 작은 단어의 등장 빈도수의 총 합

        # 단어와 빈도수의 쌍(pair)을 key와 value로 받는다.
        for key, value in tokenizer.word_counts.items():
            total_freq = total_freq + value

            # 단어의 등장 빈도수가 threshold보다 작으면
            if(value < threshold):
                rare_cnt = rare_cnt + 1
                rare_freq = rare_freq + value

        print(f'단어 집합(vocabulary)의 크기 : {total_cnt}')
        print(f'등장 빈도가 {threshold - 1}번 이하인 희귀 단어의 수 : {rare_cnt}')
        print(f'단어 집합에서 희귀 단어를 제외시킬 경우의 단어 집합의 크기 : {(total_cnt - rare_cnt)}')
        print(f'단어 집합에서 희귀 단어의 비율: {(rare_cnt / total_cnt)*100}')
        print(f'전체 등장 빈도에서 희귀 단어 등장 빈도 비율: {(rare_freq / total_freq)*100}')
        print('='*50)
        
        top_ks.append(total_cnt - rare_cnt)
    return top_ks

def ngram_vectorize(train_data, label, test_data, top_k) : 
    kwargs = {
    
        'ngram_range' : (1,2),
        'dtype' : 'int32',
        'strip_accents' : False,
        'lowercase' : False,
        'decode_error' : 'replace',
        'analyzer': 'char',
        'min_df' : 2,
        
            }
    vectorizer = TfidfVectorizer(**kwargs)

    x_train = vectorizer.fit_transform(train_data)
    x_test = vectorizer.transform(test_data)

    selector = SelectKBest(f_classif, k=min(80000,top_k))
    selector.fit(x_train, label.values)
    x_train = selector.transform(x_train).astype('float32')
    x_test = selector.transform(x_test).astype('float32')
    
    return x_train, x_test

def vectorize_data(train, test, top_ks, target_columns):
    train_inputs = []
    test_inputs = []
    for top_k, column in zip(top_ks, target_columns):
        train_input, test_input = ngram_vectorize(train[column], train['label'], test[column], min(80000,top_k))
        train_inputs.append(train_input)
        test_inputs.append(test_input)
        
    return train_inputs, test_inputs

def data_loading_and_setting_main():
    
    # kwargs
    path = '../data/dacon2/open/'
    target_columns = ['과제명','요약문_연구목표','요약문_한글키워드']

    ########################################################################
    train_texts, test_texts = data_loading(path, target_columns)
    
    ########################################################################
    train_texts = data_preprocessing(train_texts,target_columns)
    print('TRAIN NA DATA NUM : ', train_texts.isna().sum().sum())

    test_texts = data_preprocessing(test_texts,target_columns)
    print('TEST NA DATA NUM : ', train_texts.isna().sum().sum())
    
    ########################################################################
    train_texts = drop_short_texts(train_texts, target_columns)
    
    ########################################################################
    visualize_train_length(train_texts, target_columns)
    
    ########################################################################
    train_texts = sampling_data(train_texts, target_columns)
    visualize_train_length(train_texts, target_columns)
    
    ########################################################################
    train_texts = oversampling_minor_classes(train_texts, target_columns)

    ########################################################################
    top_ks = get_topk_nums(train_texts, target_columns)
    
    ########################################################################
    train_inputs, test_inputs = vectorize_data(train_texts, test_texts, top_ks, target_columns)
    
    print('=========================================== !!!PREPARING DATA DONE!!! ===========================================')
    return train_inputs, test_inputs, train_texts['label']

train_inputs, test_inputs, labels = data_loading_and_setting_main()

with open('./inputs.pkl', 'wb') as f:
     pickle.dump((train_inputs, test_inputs, labels), f)



