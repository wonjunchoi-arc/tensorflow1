from re import X
import re
import numpy as np
import pandas as pd
from pandas.core.arrays import categorical
from tensorflow.keras.models import Sequential,load_model
from tensorflow.keras.layers import Dense
from tensorflow.python.keras.layers.convolutional import Conv1D
import kss
from hanspell import spell_checker #네이버 맞춤법 검사기임!!



import warnings 
warnings.filterwarnings(action='ignore')



train = pd.read_csv('../data/train_data.csv',index_col=None,  
                         header=0,usecols=[1, 2])#usecols=원하는 컬럼 가져오기
test = pd.read_csv('../data/test_data.csv',index_col=None,
                         header=0)

submission = pd.read_csv('../data/sample_submission.csv', header=0)


def clean_text(sent):
    sent_clean = re.sub("[^a-zA-Z가-힣ㄱ-ㅎㅏ-ㅣ\\s]", " ", sent)
    return sent_clean
train = train['title'].apply(lambda x : clean_text(x))
test = test['title'].apply(lambda x : clean_text(x))



# train=train.loc[3]
# print(type(train))



# print(train)

train_list = []
for i, line in enumerate(train):
        A= spell_checker.check(line)
        A = A.checked  #NYT 클린턴 측근韓기업 특수관계 조명…공과 사 맞물려종합 ->NYT 클린턴 측근韓기업 특수 관계 조명…공과 사 맞물 려종합
        train_list.append(A.strip())

test_list = []
for i, line in enumerate(test):
        A= spell_checker.check(line)
        A = A.checked
        test_list.append(A.strip())

print('111111111111111111',train)

train = np.array(train_list)
test = np.array(test_list)
print('2222222222222222222222222',train)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

print('3333333333333333333333333',train)


train.to_csv('../data/dacon/train.csv',index=False)
test.to_csv('../data/dacon/test.csv',index =False)
