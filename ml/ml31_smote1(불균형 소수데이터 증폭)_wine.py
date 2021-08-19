'''
SMOTE(synthetic minority oversampling technique)란, 합성 소수 샘플링 기술로 다수 클래스를 샘플링하고 기존 소수 샘플을 보간하여 새로운 소수 인스턴스를 합성해낸다.
일반적인 경우 성공적으로 작동하지만, 소수데이터들 사이를 보간하여 작동하기 때문에 모델링셋의 소수데이터들 사이의 특성만을 반영하고 새로운 사례의 데이터 예측엔 취약할 수 있다.

명령 프롬프트에서 pip install smote해주자!
'''
from imblearn.over_sampling import SMOTE
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_wine
from re import X
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import warnings

from xgboost.sklearn import XGBClassifier
warnings.filterwarnings('ignore')

dataset = load_wine()
x = dataset.data
y = dataset.target
print(x.shape, y.shape)

print(pd.Series(y).value_counts())
#1    71
#0    59
#2    48

x_new = x[:-40]
y_new = y[:-40]
print(x_new.shape, y_new.shape)
print(pd.Series(y_new).value_counts())
#1    71
#0    59
#2    18

x_train, x_test, y_train, y_test = train_test_split(
    x_new, y_new, train_size=0.75, shuffle=True, random_state=66, stratify=y_new
) # stratify y의 라벨의 수를 동일하게 나오게 한다. 

print(pd.Series(y_train).value_counts())

#1    53
#0    44
#2    14

model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('모델스코어:',score)

#모델스코어: 0.9459459459459459
############################# SMOTE 적용 ############################################
print("=============================smote 적용 ==================")

smote= SMOTE(random_state=66) 

x_smote_train, y_smote_train = smote.fit_resample(x_train,y_train)

print(pd.Series(y_smote_train).value_counts())
#0    53
#1    53
#2    53

print(x_smote_train.shape, y_smote_train.shape)
#(159, 13) (159,)


print('smote 전:',x_train.shape, y_train.shape)
print('smote 후:',x_smote_train.shape, y_smote_train.shape)
print('smote 전 레이블값 분포:\n',pd.Series(y_train).value_counts())
print('smote 후 레이블값 분포:\n',pd.Series(y_smote_train).value_counts())

model2 = XGBClassifier(n_jobs = -1)
model2.fit(x_smote_train, y_smote_train,eval_metric='mlogloss')

score = model2.score(x_test,y_test)
print("model2 : ", score)
#model2 :  0.972972972972973

