#cancer로 맹그러봐
# f1
#라벨 0을 112개 삭제


#cancer로 맹그러봐
# f1


from imblearn.over_sampling import SMOTE
import numpy as np
from numpy.core.fromnumeric import shape
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_wine, load_breast_cancer
from re import X
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import warnings
from sklearn.metrics import f1_score,accuracy_score

from xgboost.sklearn import XGBClassifier
warnings.filterwarnings('ignore')

dataset = load_breast_cancer()

x = dataset.data
print(x.shape)

# print(dataset.target_names)
y = dataset.target
print(y.shape)

# print(x.shape, y.shape)
# print(pd.Series(y).value_counts())

x = pd.DataFrame(x)
y = pd.DataFrame(y)
y.columns = ['label']
data = pd.concat([x,y],axis=1)
data= data.sort_values(by='label')

data = data[112:]
print(data)

# 1    357
# 0    212

# y_new= y[:1]
# print(pd.Series(y_new).value_counts())

"""
x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66
) # stratify y의 라벨의 수를 동일하게 나오게 한다. 
print(pd.Series(y_train).value_counts())


model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('모델스코어:',score)

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred)
print("f1_score:", f1)
# 모델스코어: 0.9824561403508771
# f1_score: 0.9863013698630138

############################# SMOTE 적용 ############################################
smote =SMOTE(random_state=66, k_neighbors=10)

start_time = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train,y_train)
end_time = time.time()
print("걸린시간:", end_time-start_time) 

print(pd.Series(y_smote_train).value_counts())
# 0    247
# 1    247

print('smote 전:',x_train.shape, y_train.shape)
print('smote 후:',x_smote_train.shape, y_smote_train.shape)
print('smote 전 레이블값 분포:\n',pd.Series(y_train).value_counts())
print('smote 후 레이블값 분포:\n',pd.Series(y_smote_train).value_counts())


model2 = XGBClassifier(n_jobs = -1)
model2.fit(x_smote_train, y_smote_train,eval_metric='mlogloss')

score = model2.score(x_test,y_test)
print("model2 : ", score)

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred)
print("f1_score:", f1)

# model2 :  0.9824561403508771
# f1_score: 0.9863013698630138
"""