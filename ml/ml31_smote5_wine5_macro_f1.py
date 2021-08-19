#실습

#범위변경 345 =>0  6 => 1  789- > 2
#스포트하기전 후 비교해볼것


from imblearn.over_sampling import SMOTE
import numpy as np
from numpy.lib.function_base import average
import pandas as pd
from pandas.io.stata import precision_loss_doc
from sklearn import datasets
from re import X
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import time
import warnings
import time
from sklearn.metrics import f1_score,accuracy_score

from xgboost.sklearn import XGBClassifier
warnings.filterwarnings('ignore')


dataset = pd.read_csv('../data/winequality-white.csv', sep=';', 
                        index_col=None, header=0)

dataset=dataset.values

x= dataset[:,:-1]
y= dataset[:,-1]
print(x.shape, y.shape)

print(pd.Series(y).value_counts())
'''
6.0    2198
5.0    1457
7.0     880
8.0     175
4.0     163
3.0      20
9.0       5
'''

################################################33
################ 라베 ㄹ대통합!!!!!!!!!!!!!11
################################################333




# print(type(y))
# newlist = []
# for i in y : 
#     if i == 9:
#         newlist.append(i-1)
#     else:
#         newlist.append(i)
# # print(newlist)
# y = np.array(newlist)
# print(type(y))
#범위변경 345 =>0  6 => 1  789- > 2

######### 선생님 코드#######################
for index, value in enumerate(y):
    if value == 3 :
        y[index] = 0
    if value == 4 :
        y[index] = 0
    if value == 5 :
        y[index] = 0
    if value == 6 :
        y[index] = 1

    if value == 7 :
        y[index] = 2
    if value == 8 :
        y[index] = 2
    if value == 9 :
        y[index] = 2
    # elif value ==4 : 
    #     y[index] = 5
print(pd.Series(y).value_counts())


x_train, x_test, y_train, y_test = train_test_split(
    x, y, train_size=0.7, shuffle=True, random_state=66
) # stratify y의 라벨의 수를 동일하게 나오게 한다. 

print(pd.Series(y_train).value_counts())

# 6.0    1648
# 5.0    1093
# 7.0     660
# 8.0     131
# 4.0     122
# 3.0      15
# 9.0       4





# print(pd.Series(y).value_counts())


model = XGBClassifier(n_jobs=-1)
model.fit(x_train, y_train, eval_metric='mlogloss')

score = model.score(x_test, y_test)
print('모델스코어:',score)

y_pred = model.predict(x_test)
f1 = f1_score(y_test, y_pred,average='macro')
print("f1_score:", f1)


# 모델스코어: 0.7068027210884353
# f1_score: 0.7049595056751449
############################# SMOTE 적용 ############################################
print("=============================smote 적용 ==================")

smote= SMOTE(random_state=66, k_neighbors=10) 


start_time = time.time()
x_smote_train, y_smote_train = smote.fit_resample(x_train,y_train)
end_time = time.time()
print("걸린시간:", end_time-start_time) 

#네이버는 주변의 값을들 몇개의 선으로 연결시킬것이냐다
print(pd.Series(y_smote_train).value_counts())


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

y_pred = model2.predict(x_test)
f1 = f1_score(y_test, y_pred,average='macro')
print("f1_score:", f1)

# model2 :  0.7013605442176871
# f1_score: 0.7023279294360677
#k_neibor가 높을수록 값증가 일정 수준만