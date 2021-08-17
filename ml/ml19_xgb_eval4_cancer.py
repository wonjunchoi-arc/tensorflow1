#실습 
#분류 -> eval_metric을 찾아서 추가 
from sklearn import datasets
from xgboost import XGBRegressor,XGBRFRegressor, XGBClassifier,XGBRFClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = load_breast_cancer()
x= dataset['data']
y = dataset['target']

print(x.shape, y.shape) #(506, 13) (506,)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, train_size=0.8, random_state=66)

# scaler =MinMaxScaler()
scaler = StandardScaler()
# scaler.fit(x_train)
# x_train = scaler.transform(x_train)
# x_test = scaler.transform(x_test)

#2.모델
model = XGBClassifier()

'''
params = {
  'colsample_bynode': 0.8,
  'learning_rate': 1,
  'max_depth': 5,
  'num_parallel_tree': 100,
  'objective': 'binary:logistic',
  'subsample': 0.8,
  'tree_method': 'gpu_hist'
}

'''

#3. 훈련
model.fit(x_train, y_train,verbose=1, 
eval_metric='mlogloss',
#regressor와 classifier에 들어가는 평가지표가 다름
eval_set=[(x_train, y_train),(x_test, y_test)])

#4. 평가
results = model.score(x_test, y_test)
print("results:", results)

y_predict =model.predict(x_test)
r2 =r2_score(y_test, y_predict)
print("r2:", r2)

'''
[97]    validation_0-rmse:0.96477       validation_0-mae:0.72735    validation_0-logloss:-791.72473 validation_1-rmse:2.33095   validation_1-mae:1.75278   validation_1-logloss:-799.52997
[98]    validation_0-rmse:0.95683       validation_0-mae:0.72212    validation_0-logloss:-791.72473 validation_1-rmse:2.32714   validation_1-mae:1.75079   validation_1-logloss:-799.52997
[99]    validation_0-rmse:0.94395       validation_0-mae:0.71374    validation_0-logloss:-791.72473 validation_1-rmse:2.32317   validation_1-mae:1.74934   validation_1-logloss:-799.52997
results: 0.9354279880084962
r2: 0.9354279880084962

'''