#실습 
#분류 -> eval_metric을 찾아서 추가 
from sklearn import datasets
from xgboost import XGBRegressor,XGBRFRegressor
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler

#1. 데이터
dataset = load_diabetes()
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
model = XGBRegressor(n_estimators=150, learning_rate=0.05, n_jobs=1)

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
hist = model.fit(x_train, y_train,verbose=1, 
eval_metric='rmse',#'mae','logloss'
eval_set=[(x_train, y_train),(x_test, y_test)],
early_stopping_rounds=10)



#4. 평가
results = model.score(x_test, y_test)
print("results:", results)

y_predict =model.predict(x_test)
r2 =r2_score(y_test, y_predict)
print("r2:", r2)


print("====================================")
model.evals_result()
print(hist)

#저장
# import pickle
# pickle.dump(model, open('./_save/xgb_save/ml21.pickle.dat', 'wb'))
##############################pickle#################################

#저장
# import pickle
# pickle.dump(model, open('./_save/xgb_save/ml21.pickle.dat', 'wb'))
import joblib
joblib.dump(model,'./_save/xgb_save/ml22.joblib.dat')
########################################################################
