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
model = XGBRegressor(n_estimators=20, learning_rate=0.05, n_jobs=1)

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
eval_metric=['rmse','logloss']
,eval_set=[(x_train, y_train),(x_test, y_test)])

#4. 평가
results = model.score(x_test, y_test)
print("results:", results)

y_predict =model.predict(x_test)
r2 =r2_score(y_test, y_predict)
print("r2:", r2)


print("====================================")
hist = model.evals_result()
print(hist)

# A = hist['validation_0']
# B = hist['validation_1']

# histA = dict(A)
# histB = dict(B)

# print(type(hist))
# print(histA['rmse'].history)

# import matplotlib.pyplot as plt

# plt.figure(figsize=(12, 4))

# plt.subplot(1, 2, 1)
# plt.title('loss of XGBRegressor', fontsize= 15)
# plt.plot(histA['rmse'], 'b-', label='rmse')
# plt.plot(histB['rmse'],'r--', label='val_rmse')
# plt.xlabel('Epoch')
# plt.legend()

# plt.subplot(1, 2, 2)
# plt.title('accuracy of Bidirectional LSTM (model3) ', fontsize= 15)
# plt.plot(hist.history['mae'], 'g-', label='mae')
# plt.plot(hist.history['val_mae'],'k--', label='val_mae')
# plt.xlabel('Epoch')
# plt.legend()

print("--------------------------선생님 코드----------------")
import matplotlib.pyplot as plt

epochs = len(results['validation_0']['rmse'])
x_axis = range(0, epochs)

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['logloss'], label='Train')
ax.plot(x_axis, results['validation_1']['logloss'], label='Test')
ax.legend()
plt.ylabel('Log Loss')
plt.title('XGBoost Log Loss')
# plt.show()

fig, ax = plt.subplots()
ax.plot(x_axis, results['validation_0']['rmse'], label='Train')
ax.plot(x_axis, results['validation_1']['rmse'], label='Test')
ax.legend()
plt.ylabel('Rmse')
plt.title('XGBoost RMSE')
plt.show()
