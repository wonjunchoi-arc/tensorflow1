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
model = XGBRegressor(n_estimators=10000, learning_rate=0.01, #n_jobs=2
      tree_method ='gpu_hist',gpu_id=0,predictors ='gpu_predictor', #cpu_predictor 
)

#3. 훈련
import time
start_time=time.time()

hist = model.fit(x_train, y_train,verbose=1, 
eval_metric='rmse',#'mae','logloss'
eval_set=[(x_train, y_train),(x_test, y_test)])

print("걸린시간:",time.time()-start_time)
#i7-9700 / 2080tu
#n_jobs =1 : 걸린시간: 11.773041486740112
#n =2  걸린시간: 10.789015531539917
#n4 : 걸린시간: 10.661615133285522
#n-1 : 걸린시간: 12.695220708847046


#Gpu 썻을 때 걸린시간: 52.20238900184631


'''
CPU
복잡한 계산을 코어 갯수 만큼씩 처리하게 된다.
예로 복잡한 팩토리얼 계산식을 2개 계산해야 한다고 했을 때 CPU로 계산을 해주면 빨리할 수 있다.
단점 – 간단하고 많은 계산식은 오래걸린다
2. GPU

간단한 아주 많은 계산식을 동시에 빠르게 처리할 수 있다.
예로 1000개의 덧셈식을 한번에 병렬로 처리가 가능하다.
단점 – 초기에 알고리즘을 하드웨어에 병렬로 부여해 주어야 하고, 복잡한 식을 입력하면 도리어 CPU 연산 속도보다 느려질 수 있다.


'''