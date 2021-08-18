#실습
#데이터는 diabets

#1. 상단 모델에 그리드서치 랜덤서치로 튜닝 모델 구성
#최적의 R2값과 피처임포턴스 구할 것

#2 .위 스레드 값으로 SelectFrom돌려서
#최적의 피쳐갯수 구할 것

#3 . 위 피쳐 갯수로 피쳐 갯수를 조정한뒤
#그걸로 다시 랜덤서치 그리드 서치해서 
#최적의 R2 구할 것

# 1번값과 3번값 비교 # 0.47이상
import numpy as np 
from scipy.sparse import data
from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel
from sklearn.datasets import load_diabetes

dataset = load_diabetes()

x, y = load_diabetes(return_X_y=True)
print(x.shape , y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(

    x,y , train_size=0.8, shuffle=True, random_state=66
)


from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 



parameter = [
{"n_estimators":[1, 10, 100, 1000],"max_depth":[2,4,6,8,10],'learning_rate': [0.005,0.01,0.05,0.1,0.2],"colsample_bytree":[0.1,0.6, 0.9, 1],"colsample_bylevel":[0.6, 0.7, 0.9,0.1]}, #교차검증 해서 여기만 (4*5)20번
] # 총 ==> 90번 돌아간다!!!
'''
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=0.9,   
             colsample_bynode=1, colsample_bytree=0.9, gamma=0.001, gpu_id=-1,
             importance_type='gain', interaction_constraints='',
             learning_rate=0.05, max_delta_step=0, max_depth=2,
             min_child_weight=1, missing=nan, monotone_constraints='()',
             n_estimators=100, n_jobs=8, num_parallel_tree=1, random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
             tree_method='exact', validate_parameters=1, verbosity=None)
best_score: 0.4838666527720566
'''

from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel ## 모델에서 뭔가를 선택한다는 거겠찌!

#2. 모델
model = XGBRegressor()

# #3. 훈련
model.fit(x_train, y_train)

# #4. 평가, 예측
score = model.score(x_test, y_test)
print("model.score:",score)

aaa = model.feature_importances_
print(aaa) #컬럼의 순서별로 나온거임


thresholds = np.sort(model.feature_importances_)
print(thresholds) #컬럼별 크기로 정렬해준것!

# for thresh in thresholds : 
#     selection = SelectFromModel(model, threshold=thresh, prefit=True)
#     # thresh 이상 되는 수치를 가진 컬럼들로 모델을 구성 
#     # print(selection)

#     select_x_train = selection.transform(x_train)
#     select_x_test = selection.transform(x_test)
#     #삭제한 컬럼에 맞게 x값 재구성
#     print(select_x_train.shape)

#     selection_model = XGBRegressor(n_jobs=8,base_score=0.5, booster='gbtree', colsample_bylevel=0.9,   
#              colsample_bynode=1, colsample_bytree=0.9, gamma=0.001, gpu_id=-1,
#              importance_type='gain', interaction_constraints='',
#              learning_rate=0.05, max_delta_step=0, max_depth=2,
#              min_child_weight=1, monotone_constraints='()',
#              n_estimators=100, num_parallel_tree=1, random_state=0,
#              reg_alpha=0, reg_lambda=1, scale_pos_weight=1, subsample=1,
#              tree_method='exact', validate_parameters=1, verbosity=None)
#     selection_model.fit(select_x_train, y_train)
#     #삭제한 컬럼으로 모델 다시 훈련!!

#     y_predict = selection_model.predict(select_x_test)

#     score = r2_score(y_test, y_predict)

#     print("Thresh = %.3f, m=%d, R2 :  %.2f%%" %(thresh, select_x_train.shape[1],score*100))


selection = SelectFromModel(model, threshold=0.066, prefit=True)
    # thresh 이상 되는 수치를 가진 컬럼들로 모델을 구성 
    # print(selection)

select_x_train = selection.transform(x_train)
select_x_test = selection.transform(x_test)
#삭제한 컬럼에 맞게 x값 재구성
print(select_x_train.shape)

selection_model = GridSearchCV(XGBRegressor(), parameter,verbose=1 )
selection_model.fit(select_x_train, y_train)
#삭제한 컬럼으로 모델 다시 훈련!!



#4. 예측 평가
print("최적의 매개변수:", selection_model.best_estimator_)
print("best_score :", selection_model.best_score_)
##x_train에 대한 최적의값



# x_test와 y_test에 대한 값
print("model.score :", selection_model.score(select_x_test,y_test))
y_predict = selection_model.predict(select_x_test)
print("정답률 :", r2_score(y_test, y_predict))