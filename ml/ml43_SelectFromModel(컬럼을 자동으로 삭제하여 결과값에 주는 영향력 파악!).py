from scipy.sparse import data
from sklearn import datasets
from xgboost import XGBRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import r2_score
from sklearn.feature_selection import SelectFromModel ## 모델에서 뭔가를 선택한다는 거겠찌!

# dataset = load_boston()
# x = dataset.data
# y= dataset.target

x, y = load_boston(return_X_y=True) #TRUE하면 x,y 분리해서 반환해준다.
print(x.shape , y.shape) #(506, 13) (506,)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,test_size=0.8, shuffle=True, random_state=66
)

#2. 모델
model = XGBRegressor(n_jobs=8)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
score = model.score(x_test, y_test)
print("model.score:",score)

# aaa = model.feature_importances_
# print(aaa) #컬럼의 순서별로 나온거임


thresholds = np.sort(model.feature_importances_)
print(thresholds) #컬럼별 크기로 정렬해준것!

for thresh in thresholds : 
    selection = SelectFromModel(model, threshold=thresh, prefit=True)
    # thresh 이상 되는 수치를 가진 컬럼들로 모델을 구성 
    print(selection)

    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    #삭제한 컬럼에 맞게 x값 재구성
    print(select_x_train.shape)

    selection_model = XGBRegressor(n_jobs = -1)
    selection_model.fit(select_x_train, y_train)
    #삭제한 컬럼으로 모델 다시 훈련!!

    y_predict = selection_model.predict(select_x_test)

    score = r2_score(y_test, y_predict)

    print("Thresh = %.3f, m=%d, R2 :  %.2f%%" %(thresh, select_x_train.shape[1],score*100))