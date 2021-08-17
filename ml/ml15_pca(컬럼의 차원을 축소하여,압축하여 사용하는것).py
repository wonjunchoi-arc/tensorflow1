#PCA는 컬럼을 삭제하는 것이 아니라 임베딩처럼 합쳐지는 것
#그러므로 원래의 값으로도 되돌릴 수 있다. 
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_diabetes
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

datasets =load_diabetes()
x=datasets.data
y = datasets.target
print(x.shape, y.shape) #(506, 13) (506,)


pca = PCA(n_components=2)

x = pca.fit_transform(x)
print(x)
print(x.shape)

x_train, x_test, y_train, y_test = train_test_split(
    x,y,train_size=0.8, random_state=66, shuffle=True
)

#2. 모델
from xgboost import XGBRegressor
model =XGBRegressor()

#3. 훈련
model.fit(x,y)

#4. 평가, 예측
results = model.score(x,y)
print("결과:",results) #결과: 0.9999349120798557