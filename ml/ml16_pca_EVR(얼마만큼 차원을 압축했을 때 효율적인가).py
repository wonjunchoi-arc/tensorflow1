#PCA는 컬럼을 삭제하는 것이 아니라 임베딩처럼 합쳐지는 것
#그러므로 원래의 값으로도 되돌릴 수 있다. 
import numpy as np
from sklearn import datasets
from sklearn.datasets import load_boston
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

datasets =load_boston()
x=datasets.data
y = datasets.target
print(x.shape, y.shape) #(506, 13) (506,)


pca = PCA(n_components=5)

x = pca.fit_transform(x)
print(x)
print(x.shape)

pca_EVR=pca.explained_variance_ratio_
print(pca_EVR)
# 압축한 결과에 대한 중요도 큰 순서대로 나열함
'''
[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
 0.05365605 0.04336832 0.007831
'''

print(sum(pca_EVR))
#sum값을 기준으로 0.95이상이라면 등 나만의 기준을 정하여 PCA갯수를 조정하면 되는 것이다!!

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.995)+1)





# 누적해서 계산해주는 친구 이 친구를 활용해서 pca갯수 설정 
'''
[0.40242142 0.55165324 0.67224947 0.76779711 0.83401567 0.89428759
 0.94794364 0.99131196 0.99914395]
'''


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
