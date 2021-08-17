from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier #의사결정나무
from sklearn.ensemble import RandomForestClassifier #의사결정 나무가 모여 숲을 이룸(앙상블)

import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

datasets = load_wine()



x= datasets.data
y = datasets.target

print(datasets.DESCR)
print(x.shape, y.shape)
print(y)

#1 .데이터


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)



from sklearn.preprocessing import StandardScaler,MinMaxScaler,QuantileTransformer, RobustScaler
# scaler =MinMaxScaler()
# scaler = StandardScaler()
# scaler =QuantileTransformer()
scaler = RobustScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성

model = LinearSVC()
#이벨류에이트대신 쓰는 SCORE!! 0.992

model = SVC()
# acc_score 0.984

model = KNeighborsClassifier()
# acc_score 0.976

model = LogisticRegression()
#acc_score 0.992

model = DecisionTreeClassifier()
#acc_score 0.928

model = RandomForestClassifier()
#acc_score 0.96

#3. 컴파일 및 훈련

model.fit(x_train,y_train)


#4. 예측 평가

y_predict = model.predict(x_test)
# print('y_predict', y_predict)

predict = []
for i in y_predict:
    predict.append(np.round(i))


from sklearn.metrics import r2_score, accuracy_score
results = model.score(x_test, y_test)
print('이벨류에이트대신 쓰는 SCORE!!', results)

acc= accuracy_score(y_test, predict)
print('acc_score', acc)

'''
머신 러닝에는 이벨류에이트 읎다!!
results = model.evaluate(x_train, y_train)
print('model score!!!!',results)

'''

