import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer


datasets = load_breast_cancer()

#1. 데이터
print(datasets.DESCR)

print(datasets.feature_names)

x= datasets.data
y= datasets.target

print(x.shape , y.shape)

print(y[:20])
print(np.unique(y))
#unique 는 특이한 애들을 찾는다. [0 1] 밖에 없다.  인진 분류 모델 

x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
scaler =MinMaxScaler()
# scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성
from sklearn.svm import LinearSVC, SVC
from sklearn.neighbors import KNeighborsClassifier,KNeighborsRegressor
#분류 Classifier , 회귀면 regress
from sklearn.linear_model import LogisticRegression
# LogisticRegression은 분류모델
from sklearn.tree import DecisionTreeClassifier #의사결정나무
from sklearn.ensemble import RandomForestClassifier #의사결정 나무가 모여 숲을 이룸(앙상블)

model = LinearSVC()
#이벨류에이트대신 쓰는 SCORE!! 0.9598997493734336
model = SVC()
#이벨류에이트대신 쓰는 SCORE!! 0.9624060150375939

model = KNeighborsClassifier()
# 이벨류에이트대신 쓰는 SCORE!! 0.9548872180451128

model = LogisticRegression()
#이벨류에이트대신 쓰는 SCORE!! 0.9523809523809523

model = DecisionTreeClassifier()
#이벨류에이트대신 쓰는 SCORE!! 0.8922305764411027

model = RandomForestClassifier()
#acc_score 0.9473684210526315




#3. 컴파일 및 훈련

# from tensorflow.keras.callbacks import EarlyStopping
# es = EarlyStopping(monitor='val_loss', patience= 5, mode= 'min', verbose=1)
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



