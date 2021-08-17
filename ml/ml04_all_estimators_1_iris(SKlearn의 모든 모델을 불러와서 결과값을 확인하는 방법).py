from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

import numpy as np 
from sklearn.datasets import load_iris
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import warnings
warnings.filterwarnings('ignore')


datasets = load_iris()
print(datasets.DESCR)
print(datasets.feature_names)

x =datasets.data
y =datasets.target

print(x.shape, y.shape) #(150, 4) (150,)
print(y)


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test =train_test_split(
    x,y, test_size=0.7, random_state=66)

from sklearn.preprocessing import StandardScaler,MinMaxScaler
# scaler =MinMaxScaler()
scaler = StandardScaler()
scaler.fit(x_train)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


#2. 모델 구성 # all_estimators는 즉 모든 ml모델을 불러오는 것!
allAlgorithms= all_estimators(type_filter='classifier')
# print(allAlgorithms)
print('모델의 갯수',len(allAlgorithms))
for name, algorithms in allAlgorithms:
    try:
        model = algorithms()

        model.fit(x_train, y_train)

        y_predict = model.predict(x_test)

        acc = accuracy_score(y_test, y_predict)
        print(name, '의 정답률:', acc)
    except:
        print(name)
        continue

'''
AdaBoostClassifier 의 정답률: 0.8571428571428571
BaggingClassifier 의 정답률: 0.9428571428571428
BernoulliNB 의 정답률: 0.8
CalibratedClassifierCV 의 정답률: 0.8666666666666667
CategoricalNB
ClassifierChain
ComplementNB
DecisionTreeClassifier 의 정답률: 0.9428571428571428
DummyClassifier 의 정답률: 0.3238095238095238
ExtraTreeClassifier 의 정답률: 0.8380952380952381
ExtraTreesClassifier 의 정답률: 0.9238095238095239
GaussianNB 의 정답률: 0.9714285714285714
GaussianProcessClassifier 의 정답률: 0.9428571428571428
GradientBoostingClassifier 의 정답률: 0.9428571428571428
HistGradientBoostingClassifier 의 정답률: 0.8095238095238095
KNeighborsClassifier 의 정답률: 0.9523809523809523
LabelPropagation 의 정답률: 0.9238095238095239
LabelSpreading 의 정답률: 0.9238095238095239
LinearDiscriminantAnalysis 의 정답률: 1.0
LinearSVC 의 정답률: 0.9523809523809523
LogisticRegression 의 정답률: 0.9714285714285714
LogisticRegressionCV 의 정답률: 0.9428571428571428
MLPClassifier 의 정답률: 0.9619047619047619
MultiOutputClassifier
OneVsOneClassifier
OneVsRestClassifier
OutputCodeClassifier
PassiveAggressiveClassifier 의 정답률: 0.8857142857142857
Perceptron 의 정답률: 0.9047619047619048
QuadraticDiscriminantAnalysis 의 정답률: 0.9809523809523809
RadiusNeighborsClassifier
RandomForestClassifier 의 정답률: 0.9428571428571428
RidgeClassifier 의 정답률: 0.8380952380952381
RidgeClassifierCV 의 정답률: 0.8380952380952381
SGDClassifier 의 정답률: 0.9238095238095239
SVC 의 정답률: 0.9428571428571428
StackingClassifier
VotingClassifier

'''