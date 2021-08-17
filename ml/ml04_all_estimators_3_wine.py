import numpy as np
from sklearn.datasets import load_wine
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.utils import all_estimators
from sklearn.metrics import accuracy_score

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
daBoostClassifier 의 정답률: 0.848
BaggingClassifier 의 정답률: 0.96
BernoulliNB 의 정답률: 0.944
CalibratedClassifierCV 의 정답률: 0.976
CategoricalNB
ClassifierChain
ComplementNB
DecisionTreeClassifier 의 정답률: 0.864
DummyClassifier 의 정답률: 0.384
ExtraTreeClassifier 의 정답률: 0.8
ExtraTreesClassifier 의 정답률: 0.968
GaussianNB 의 정답률: 0.96
GaussianProcessClassifier 의 정답률: 0.976
GradientBoostingClassifier 의 정답률: 0.888
HistGradientBoostingClassifier 의 정답률: 0.96
KNeighborsClassifier 의 정답률: 0.976
LabelPropagation 의 정답률: 0.92
LabelSpreading 의 정답률: 0.92
LinearDiscriminantAnalysis 의 정답률: 0.968
LinearSVC 의 정답률: 0.992
LogisticRegression 의 정답률: 0.992
LogisticRegressionCV 의 정답률: 0.952
C:\ProgramData\Anaconda3\lib\site-packages\sklearn\neural_network\_multilayer_perceptron.py:614: ConvergenceWarning: Stochastic Optimizer: Maximum iterations (200) reached and the optimization hasn't converged yet.
  warnings.warn(
MLPClassifier 의 정답률: 0.976
MultiOutputClassifier
MultinomialNB
NearestCentroid 의 정답률: 0.96
NuSVC 의 정답률: 0.976
OneVsOneClassifier
OneVsRestClassifier
OutputCodeClassifier
PassiveAggressiveClassifier 의 정답률: 0.992
Perceptron 의 정답률: 0.968
QuadraticDiscriminantAnalysis 의 정답률: 0.696
RadiusNeighborsClassifier
RandomForestClassifier 의 정답률: 0.968
RidgeClassifier 의 정답률: 0.984
RidgeClassifierCV 의 정답률: 0.984
SGDClassifier 의 정답률: 0.968
SVC 의 정답률: 0.984
StackingClassifier
VotingClassifier
'''