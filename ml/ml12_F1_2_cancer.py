from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor
import pandas as pd
import numpy as np


#1. 데이터
datasets = load_breast_cancer()
print(datasets.feature_names)

x = pd.DataFrame(data=datasets.data)
y = pd.DataFrame(datasets.target)


x.drop([2,8,9,11,16,26], axis=1, inplace=True)
print(len(x.columns.tolist()))



x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.7, random_state=66
)

#2. 모델
model =DecisionTreeClassifier()
model =RandomForestClassifier()
model = GradientBoostingClassifier()
model = XGBClassifier()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc =model.score(x_test, y_test)
print('acc:', acc)

print(model.feature_importances_)

'''
acc: 0.9333333333333333
[0.0125026  0.         0.53835801 0.44913938]
해당 모델을 돌린 조건에서 의 컬럼이 미치는 영향

'''
import matplotlib.pyplot as plt
import numpy as np

def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]-6
    plt.barh(np.arange(n_features), model.feature_importances_,
    align='center')
    plt.yticks(np.arange(n_features), x.columns.tolist())
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

'''
1. Dec
acc: 0.9532163742690059
[0.         0.0271628  0.         0.         0.     
    0.
 0.         0.00800332 0.         0.02881267 0.01402487 0.
 0.         0.         0.         0.         0.00323367 0.
 0.00711407 0.00711407 0.         0.04366583 0.00973904 0.73439563
 0.01422813 0.         0.         0.10250589 0.     
    0.        ]

2. RAn
acc: 0.9707602339181286
[0.05628716 0.01804882 0.018506   0.08133377 0.00738034 0.01544968
 0.04037253 0.06400322 0.0050658  0.00409211 0.01600564 0.00571497
 0.01544556 0.02350008 0.0048503  0.00427426 0.00747851 0.00422729
 0.00309069 0.00573312 0.16055655 0.01885634 0.10658588 0.16587334
 0.01231705 0.00738367 0.02300766 0.09032847 0.00619589 0.00803531]

 3.GB
acc: 0.9649122807017544
[5.57071682e-05 4.63617099e-02 4.41383624e-04 1.63637201e-03
 3.18323933e-05 2.75450850e-03 2.84515921e-03 3.41253768e-02
 5.35042160e-04 3.69250446e-07 6.29062827e-03 1.10249585e-04
 2.19975936e-04 1.90291339e-02 3.39703200e-03 2.94618971e-03
 1.96422758e-03 3.35957677e-03 5.49810999e-06 1.22037092e-03
 2.88045460e-01 3.13207883e-02 2.78463843e-02 4.22033599e-01
 3.48449007e-03 1.09447628e-04 2.85370501e-03 9.69296855e-02
 1.25164544e-05 3.35797593e-05]

 4. XGB
acc: 0.9824561403508771
[0.03474978 0.03467545 0.         0.00751567 0.00292065 0.00687929
 0.01235841 0.02046652 0.         0.00054058 0.01655321 0.00162256
 0.01760868 0.00867776 0.01414008 0.00689266 0.00062871 0.00279585
 0.00357394 0.00587361 0.15269534 0.01300185 0.33814007 0.21980153
 0.00525721 0.         0.00569597 0.06131698 0.00166174 0.00395583]

'''