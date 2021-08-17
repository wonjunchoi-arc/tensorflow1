from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor


#1. 데이터
datasets = load_wine()
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

import matplotlib.pyplot as plt
import numpy as np


def plot_feature_importances_dataset(model):
    n_features = datasets.data.shape[1]
    plt.barh(np.arange(n_features), model.feature_importances_,
    align='center')
    plt.yticks(np.arange(n_features), datasets.feature_names)
    plt.xlabel("Feature Importances")
    plt.ylabel("Features")
    plt.ylim(-1, n_features)

plot_feature_importances_dataset(model)
plt.show()

'''
1.DEC
acc: 0.9629629629629629
[0.0055201  0.         0.         0.03457325 0.     
    0.
 0.05378155 0.         0.         0.12477838 0.     
    0.37054514
 0.41080159]

2. RAn
acc: 1.0
[0.14026716 0.03697919 0.01360676 0.02604612 0.02726898 0.04501779
 0.1447117  0.01160828 0.02038732 0.13300717 0.06831458 0.14883991
 0.18394505]

 3.GB
aacc: 0.9629629629629629
[2.82602994e-03 5.49829686e-02 6.28444383e-03 1.09137900e-02
 9.94168696e-04 2.80338219e-05 1.09717295e-01 9.90881342e-05
 8.20179648e-03 2.70685616e-01 2.17855675e-02 2.40652487e-01
 2.72828715e-01]

 4. XGB
acc: 0.9814814814814815
[0.01761142 0.03772116 0.01630196 0.03029412 0.0166176  0.00937404
 0.11020903 0.02074783 0.01344708 0.16075327 0.02660201 0.42059025
 0.1197302 ]
'''
