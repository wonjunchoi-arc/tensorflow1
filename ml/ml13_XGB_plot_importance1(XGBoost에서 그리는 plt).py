from sklearn.tree import DecisionTreeClassifier,DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor,GradientBoostingRegressor
from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRFRegressor
from xgboost import plot_importance


#1. 데이터
datasets = load_boston()
x_train, x_test, y_train, y_test = train_test_split(
    datasets.data, datasets.target, train_size=0.8, random_state=66
)

#2. 모델
model =DecisionTreeRegressor()
model =RandomForestRegressor()
model = GradientBoostingRegressor()
model = XGBRFRegressor()

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
acc =model.score(x_test, y_test)
print('acc:', acc)

print(model.feature_importances_)

'''

'''
import matplotlib.pyplot as plt
import numpy as np


# def plot_feature_importances_dataset(model):
#     n_features = datasets.data.shape[1]
#     plt.barh(np.arange(n_features), model.feature_importances_,
#     align='center')
#     plt.yticks(np.arange(n_features), datasets.feature_names)
#     plt.xlabel("Feature Importances")
#     plt.ylabel("Features")
#     plt.ylim(-1, n_features)

# plot_feature_importances_dataset(model)
# plt.show()

plot_importance(model)
plt.show()

'''
1.de
acc: 0.7973681092324066

2. RAN
acc: 0.9217961332680954
[0.0713492  0.0132194  0.52293176 0.39249964]   

3. GB
acc: 0.9456150989046179


4.XGB
acc: 0.918604686183674

'''