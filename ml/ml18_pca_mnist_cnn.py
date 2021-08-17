#실습
# mnist데이터를 pca를 통해 cnn으로 구성

#ml 0.95 이상 n_compo
#xgb 모델을 만들것
#mnist dnn 보다 성능 좋게
#dnn cnn 비교!! 

#RandomForest
# Grid, RandomSearch with mnist

params = [
    {"n_estimators":[90, 100, 110, 200, 300], 
    "learning_rate":[0.001, 0.01],
    "max_depth":[4, 5, 6], 
    "colsample_bytree":[0.6, 0.9, 1], 
    "colsample_bylevel":[0.6, 0.7, 0.9],
    "n_jobs":[-1]}
]

# Practice : n_comp upper than 0.95 : 154
# make model -> Tensorflow DNN, compare with banila

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np
from tensorflow.keras.datasets import mnist

from sklearn.decomposition import PCA
import warnings 
warnings.filterwarnings(action='ignore')

(x_train, y_train), (x_test, y_test) = mnist.load_data() # (60000, 28, 28) (60000,) (10000, 28, 28) (10000,)

x = np.append(x_train, x_test, axis=0) # (70000, 28, 28)
y = np.append(y_train, y_test, axis=0) # (70000,)

x = x.reshape(x.shape[0], x.shape[1]*x.shape[2])

pca = PCA(n_components=225)
x = pca.fit_transform(x)

pca_EVR = pca.explained_variance_ratio_
print(pca_EVR)

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.95)+1) 

y = y.reshape(70000,1)

from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder(sparse=False) # sparse의 default는 true로 matrix행렬로 반환한다. 하지만 False는 array로 반환 둘의 차이는 잘..
y = en.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y,
      test_size=0.14, shuffle=True, random_state=77)

from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler, MaxAbsScaler, QuantileTransformer, PowerTransformer
scaler = QuantileTransformer()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test) 




x_train =x_train.reshape(x_train.shape[0], 15,15,1)
x_test =x_test.reshape(x_test.shape[0], 15,15,1)

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, MaxPool2D, GlobalAveragePooling2D, Dropout,MaxPooling2D
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV, KFold, cross_val_score, train_test_split

model =Sequential()
model.add(Conv2D(20, kernel_size=(2,2), padding='same',input_shape=(13,13,1)))
model.add(Conv2D(20, (2,2), activation='relu'))
model.add(Conv2D(30, (2,2), activation='relu'))
model.add(Conv2D(40, (2,2), activation='relu'))
model.add(Conv2D(50, (2,2), activation='relu'))
model.add(MaxPool2D())   
model.add(Conv2D(70, (2,2), activation='relu'))
model.add(Conv2D(80, (2,2), activation='relu'))
model.add(Conv2D(900, (2,2), activation='relu'))
model.add(Flatten())
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(10, activation='softmax'))



model.summary()

# #3. 컴파일, 훈련 , metrics=['acc']

import time

start_time = time.time()
from tensorflow.keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='val_loss', patience= 50, mode= 'min', verbose=1)
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(x_train,y_train, epochs=100,
 batch_size=1000, validation_split=0.2,verbose=3)

end_time = time.time() - start_time

# 4. predict eval -> no need to
#4. 평가, 예측
loss = model.evaluate(x_test, y_test)
print('loss', loss[0])
print('accuracy', loss[1])


'''
CNN
time =  12.263086080551147
loss :  0.08342713862657547
acc :  0.9821000099182129
DNN
time :  20.324632167816162
loss :  0.09825558215379715
acc :  0.9785000085830688
PCA_DNN 0.95
time :  55.58410167694092
loss :  0.0827394351363182
acc :  0.9757142663002014
PCA_DNN 0.999
time :  36.505455017089844
loss :  0.2318371683359146
acc :  0.9433731436729431
RandomizedSearchCV_XGB

PCA_CNN 0.95

'''