#coefficient 개수

x = [-3, 31, -11, 4, 0, 22, -2, -5, -25, -14]
y = [-3, 65, -19, 11, 3,47,-1,-77,-47,-25]

import pandas as pd
df =pd.DataFrame({'X':x,"Y":y})
print(df)
print(df.shape)

x_train = df.loc[:, 'X']
y_train = df.loc[:, 'Y']
print(x_train.shape, y_train.shape)
x_train = x_train.values.reshape(len(x_train),1) #.values는 넘파이로 바꾸는것!!!


#2. 모델
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(x_train,y_train)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가 예측
score = model.score(x_train,y_train)
print('score:', score)

print("기울기 : ", model.coef_)
print("절편:", model.intercept_)