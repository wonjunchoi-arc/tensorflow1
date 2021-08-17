from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import numpy as np

#1. 데이터
x=np.array([1,2,3])
y=np.array([1,2,3])


#2. 모델구성
model =Sequential()
model.add(Dense(1, input_dim=1))

#3. 컴파일, 훈련
model.compile(loss='mse',optimizer='adam')

model.fit(x, y, epochs=2000, batch_size=1) #epochs 훈련의 횟수, batch_size 각 자료에 대한 훈련을 1개씩 한다. 1한번 2한번 3한번 하지만 전체 훈련양은 같다. 
#여기서 모델에 Weight값과 bias값이 저장되어 있다. 

#4. 평가, 예측
loss = model.evaluate(x,y)  #위의 mse값을 통한 loss 반환해줌
print('loss', loss)

result = model.predict([4]) #위에서 생성된 weight와 bias에 따라 4라는 값을 넣었을때의 값을 예측해줌
print('4의 예측값', result)