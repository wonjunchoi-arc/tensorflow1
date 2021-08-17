from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten

#kernel size 는 28,28 크기의 이미지를 2,2사이즈로 잘라서 작업을 하겠다. 이걸 쓰는 이유는 특성을 더 명확하게 강조하겠다는 것!!
#10은 아웃풋 노드를 나타냄 !!
# padding은 데이터의 가장자리에 0값을 넣어 데이터를 보존해 주는 것이다. 즉 컨불러션에서 데이터 값이 작아지는 것을 막아준다. 
#conv를 많이 쌓는 것 또한 하이퍼파라미터 튜닝이다. 

model = Sequential()                                            #(N,5,5,1)
model.add(Conv2D(10, kernel_size=(2,2), padding='same' , input_shape=(5,5,1))) #=> (N,4,4,10)
model.add(Conv2D(20, (2,2), activation='relu'))                 #(N,3,3,20)
model.add(Conv2D(30, (2,2), padding='valid', activation='relu'))                 #(N,3,3,20)                                   
model.add(Conv2D(30, (2,2), padding='valid', activation='relu'))                 #(N,3,3,20)                                   
model.add(Conv2D(10, (2,2), padding='valid', activation='relu'))                 #(N,3,3,20)                                   
model.add(Flatten())                                            #=> (N, 180)  
model.add(Dense(64, activation='relu'))
model.add(Dense(32))
model.add(Dense(1, activation='sigmoid'))



model.summary()