from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16,VGG19

vgg16 =VGG16(weights='imagenet',
include_top=False, input_shape=(100,100,3))
 
vgg16.trainable=False
#model.trainable=False 이거쓰면 모델을 훈련을 안시키는 거니깐 아래 모델 전체로 맛이감

# model.summary()

model = Sequential()
model.add(vgg16)
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(100))

# model.trainable=False

model.summary()

print(len(model.weights))            # 26-> 30
print(len(model.trainable_weights))  # 0-> 4 새로 추가한 덴스레이어 때문
