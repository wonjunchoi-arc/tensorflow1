from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import VGG16,VGG19

'''
이미지 분류 CNN 모델들 중에 하나가 바로 VGGNet이다.

VGGNet은 몇 개의 층(layer)으로 구성되어 있는지에 따라, 16개 층으로 구성되어 있으면 VGG16,

19개 층으로 구성되어 있으면 VGG19라고 불린다.
'''


model =VGG16(weights='imagenet',include_top=False, input_shape=(100,100,3))
 
 
'''
include_top = False 하면 내가 커스터마이징 해서 쓸 수 있다는 것이다!!
삽입하는 이미지의 크기도, 그리고 아래에 FC부분도 없어진다.
'''

#가중치가 이미지넷에서 VGG16을 썻을 때의 것으로 되어 있다는 것!
# model = VGG16()
# model = VGG19()



# 훈련을 시킬지 말지 tip(훈련을 시킴, 안시킴으로 결과 확인)
model.trainable=False

'''
model.trainable=False = 훈련을 하지 않겟다! ,  전이학습 모델의 weight를 그대로 쓰겠다. weight의 갱신이 없다.
Total params: 14,714,688
Trainable params: 0  얘
Non-trainable params: 14,714,688

print(len(model.weights))   => 26
print(len(model.trainable_weights)) => 0

디폴트값은 트루!!
model.trainable=True = 훈련을 하겠다. weight를 새로 계산해서 쓰겠다. 

print(len(model.weights))   => 26
print(len(model.trainable_weights)) => 26

'''
model.summary()


# Total params: 138,357,544
# Trainable params: 138,357,544
# Non-trainable params: 0
'''
Layer (type)                 Output Shape              Param #    
================================================================= 
input_1 (InputLayer)         [(None, 224, 224, 3)]     0
_________________________________________________________________ 
block1_conv1 (Conv2D)        (None, 224, 224, 64)      1792       
________________________________________________________
_________________________________________________________________ 
flatten (Flatten)            (None, 25088)             0
_________________________________________________________________ 
fc1 (Dense)                  (None, 4096)              102764544  
_________________________________________________________________ 
fc2 (Dense)                  (None, 4096)              16781312   
_________________________________________________________________ 
predictions (Dense)          (None, 1000)              4097000    
================================================================= 
'''

#FC = Fully Connected Layer
'''
FC(Fully connected layer)를 정의하자면,

완전히 연결 되었다라는 뜻으로,

한층의 모든 뉴런이 다음층이 모든 뉴런과 연결된 상태로

2차원의 배열 형태 이미지를 1차원의 평탄화 작업을 통해 이미지를 분류하는데 사용되는 계층입니다.

​

1. 2차원 배열 형태의 이미지를 1차원 배열로 평탄화

2. 활성화 함수(Relu, Leaky Relu, Tanh,등)뉴런을 활성화

3. 분류기(Softmax) 함수로 분류

​

1~3과정을 Fully Connected Layer라고 말할 수 있습니다.
[출처] [딥러닝 레이어] FC(Fully Connected Layers)이란?|작성자 인텔리즈
'''