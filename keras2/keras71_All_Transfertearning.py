#pre trained model 

from tensorflow.keras.applications import VGG16,VGG19,Xception
from tensorflow.keras.applications import ResNet101, ResNet101V2, ResNet152,ResNet152V2
from tensorflow.keras.applications import ResNet50,ResNet50V2
from tensorflow.keras.applications import DenseNet121, DenseNet169, DenseNet201
from tensorflow.keras.applications import InceptionV3, InceptionResNetV2
from tensorflow.keras.applications import MobileNet, MobileNetV2,MobileNetV3Large,MobileNetV3Small
from tensorflow.keras.applications import NASNetLarge, NASNetMobile
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB7

A = [ResNet101, ResNet101V2, ResNet152,ResNet152V2,ResNet50,ResNet50V2,DenseNet121, DenseNet169, DenseNet201,InceptionV3, InceptionResNetV2
,MobileNet, MobileNetV2,MobileNetV3Large,MobileNetV3Small,NASNetLarge, NASNetMobile,EfficientNetB0, EfficientNetB1, EfficientNetB7]


model = EfficientNetB7()

model.trainable=False

model.summary()

print('전체 가중치 갯수:',len(model.weights))
print('훈련가능 가중치 갯수:',len(model.trainable_weights))


# 모델별 파라미터와 웨이트 수들 정리할 것!!

#VGG16
# Total params: 138,357,544
# Trainable params: 0
# Non-trainable params: 138,357,544 
# 32
# 0


#Xception
# Total params: 22,910,480
# Trainable params: 0
# Non-trainable params: 22,910,480
# 236
# 0

# ResNet101
# Total params: 44,707,176
# Trainable params: 0
# Non-trainable params: 44,707,176
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 626
# 훈련가능 가중치 갯수: 0

# ResNet101V2
# Total params: 44,675,560
# Trainable params: 0
# Non-trainable params: 44,675,560
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 544
# 훈련가능 가중치 갯수: 0

# ResNet152
# Total params: 60,419,944
# Trainable params: 0
# Non-trainable params: 60,419,944
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 932
# 훈련가능 가중치 갯수: 0

# ResNet152V2
# Total params: 60,380,648
# Trainable params: 0
# Non-trainable params: 60,380,648
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 816
# 훈련가능 가중치 갯수: 0

# ResNet50
# Total params: 25,636,712
# Trainable params: 0
# Non-trainable params: 25,636,712
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 320
# 훈련가능 가중치 갯수: 0

# ResNet50V2

# Total params: 25,613,800
# Trainable params: 0
# Non-trainable params: 25,613,800
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 272
# 훈련가능 가중치 갯수: 0

# DenseNet121

# Total params: 8,062,504
# Trainable params: 0
# Non-trainable params: 8,062,504
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 606
# 훈련가능 가중치 갯수: 0

# DenseNet169

# Total params: 14,307,880
# Trainable params: 0
# Non-trainable params: 14,307,880
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 846
# 훈련가능 가중치 갯수: 0


# DenseNet201

# Total params: 20,242,984
# Trainable params: 0
# Non-trainable params: 20,242,984
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 1006
# 훈련가능 가중치 갯수: 0



# InceptionV3
# Total params: 23,851,784
# Trainable params: 0
# Non-trainable params: 23,851,784
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 378
# 훈련가능 가중치 갯수: 0



# InceptionResNetV2
# Total params: 55,873,736
# Trainable params: 0
# Non-trainable params: 55,873,736
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 898
# 훈련가능 가중치 갯수: 0

# MobileNet
# Total params: 4,253,864
# Trainable params: 0
# Non-trainable params: 4,253,864
# _________________________________________________________________
# 전체 가중치 갯수: 137
# 훈련가능 가중치 갯수: 0

# MobileNetV2

# Total params: 3,538,984
# Trainable params: 0
# Non-trainable params: 3,538,984
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 262
# 훈련가능 가중치 갯수: 0

# MobileNetV3Large
# Total params: 5,507,432
# Trainable params: 0
# Non-trainable params: 5,507,432
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 266
# 훈련가능 가중치 갯수: 0


# MobileNetV3Small
# Total params: 2,554,968
# Trainable params: 0
# Non-trainable params: 2,554,968
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 210
# 훈련가능 가중치 갯수: 0

# NASNetLarge
# Total params: 88,949,818
# Trainable params: 0
# Non-trainable params: 88,949,818
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 1546
# 훈련가능 가중치 갯수: 0


# NASNetMobile

# Total params: 5,326,716
# Trainable params: 0
# Non-trainable params: 5,326,716
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 1126
# 훈련가능 가중치 갯수: 0


# EfficientNetB0
# Total params: 5,330,571
# Trainable params: 0
# Non-trainable params: 5,330,571
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 314
# 훈련가능 가중치 갯수: 0

# EfficientNetB1

# Total params: 7,856,239
# Trainable params: 0
# Non-trainable params: 7,856,239
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 442
# 훈련가능 가중치 갯수: 0

# EfficientNetB7
# Total params: 66,658,687
# Trainable params: 0
# Non-trainable params: 66,658,687
# __________________________________________________________________________________________________
# 전체 가중치 갯수: 1040
# 훈련가능 가중치 갯수: 0