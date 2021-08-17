import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
# 제너레이터는 1. 데이터를 수치화, 2. 데이터를 증폭하는 기능을 가짐


#1. 데이터 제너레이터를 정의하는 과정
train_datagen= ImageDataGenerator( #train은 증폭시켜서 훈련을 해야겠죠!! 훈련하는 양이 많을 수록 좋자나여!!
    rescale=1./255,
    horizontal_flip=True, #수평으로 이동하겠냐
    vertical_flip=True,
    width_shift_range=0.1, #좌우로 0.1만큼 움직여서 증폭시키겠냐
    height_shift_range=0.1, # 상하로
    rotation_range=5, # 반전시켜서
    zoom_range=1.2, #원래 이미지에서 20정도 더 크게 
    shear_range=0.7,
    fill_mode='nearest', # 내사진을 옆으로 조금 옮기면 공백이 생기는 데 그것을 비슷한 애들로 채우겠당
)

test_datagen = ImageDataGenerator(rescale= 1./255 #검사하려는 애를 바꾸면 안되겠죵
)

# 데이터를 불러오는 과정 
xy_train = train_datagen.flow_from_directory(
'../data/brain/train', # 이미지가 들어있는 폴더 말고 그 전에 폴더를 지정한 이유? 분류된 이미지들이 있는 폴더 자체가 label이 되기 때문이다.  
target_size=(150,150), #내가 가져오려는 데이터를 모두 다음과 같은 사이즈로 가져옴, 데이터가 다르면 분석을 할 수 없자나
batch_size= 5, #y값
class_mode='binary', # Train 폴더안에 2개의 폴더로 label을 줘서
)
#Found 160 images belonging to 2 classes.

xy_test = test_datagen.flow_from_directory(
'../data/brain/test', # 이미지가 들어있는 폴더 말고 그 전에 폴더를 지정한 이유? 분류된 이미지들이 있는 폴더 자체가 label이 되기 때문이다.  
target_size=(150,150), #내가 가져오려는 데이터를 모두 다음과 같은 사이즈로 가져옴, 데이터가 다르면 분석을 할 수 없자나
batch_size= 5, #y값
class_mode='binary', # Train 폴더안에 2개의 폴더로 label을 줘서
)

#Found 120 images belonging to 2 classes.

print(len(xy_train))


# print(xy_train)
#<tensorflow.python.keras.preprocessing.image.DirectoryIterator object at 0x00000176346C8550>
# print(xy_train[0][0])  #x값
# print(xy_train[0][1])   # y값 [1. 0. 1. 1. 0.]

# #[[ [x0] [x1] [x2] [x3] [x4] ], [ [y0], [y1], [y2], [y3], [y4]]] 이런식으로 있는것임


# print(xy_train[0][0].shape,xy_train[0][1].shape) #(5, 150, 150, 3) (5,)

# print(xy_train[31][0]) #마지막 배치 y
# print(xy_train[31][1]) #없어

# print(type(xy_train))#<class 'tensorflow.python.keras.preprocessing.image.DirectoryIterator'>
# print(type(xy_train[0])) #<class 'tuple'>
# print(type(xy_train[0][0])) #<class 'numpy.ndarray'>