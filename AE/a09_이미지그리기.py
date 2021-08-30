#conv로 만들어라!1


import numpy as np
from scipy.sparse.construct import random
from tensorflow.keras.datasets import mnist
from tensorflow.python.keras.engine.input_layer import Input
import tensorflow as tf
#1. 데이터

x= np.load('./_save/_npy/image/CAE_x.npy')

from sklearn.model_selection import train_test_split
x_train,x_test = train_test_split(
    x, random_state=66, shuffle=True
)

print(x_train.shape,x_test.shape)


# x_train = x_train.reshape(60000,784).astype('float')/255
# x_test = x_test.reshape(10000,784).astype('float')/255


x_train = x_train.reshape(2481,224,224,3).astype('float')
x_test = x_test.reshape(828,224,224,3).astype('float')
#255로 안나눠주는 이유는 이미지 제너레이터에서 이미 255로 나눠줬기 때문이다.




#.2 모델



from matplotlib import pyplot as plt
import random


print(x_train[0].shape)


#이미지 5개를 무작위로 고른다.
# random_images = random.sample(range(output.shape[0]),5)

# #원본(입력) 이미지를 맨 위에 그린다.
#     ax.imshow((x_test[random_images[i]]))
#     # ax.imshow(x_test[random_images[i]].reshape(224,224,3).astype(np.uint8), cmap='gray')
#     if i ==0:
#         ax.set_ylabel("INPUT", size=20)
#     ax.grid(False)
#     ax.set_xticks([])
#     ax.set_yticks([])

plt.imshow(x_train[0])
# plt.tight_layout()
plt.show()

    
