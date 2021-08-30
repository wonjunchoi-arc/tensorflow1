import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch
from keras.models import Sequential
from keras.layers import Conv2D
tf.set_random_seed(66)

#1. 데이터
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train =to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28,1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28,1).astype('float32')/255

learning_rate = 0.001
training_epochs = 15
batch_size = 100
total_batch =int(len(x_train)/batch_size)


x = tf.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.placeholder(tf.float32, [None, 10])

#모델구성

W1 = tf.get_variable('w1', shape=[3,3,1,32])
# W2 = tf.Variable([3,3,1,32])
'''
x가 4차원이기 때문에 w도 차원이고 
3,3은 kernel을 나타내고, 그다음 1은 채널의 수(인풋의)
마지막 32는 아웃풋의 채널의 수를 나타낸다. 
'''
L1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding='SAME')
'''
stride = [batch, width, height, depth] 로 설정한다.

00 01 02 03 04 ...
10 11 12 13 14 ...
20 21 22 23 24 ...
30 31 32 33 34 ...
...
위와 같은 이미지가 있다고 했을 때, strde = [1, 1, 1, 1] 이면 필터는 아래처럼 잡힌다.
F(00 01        ->    F(01 02              -> ...
  10 11)                 11 12)

-> F(10 11  
      20 21)

width 와 height 를 1로 잡는 바람에 필터에 들어가는 내용이 겹쳐지게 되었다.
이것을 방지하기 위해서 width와 heigth를 2로 설정하면 아래처럼 된다.
즉, strde = [1, 2, 2, 1] 이면 필터는 아래처럼 잡힌다.

F(00 01        ->    F(02 03              -> ...
  10 11)                 12 13)

-> F(20 21  
      30 31)


batch를 1로 설정한 이유는 필터에 내용을 넣을 때 모든 이미지의 원소를 다 넣고싶기 때문이다. 그리고 depth를 1로 설정한 이유도 같다.
'''
model = Sequential()
model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
padding='same', input_shape=(28,28,1)))


####################################get_variable 연구 ########################
#get_Variable은 shape명시해주고 이름도 명시해줘야함 
# 굳이 Variable과의 차이점이라고 한다면,

# Variable 함수는 픽스된 고정값을
# 생성해 주는 반면에,
# get_variable은 매번 새로운 랜덤값을
# 생성해주고 범위 이름과 변수 이름이
# 지정되어야 한다는 점입니다.

#그렇기 때문에 이전시간에 Variable안에 randomnormal을 생성해준것





sess = tf.Session()
sess.run(tf.global_variables_initializer())
print(np.min(sess.run(W2)))
print('===============================================')
print(np.max(sess.run(W2)))
print('===============================================')
print(np.mean(sess.run(W2)))
print('===============================================')
print(np.median(sess.run(W2)))
##################################################################
