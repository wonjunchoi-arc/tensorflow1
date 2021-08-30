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

L1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool =tf.nn.max_pool(L1, ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

W2 =tf.get_variable('w2', shape=[3 ,3 ,32 ,64])
L2 = tf.nn.conv2d(L1_maxpool, W2, strides=[1,1,1,1], padding='SAME')
L2= tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L2) #(14,14,64)
print(L2_maxpool) #

#layer 3

W3 =tf.get_variable('w3', shape=[3 ,3 ,64 ,128])
L3 = tf.nn.conv2d(L2_maxpool, W3, strides=[1,1,1,1], padding='SAME')
L3= tf.nn.selu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L3) #(14,14,64)
print(L3_maxpool) 

#layer 4

W4 =tf.get_variable('w4', shape=[2 ,2 ,128 ,64])
L4 = tf.nn.conv2d(L3_maxpool, W4, strides=[1,1,1,1], padding='SAME')
L4= tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L4) #(14,14,64)
print(L4_maxpool) #


#Flatten 
L_flat =tf.reshape(L4_maxpool, [-1,2*2*64])
print("플레튼:",L_flat)


#L5 DNN

W5 = tf.get_variable('w5', shape=[2*2*64,64])
b5 = tf.Variable(tf.random_normal([64]), name='b1')
L5 = tf.matmul(L_flat, W5)+b5
L5 = tf.nn.selu(L5)
L5 = tf.nn.dropout(L5, keep_prob=0.2)

print(L5)  # (?, 64)

#L6 DNN


W6 = tf.get_variable('w6', shape=[64,32])
b6 = tf.Variable(tf.random_normal([32]), name='b2')
L6 = tf.matmul(L5, W6)+b6
L6 = tf.nn.selu(L6)
L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)   #(?, 32)

#L7 Softmax

W7 = tf.get_variable('w7', shape=[32,10])
b7 = tf.Variable(tf.random_normal([10]), name='b3')
L7 = tf.matmul(L6, W7)+b7
hypothesis = tf.nn.softmax(L7)
print(hypothesis)



# model = Sequential()
# model.add(Conv2D(filters=32, kernel_size=(3,3), strides=1,
# padding='same', input_shape=(28,28,1)))
