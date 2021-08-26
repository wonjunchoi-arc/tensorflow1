#실습 !! 맹그러보자
from tensorflow.python.ops.nn_ops import dropout
from tensorflow.keras.datasets import mnist
from keras.datasets import mnist

(x_train, y_train), (x_test, y_test)= mnist.load_data()

print(x_train.shape,y_train.shape)
#(60000, 28, 28) (60000,)
x_train = x_train.reshape(60000,784).astype('float')/255
x_test = x_test.reshape(10000,784).astype('float')/255

y_train = y_train.reshape(60000,1)
y_test = y_test.reshape(10000,1)

from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder(sparse=False) # sparse의 default는 true로 matrix행렬로 반환한다. 하지만 False는 array로 반환 둘의 차이는 잘..
y_train = en.fit_transform(y_train)
y_test = en.fit_transform(y_test)

print(x_train.shape,y_train.shape)

import tensorflow as tf
import numpy as np

x = tf.placeholder(tf.float32, shape=[None,784]) 
y = tf.placeholder(tf.float32, shape=[None,10])

# #히든 레이어 1
W1 = tf.Variable(tf.random_normal([784,10],stddev=0.1), name='weight')
b1 = tf.Variable(tf.zeros([10]), name='bias')
#stddev = 표본을 사용하여 표준편차를 추정합니다. 가중치라 너무 크지않게 조절해줌 !!

# 히든 레이어4
W2 = tf.Variable(tf.random_normal([10,10],stddev=0.1), name='weight')
b2 = tf.Variable(tf.zeros([10]), name='bias')


layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)
# dropout1 = tf.nn.dropout(layer1, keep_prob=0.3)
layer4 = tf.nn.softmax(tf.matmul(layer1, W2) + b2)



# #인풋 (4,2)가 (2,10
# # cost = tf.reduce_mean(tf.square(y-hypothesis))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(layer4), axis=1))
# # binary_crossentropy 

'''
reduce mean은 전체 원소의 합 나누기 갯수 
reduce sum 은 그냥 전체 원소들의 더하기

# '''


optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

predicted = tf.equal(tf.argmax(layer4,1), tf.argmax(y_train, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for epochs in range(100):
    cost_val, hy_val, _ = sess.run([loss, layer4, train],
        feed_dict={x:x_train, y:y_train})
    if epochs % 20 == 0:
        print(epochs, "cost :", cost_val, "\n", hy_val)


is_correct = tf.equal(tf.argmax(layer4, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('정확도:', sess.run(accuracy,
                        feed_dict={x: x_test,
                                   y: y_test}))
'''
정확도: 0.9228
'''

sess.close()