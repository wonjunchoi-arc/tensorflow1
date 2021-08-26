#실습 !! 맹그러보자
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
print(y_test)
print(y_test.shape)



print(x_train.shape,y_train.shape)


# # 인공지능의 겨울을 극복하자
# # preceptron = > mlp

import tensorflow as tf
import numpy as np



x = tf.placeholder(tf.float32, shape=[None,784]) 
y = tf.placeholder(tf.float32, shape=[None,10])

# #히든 레이어 1
W1 = tf.Variable(tf.random.normal([784,50]), name='weight')
b1 = tf.Variable(tf.random.normal([50]), name='bias')

# hyporthesis = x * W + b # 행렬 연산 에러 발생 
layer1 = tf.nn.relu(tf.matmul(x, W1) + b1)

# 히든 레이어2
W2 = tf.Variable(tf.random.normal([50,25]), name='weight')
b2 = tf.Variable(tf.random.normal([25]), name='bias')

# hyporthesis = x * W + b # 행렬 연산 에러 발생 
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)

# 히든 레이어3
# W3 = tf.Variable(tf.random.normal([25,10]), name='weight')
# b3 = tf.Variable(tf.random.normal([10]), name='bias')

# # hyporthesis = x * W + b # 행렬 연산 에러 발생 
# layer3 = tf.nn.relu(tf.matmul(layer2, W3) + b3)

# 히든 레이어4
W4 = tf.Variable(tf.random.normal([25,10]), name='weight')
b4 = tf.Variable(tf.random.normal([10]), name='bias')

# hyporthesis = x * W + b # 행렬 연산 에러 발생 
layer4 = tf.nn.softmax(tf.matmul(layer2, W4) + b4)



# #인풋 (4,2)가 (2,10
# # cost = tf.reduce_mean(tf.square(y-hypothesis))
loss = tf.reduce_mean(-tf.reduce_sum(y*tf.log(layer4), axis=1))
# # binary_crossentropy 

'''
reduce mean은 전체 원소의 합 나누기 갯수 
reduce sum 은 그냥 전체 원소들의 더하기

# '''


optimizer = tf.train.GradientDescentOptimizer(learning_rate=1e-6)
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

h, c, a = sess.run([layer4, predicted, accuracy], feed_dict = {x:x_train,y:y_train})
print("Hypothesis : \n", h, "\npredict : \n" ,c , "\n Accuarcy : ",a)


results = sess.run(layer4, feed_dict={x:x_test})
print(results, sess.run(tf.argmax(results, 1)))

predicted = tf.equal(tf.argmax(results,1), tf.argmax(y_test, 1))
accuracy = tf.reduce_mean(tf.cast(predicted, dtype=tf.float32))

_, c, a = sess.run([layer4, predicted, accuracy], feed_dict = {x:x_train,y:y_train})
print("\npredict : \n" ,c , "\n test Accuarcy : ",a)

sess.close()