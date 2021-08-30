from numpy.lib.function_base import average
import tensorflow as tf
import numpy as np
from tensorflow.python.ops.gen_batch_ops import batch
from keras.models import Sequential
from keras.layers import Conv2D

tf.compat.v1.disable_eager_execution()
print(tf.executing_eagerly()) #False
print(tf.__version__) #  1.14 -> 2.4.1

##위에코드 실행 후 compat.v1 실행 안되는 각 요소들에 그러면 gpu로 조지게 빨리 돌아갈 것이야!!

tf.compat.v1.set_random_seed(66)

#1. 데이터
from tensorflow.keras.datasets import mnist
(x_train,y_train),(x_test,y_test) = mnist.load_data()

from tensorflow.keras.utils import to_categorical
y_train =to_categorical(y_train)
y_test = to_categorical(y_test)

x_train = x_train.reshape(60000, 28, 28,1).astype('float32')/255
x_test = x_test.reshape(10000, 28, 28,1).astype('float32')/255

learning_rate = 0.0001
training_epochs = 15
batch_size = 100
total_batch =int(len(x_train)/batch_size)


x = tf.compat.v1.placeholder(tf.float32, [None, 28, 28, 1])
y = tf.compat.v1.placeholder(tf.float32, [None, 10])

#모델구성

W1 = tf.compat.v1.get_variable('w1', shape=[3,3,1,32])

L1 = tf.nn.conv2d(x, W1, strides=[1,1,1,1], padding='SAME')
L1 = tf.nn.relu(L1)
L1_maxpool =tf.nn.max_pool(L1, ksize=[1,2,2,1],
                            strides=[1,2,2,1],
                            padding='SAME')

W2 =tf.compat.v1.get_variable('w2', shape=[3 ,3 ,32 ,64])
L2 = tf.nn.conv2d(L1_maxpool, W2, strides=[1,1,1,1], padding='SAME')
L2= tf.nn.selu(L2)
L2_maxpool = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L2) #(14,14,64)
print(L2_maxpool) #

#layer 3

W3 =tf.compat.v1.get_variable('w3', shape=[3 ,3 ,64 ,128])
L3 = tf.nn.conv2d(L2_maxpool, W3, strides=[1,1,1,1], padding='SAME')
L3= tf.nn.selu(L3)
L3_maxpool = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L3) #(14,14,64)
print(L3_maxpool) 

#layer 4

W4 =tf.compat.v1.get_variable('w4', shape=[2 ,2 ,128 ,64],)
#initializer = tf.contrib.layers.xavier_initializer())
#초반에 너무 큰 가중치가 들어가서 터질경우 등에 대하여 가중치를 초기화해주는 것이다!!
L4 = tf.nn.conv2d(L3_maxpool, W4, strides=[1,1,1,1], padding='SAME')
L4= tf.nn.leaky_relu(L4)
L4_maxpool = tf.nn.max_pool(L4, ksize=[1,2,2,1], strides=[1,2,2,1],padding='SAME')
print(L4) #(14,14,64)
print(L4_maxpool) #


#Flatten 
L_flat =tf.reshape(L4_maxpool, [-1,2*2*64])
print("플레튼:",L_flat)


#L5 DNN

W5 = tf.compat.v1.get_variable('w5', shape=[2*2*64,64],)
#initializer = tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.compat.v1.random_normal([64]), name='b1')
L5 = tf.matmul(L_flat, W5)+b5
L5 = tf.nn.selu(L5)
# L5 = tf.nn.dropout(L5, keep_prob=0.2)

print(L5)  # (?, 64)

#L6 DNN


W6 = tf.compat.v1.get_variable('w6', shape=[64,32],)
#initializer = tf.contrib.layers.xavier_initializer())
b6 = tf.Variable(tf.compat.v1.random_normal([32]), name='b2')
L6 = tf.matmul(L5, W6)+b6
L6 = tf.nn.selu(L6)
# L6 = tf.nn.dropout(L6, keep_prob=0.2)
print(L6)   #(?, 32)

#L7 Softmax

W7 = tf.compat.v1.get_variable('w7', shape=[32,10])
b7 = tf.Variable(tf.compat.v1.random_normal([10]), name='b3')
L7 = tf.matmul(L6, W7)+b7
hypothesis = tf.nn.softmax(L7)
print(hypothesis)



#3.모델 컴파일, 훈련


loss = tf.reduce_mean(-tf.reduce_sum(y*tf.compat.v1.log(hypothesis), axis=1))
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
# train = optimizer.minimize(loss)

# optimizers = tf.train.GradientDescentOptimizer(learning_rate=0.01).minimize
optimizers = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)

sess = tf.compat.v1.Session()
sess.run(tf.compat.v1.global_variables_initializer())

for epoch in range(training_epochs):
    average_loss = 0
    #아래는 1epoch돌리기
    for i in range(total_batch) :
        start = i *batch_size
        end = start + batch_size
        batch_x, batch_y = x_train[start:end], y_train[start:end]
        ###배치 크기별로 짤라서 훈련시켜주는 작업이다!!

        feed_dict = {x:batch_x, y:batch_y}

        batch_loss,__ = sess.run([loss, optimizers], feed_dict=feed_dict)

        average_loss += batch_loss/total_batch
    
    print('Epoch:','%04d' %(epoch +1), 'loss: {:.9f}'.format(average_loss))
print("훈련 끘!!!")


is_correct = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
#cast 소수점을 빼준다!
print('ACC:',sess.run(accuracy,feed_dict={x:x_test, y:y_test}))
