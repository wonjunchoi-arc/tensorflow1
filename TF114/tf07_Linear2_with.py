
# y =wx + b 
# w , b 변수로 두고  x,y는 입력되는 값이므로 placeholder로

import tensorflow as tf
tf.set_random_seed(66)

x_train = [1,2,3]
y_train = [1,2,3]

W = tf.Variable(1, dtype=tf.float32) # 핸덤하게 내맘대로 넣어준
b = tf.Variable(1, dtype=tf.float32) # 초기값!!

hypothesis = x_train *W +b
#hypothesis 얘랑 f(x) 즉 y이랑 같은 말임! =f(x) = wx +b

loss = tf.reduce_mean(tf.square(hypothesis - y_train)) # mse
# ==> 평균(제곱(예측값 -실제값))

optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train = optimizer.minimize(loss)

###########################여기까지가 연산할 수 있는 그래프 모양만 만든것이다###################3333

# sess = tf.Session()
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer()) #위에 선언한 변수때문에

    for step in range(2001):
        sess.run(train)
        if step % 20 ==0:
            print(step, sess.run(loss), sess.run(W),sess.run(b))
