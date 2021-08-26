import tensorflow as tf
print(tf.__version__)

print("hello world")

hello = tf.constant("Hello World")
print(hello)
# Tensor("Const:0", shape=(), dtype=string)
'''
텐서도 넘파이와 비슷하게 자료형이다. 

'''

# sess = tf.Session()
sess = tf.compat.v1.Session()
print(sess.run(hello))
'''
모든 변수고 이런걸 
sess = tf.Session() 이안에 넣어줘야 출력이 된다. 

'''