import tensorflow as tf
print(tf.__version__)
print(tf.executing_eagerly())
#False

tf.compat.v1.disable_eager_execution()
    #즉시 실행 모드!
#  얘쓰면 텐서 2에서 텐서 1을 쓸수 있는 것이다.
#원래는 tf.Session() 구조가 안먹히는데 가능하게 해주는 아이인 것이다. 


print(tf.executing_eagerly())
#False

"""
# print("hello world")

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
"""