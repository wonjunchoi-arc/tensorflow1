import tensorflow as tf
sess =tf.Session()

x = tf.Variable([2], dtype=tf.float32, name='test')

init = tf.global_variables_initializer()
#변수를 그래프에 들어가기 적합한 형태로 초기화를 시킨다는 말이다
#변수 하나하나 할 필요없고 이거 한번 하면 변수 전부 먹음

sess.run(init)
#모든 것은 Session안에서 실행이 되는 것이다!
print('프린트 x 나왔나 확인',sess.run(x))

'''
초기화를 해달라!!
Attempting to use uninitialized value test
         [[{{node _retval_test_0_0}}]] 

'''