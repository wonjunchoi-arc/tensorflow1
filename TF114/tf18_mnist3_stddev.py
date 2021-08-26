import tensorflow as tf
import matplotlib.pyplot as plt

a=tf.random_normal([784,10],stddev=0.1)
with tf.Session() as sess :
    A = a.eval()
    plt.hist(A, density=True)
    plt.show()