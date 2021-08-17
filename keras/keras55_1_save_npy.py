#매번 csv로 데이터를 불러오는 번거로움을 해소하기 위함이다.
#이에 numpy 형태로 저장해서 불러오자

from sklearn import datasets
from sklearn.datasets import load_iris,load_boston,load_breast_cancer,load_diabetes,load_wine
import numpy as np
from tensorflow.keras.datasets import mnist, fashion_mnist, cifar10, cifar100

dataset1 = load_iris()
dataset2 = load_boston()
dataset3 = load_breast_cancer()
dataset4 = load_diabetes()
dataset5 = load_wine()

(x_train_mnist, y_train_mnist), (x_test_mnist, y_test_mnist) = mnist.load_data()
(x_train_fashion, y_train_fashion), (x_test_fashion, y_test_fashion) = fashion_mnist.load_data()
(x_train_cifar10, y_train_cifar10), (x_test_cifar10, y_test_cifar10) = cifar10.load_data()
(x_train_cifar100, y_train_cifar100), (x_test_cifar100, y_test_cifar100) = cifar100.load_data()



x_data_iris = dataset1.data
y_data_iris = dataset1.target

x_data_boston = dataset2.data
y_data_boston = dataset2.target

x_data_cancer = dataset3.data
y_data_cancer = dataset3.target

x_data_diabete = dataset4.data
y_data_diabete = dataset4.target

x_data_wine = dataset5.data
y_data_wine = dataset5.target



# print(type(x_data),type(y_data))
#<class 'numpy.ndarray'> <class 'numpy.ndarray'>

np.save('./_save/_npy/k55_x_data_iris.npy', arr=x_data_iris)
np.save('./_save/_npy/k55_y_data_iris.npy', arr=y_data_iris)


np.save('./_save/_npy/k55_x_data_boston.npy', arr=x_data_boston)
np.save('./_save/_npy/k55_y_data_boston.npy', arr=y_data_boston)


np.save('./_save/_npy/k55_x_data_cancer.npy', arr=x_data_cancer)
np.save('./_save/_npy/k55_y_data_cancer.npy', arr=y_data_cancer)


np.save('./_save/_npy/k55_x_data_diabete.npy', arr=x_data_diabete)
np.save('./_save/_npy/k55_y_data_diabete.npy', arr=y_data_diabete)


np.save('./_save/_npy/k55_x_data_wine.npy', arr=x_data_wine)
np.save('./_save/_npy/k55_y_data_wine.npy', arr=y_data_wine)


np.save('./_save/_npy/x_train_mnist.npy', arr=x_train_mnist)
np.save('./_save/_npy/y_train_mnist.npy', arr=y_train_mnist)
np.save('./_save/_npy/x_test_mnist.npy', arr=x_test_mnist)
np.save('./_save/_npy/y_test_mnist.npy', arr=y_test_mnist)

np.save('./_save/_npy/x_train_fashion.npy', arr=x_train_fashion)
np.save('./_save/_npy/y_train_fashion.npy', arr=y_train_fashion)
np.save('./_save/_npy/x_test_fashion.npy', arr=x_test_fashion)
np.save('./_save/_npy/y_test_fashion.npy', arr=y_test_fashion)

np.save('./_save/_npy/x_train_cifar10.npy', arr=x_train_cifar10)
np.save('./_save/_npy/y_train_cifar10.npy', arr=y_train_cifar10)
np.save('./_save/_npy/x_test_cifar10.npy', arr=x_test_cifar10)
np.save('./_save/_npy/y_test_cifar10.npy', arr=y_test_cifar10)

np.save('./_save/_npy/x_train_cifar100.npy', arr=x_train_cifar100)
np.save('./_save/_npy/y_train_cifar100.npy', arr=y_train_cifar100)
np.save('./_save/_npy/x_test_cifar100.npy', arr=x_test_cifar100)
np.save('./_save/_npy/y_test_cifar100.npy', arr=y_test_cifar100)
