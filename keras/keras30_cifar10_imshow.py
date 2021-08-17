import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape, y_train.shape) 
print(x_test.shape, y_test.shape)

y_train = y_train.reshape(-1,1)

print('y트레인 이거임', y_train.shape)

print(np.unique(y_train))

print(x_train[0])
from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

print(y_train[0])

plt.imshow(x_train[1], 'gray')
plt.show()
