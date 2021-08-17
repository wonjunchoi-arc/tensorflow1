import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape) #(60000, 28, 28) (60000,)
print(x_test.shape, y_test.shape)# (10000, 28, 28) (10000,)

# print(x_test)

# print(np.unique(y_train))

# print(x_train[0])
y_train = y_train.reshape(-1,1)
y_test = y_test.reshape(-1,1)

from sklearn.preprocessing import OneHotEncoder
en = OneHotEncoder()
y_train = en.fit_transform(y_train).toarray()
y_test = en.fit_transform(y_test).toarray()

print(y_train[0])

plt.imshow(x_test[0], 'gray')
plt.show()
