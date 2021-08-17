import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()
#(_) 언더바 하나는 변수로 받지 않겠다 뜻이다!! 
print(x_train.shape, x_test.shape) #(60000, 28, 28) (10000, 28, 28)

x = np.append(x_train, x_test, axis=0)
print(x.shape)

x= x.reshape(x.shape[0],28*28)


pca = PCA(n_components=780)

x = pca.fit_transform(x)
print(x)
print(x.shape)

pca_EVR=pca.explained_variance_ratio_
print(pca_EVR)
# 압축한 결과에 대한 중요도 큰 순서대로 나열함
'''
[0.40242142 0.14923182 0.12059623 0.09554764 0.06621856 0.06027192
 0.05365605 0.04336832 0.007831
'''

print(sum(pca_EVR))
#sum값을 기준으로 0.95이상이라면 등 나만의 기준을 정하여 PCA갯수를 조정하면 되는 것이다!!

cumsum = np.cumsum(pca_EVR)
print(cumsum)

print(np.argmax(cumsum >= 0.99)+1)

#0.95 ==> 154
#0.99 ==>331

#실습 
#pca를 통해 
