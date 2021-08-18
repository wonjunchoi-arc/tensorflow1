#실습 : 다차원 outlier가 출력 되도록
import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,8,90,100,5000],
[100,2000,3,4000,5000,6000,7000,8,9000,10000, 1001]])

aaa =aaa.transpose()
print(aaa.shape)

from sklearn.covariance import EllipticEnvelope

outliers = EllipticEnvelope(contamination=.2)
outliers.fit(aaa)

results = outliers.predict(aaa)

print(results)
