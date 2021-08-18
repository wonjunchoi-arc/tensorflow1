#실습 : 다차원 outlier가 출력 되도록
import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,8,90,100,5000],
[100,2000,3,4000,5000,6000,7000,8,9000,10000, 1001]])

aaa =aaa.transpose()
print(aaa.shape)



def outliers(data_out):
    quartile_1,q2,quartile_3 = np.percentile(data_out,[25,50,75])#분위수
    print("1사분위:",quartile_1)
    print("2사분위:",q2)
    print("3사분위:",quartile_3)
    iqr = quartile_3-quartile_1 #전체의 50% 범위에 해당
    lower_bound = quartile_1 - (iqr * 1.5)
    upper_bound = quartile_3 + (iqr * 1.5)
    return np.where((data_out>upper_bound) | (data_out<lower_bound))

outliers_loc = outliers(aaa)

print("이상치의 위치: ", outliers_loc)

import matplotlib.pyplot as plt
plt.figure(figsize=(7,6))
plt.boxplot(aaa)
plt.show()