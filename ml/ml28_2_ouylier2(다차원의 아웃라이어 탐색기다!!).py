#실습 : 다차원 outlier가 출력 되도록
import numpy as np

aaa = np.array([[1,2,3,4,10000,6,7,8,90,100,5000],
[100,2000,-30000,4000,5000,6000,7000,8,9000,10000, 1001]])

aaa =aaa.transpose()
print(aaa.shape)



def outliers(data_out):
    A = []
    for i in range(data_out.shape[1]):
        quartile_1,q2,quartile_3 = np.percentile(data_out[:,i],[25,50,75])#분위수
        print(f"{i}컬럼의 1사분위:",quartile_1)
        print(f"{i}컬럼의 2사분위:",q2)
        print(f"{i}컬럼의 3사분위:",quartile_3)
        iqr = quartile_3-quartile_1 #전체의 50% 범위에 해당
        lower_bound = quartile_1 - (iqr * 1.5)
        upper_bound = quartile_3 + (iqr * 1.5)
        b = np.where((data_out[:,i]>upper_bound) | (data_out[:,i]<lower_bound))
        A.append(f'{i}컬럼의 이상치 위치는:{b}',)
    return A
outliers_loc = outliers(aaa)
print(outliers_loc)

# outliers_loc = outliers(aaa
print(aaa.shape[1])


# for i in range(aaa.shape[1]):
#     quartile_1,q2,quartile_3 = np.percentile(aaa[:,i],[25,50,75])#분위수
#     print(f"{i}컬럼의 1사분위:",quartile_1)
#     print(f"{i}컬럼의 2사분위:",q2)
#     print(f"{i}컬럼의 3사분위:",quartile_3)
#     iqr = quartile_3-quartile_1 #전체의 50% 범위에 해당
#     lower_bound = quartile_1 - (iqr * 1.5)
#     upper_bound = quartile_3 + (iqr * 1.5)
#     print(lower_bound)
#     print(f'{i}컬럼의 이상치 위치는:',np.where((aaa[:,i]>upper_bound) | (aaa[:,i]<lower_bound)))



####################다중도 들어 갈 수 있도록 만들기!!!################


# import matplotlib.pyplot as plt
# plt.figure(figsize=(7,6))
# plt.boxplot(aaa)
# plt.show()