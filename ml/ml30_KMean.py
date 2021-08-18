from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import numpy as np
import pandas as pd


datasets = load_iris()
irisDF = pd.DataFrame(data = datasets.data, columns=datasets.feature_names)
print(irisDF)

# 그룹바이 해석!! https://blog.naver.com/PostView.nhn?blogId=wideeyed&logNo=221578773214&parentCategoryNo=&categoryNo=50&viewDate=&isShowPopularPosts=false&from=postView

kmean = KMeans(n_clusters=3, max_iter=300, random_state=66)
kmean.fit(irisDF)
print(irisDF)

results = kmean.labels_

irisDF['cluster'] = kmean.labels_ # 클러스트링해서 생성한 y값
irisDF['target']= datasets.target #원래 y값

a = irisDF.groupby(['cluster','target']).count()
print(a)




'''
n_clusters =  라벨을 몇개 만들겠다!!
max_iter = ~번 작업하겠다.
'''