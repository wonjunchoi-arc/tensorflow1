import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn.datasets import load_iris

dataset= load_iris()
print(dataset.keys())
#dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename'])
print(dataset.target_names)

x= dataset.data
y= dataset.target
print(x.shape, y.shape) #(150, 4) (150,)

df = pd.DataFrame(x, columns=dataset.feature_names)

print(df)
df['Target'] =y
print(df.head())

print("==================상관계수 히트 맵 =======================")
print(df.corr())  
#corr은 pandas에서 제공 하는 것이다!! 이것은 각각에 대한 linear수치이다. 
#반대로 

import matplotlib.pyplot as plt
import seaborn as sns
sns.set(font_scale =1.2)
sns.heatmap(data=df.corr(), square=True, annot=True, cbar=True)

plt.show()