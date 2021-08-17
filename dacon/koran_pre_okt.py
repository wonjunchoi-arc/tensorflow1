from gensim.corpora import dictionary
import pandas as pd
import matplotlib.pyplot as plt
import urllib.request
from gensim.models.word2vec import Word2Vec
from konlpy.tag import Okt
import numpy as np

import warnings 
# warnings.filterwarnings(action='ignore')



# train = pd.read_csv('../data/train_data.csv',index_col=None,  
#                          header=0,usecols=[1, 2])#usecols=원하는 컬럼 가져오기
# test = pd.read_csv('../data/test_data.csv',index_col=None,
#                          header=0)

# submission = pd.read_csv('../data/sample_submission.csv', header=0)

# train1 = train["title"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")
# test1 = test["title"].str.replace("[^ㄱ-ㅎㅏ-ㅣ가-힣 ]","")

# stopwords = ['의','가','이','은','들','는','좀','잘','걍','과','도','를','으로','자','에','와','한','하다']

# okt = Okt()
# train=[]
# for i in range(len(train1)):
#     try:
#         train.append(okt.nouns(train1[i]))
#     except Exception as e:
#         continue

# test=[]
# for i in range(len(test1)):
#     try:
#         test.append(okt.nouns(test1))
#     except Exception as e:
#         continue





# # train=train.loc[3]
# # print(type(train))



# # print(train)


print('111111111111111111',train)

train = np.array(train)
test = np.array(test)
print('2222222222222222222222222',train)

train = pd.DataFrame(train)
test = pd.DataFrame(test)

print('3333333333333333333333333',train)


train.to_csv('../data/dacon/train_okt.csv',index=False)
test.to_csv('../data/dacon/test_okt.csv',index =False)



Y = pd.read_csv('../data/train_data.csv',index_col=None,  
                         header=0)#usecols=원하는 컬럼 가져오기

train = pd.read_csv('../data/dacon/train_okt.csv',index_col=None,  
                         header=0)#usecols=원하는 컬럼 가져오기
test = pd.read_csv('../data/dacon/test_okt.csv',index_col=None,
                         header=0)

from gensim import corpora, models
import gensim
train = np.array(train).tolist()
print(train[0:2])


high_score_reviews=[[y for y in x if not len(y)==1]
for x in train]
print(high_score_reviews)
dictionary = corpora.Dictionary(high_score_reviews)
corpus = [dictionary.doc2bow(text) for text in high_score_reviews]


# import matplotlib.pyplot as plt
# from gensim.models import CoherenceModel

# coherenece_values =[]
# for i in range(2,15):
#         ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word=dictionary)
#         coherence_model_ida = CoherenceModel(model=ldamodel, texts=high_score_reviews, dictionary=dictionary, topn=10)
#         coherence_ida = coherence_model_ida.get_coherence()
#         coherenece_values.append(coherence_ida)

# x=range(2,15)
# plt.plot(x, coherenece_values)
# plt.xlabel("number or topics")
# plt.ylabel("coherence score")
# plt.show()

# perplexity_values = []
# for i in range(2,20):
#         ldamodel=gensim.models.ldamodel.LdaModel(corpus, num_topics=i, id2word=dictionary)
#         perplexity_values.append(ldamodel.log_perplexity(corpus))

# x = range(2,20)
# plt.plot(x, perplexity_values)
# plt.plot("number of topics")
# plt.plot("perplexity score")
# plt.show

dictionary = corpora.Dictionary(train)
corpus = [dictionary.doc2bow(text)] 