# -*- coding: utf-8 -*- 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import re
from konlpy.tag._okt import Okt
from gensim import corpora
from tqdm import tqdm
import pickle
plt.rcParams['font.family'] = 'Malgun Gothic'


data = pd.read_excel(r"C:\Users\윤유진\Downloads\엘리_전달용.xlsx")

data = data[['data_id','content','keyword']]
# data=data[data['keyword'].isin(['ace etf','kindex'])]
#%%
"""EDA"""
# """글자수 길이 확인"""

length1 = data['content'].astype(str).apply(len)
length1.head()

plt.figure(figsize=(12,5))
plt.hist(length1 ,bins=int(np.sqrt(length1.shape[0])))
plt.title("컨텐츠의 글자수 분포")
plt.yscale('log')

#낱말 수 확인 
length2 = data['content'].astype(str).apply(lambda x: len(x.split(' ')))
plt.figure(figsize=(12,5))
plt.hist(length2 ,bins=int(np.sqrt(length2.shape[0])))
plt.title("컨텐츠의 낱말수 분포")
plt.yscale('log')

#워드클라우드로 어휘 확인
review_ = [review for review in data['content'] if type(review) is str]
wordcloud = WordCloud(font_path='C:\WINDOWS\FONTS\MALGUNSL.TTF' ).generate(
                        ' '.join(review_))
plt.imshow(wordcloud,interpolation='bilinear')
plt.axis('off')
plt.show()
#%%
"""전처리"""
#정규식을 이용해 한글, 영어, 숫자 외에 모두 제거

def preprocessing(review,okt):
    a = re.sub("[^ㄱ-ㅎ가-힣a-zA-z\\s]","",review)
    
    a_morphs = okt.morphs(a,stem=True)

    stop_words = set(['은','는','이','가','하','아','것','저','들','의',
                      '있','되','수','보','주','등','한','을','다','면',
                      '를','하다','있다','으로','되다','이다','에서','보다','호랑이',
                      'the','to','of','in','and','for','is','The','^^','café'])

    # 불용어 제거 
    clean_review = [w for w in a_morphs if not w in stop_words ]
    # 한글자면 제거 
    clean_review2 = [w.encode('utf-8') for w in clean_review if not len(w)==1]

    return clean_review2


okt=Okt()
clean_review=[]

for w in tqdm(data['content']):
    #비어있는 데이터에서 멈추지 않도록 문자열인 경우에만 진행 
    if type(w)==str:
        clean_review.append(
            preprocessing(w,okt)
        )
    else: #str이 아니면 비어있는 값 추가 
        pass

#시간 너무 걸려서 피클로 저장
import io

with open('clean_review.pickle', 'wb') as f:
    pickle.dump(clean_review, f, pickle.HIGHEST_PROTOCOL)

with io.open('clean_review.pickle', 'rb') as f:
    clean_review = pickle.load(f)
#%%
"""정수 인코딩"""
# clean_review2 = [[i.encode("UTF-8") for i in w] for w in clean_review]

# clean_review = [[d.split() for d in w] for w in clean_review]
# sum(clean_review,[])
dictionary = corpora.Dictionary(clean_review)

corpus = [dictionary.doc2bow(text) for text in clean_review]

len(dictionary)

#%%
"""LDA 훈련"""
import gensim 

NUM_TOPICS = 5

ldamodel = gensim.models.ldamodel.LdaModel(
    corpus, num_topics = NUM_TOPICS, id2word=dictionary, passes=20)

topics = ldamodel.print_topics(num_words=15)
for topic in topics:
    print(topic)

with open('ldamodel.pickle', 'wb') as f:
    pickle.dump(ldamodel, f, pickle.HIGHEST_PROTOCOL)

# with open('ldamodel.pickle', 'rb') as f:
#     ldamodel = pickle.load(f)

#%%
"""시각화"""
import matplotlib.font_manager as fm
from gensim.utils import simple_preprocess

import pyLDAvis
import pyLDAvis.gensim

import gensim
import gensim.corpora as corpora
import pyLDAvis.gensim
import sys
import importlib

importlib.reload(sys)
sys.setdefaultencoding('utf-8')

# gensim.corpora.MmCorpus.serialize('corpus.mm', corpus)
# corpus = corpora.MmCorpus('corpus.mm')
# dictionary.save('dictionary.gensim')

# encoded_dictionary = {k.encode('utf-8'): v for k, v in dictionary.token2id.items()}

prepared_data = pyLDAvis.gensim.prepare(ldamodel, corpus, dictionary, mds='tsne')
#%%
"""문서별 토픽 비율 보기(csv화 해서 하나씩 좀 보자)"""
for i, topic_list in enumerate(ldamodel[corpus]):
    if i==15:
        break
    print(i,'번째 문서의 topic 비율은',topic_list)
#%%
i=2
for i in tqdm(range(len(clean_review))):  # 행
    print(i)
    if clean_review[i] == []:
        del clean_review[i]
        continue
    
    j = 0
    while True:
        # 열 (서로 길이가 다름)
        try:
            
            clean_review[i][j] = clean_review[i][j].encode('utf-8')
            j += 1
        except UnicodeEncodeError:
            print(f"Non-utf8 string found and removed: {clean_text}")
            del clean_review[i][j]
            j += 1
            continue
        
        if j >= len(clean_review[i]):
            break

#%%
data.loc[8,'content']