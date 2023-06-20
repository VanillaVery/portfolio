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
from collections import Counter
plt.rcParams['font.family'] = 'Malgun Gothic'


data = pd.read_excel(r"C:\Users\윤유진\Downloads\엘리_전달용.xlsx")

data = data[['data_id','content','keyword']]

data_ace=data[data['keyword'].isin(['ace etf','kindex'])]
data_tiger=data[data['keyword']=='tiger']
#%%
"""ace etf명사 추출"""
def preprocessing(review,okt):
    a = re.sub("[^ㄱ-ㅎ가-힣a-zA-z\\s]","",review)
    
    a_morphs = okt.nouns(a)

    stop_words = set(['은','는','이','가','하','아','것','저','들','의',
                      '있','되','수','보','주','등','한','을','다','면',
                      '를','하다','있다','으로','되다','이다','에서','보다','호랑이',
                      'the','to','of','in','and','for','is','The','^^','café'])

    # 불용어 제거 
    clean_review = [w for w in a_morphs if not w in stop_words ]
    # 한글자면 제거 
    clean_review2 = [w for w in clean_review if not len(w)==1]

    return clean_review2


okt=Okt()
clean_review_ace=[]

for w in tqdm(data_ace['content']):
    #비어있는 데이터에서 멈추지 않도록 문자열인 경우에만 진행 
    if type(w)==str:
        clean_review_ace.append(
            preprocessing(w,okt)
        )
    else: #str이 아니면 비어있는 값 추가 
        pass

clean_review_ace2=[]
for w in clean_review_ace:
    clean_review_ace2.extend(w)

count_ace=Counter(clean_review_ace2).most_common(300)
#%%
"""tiger명사 추출"""
def preprocessing(review,okt):
    a = re.sub("[^ㄱ-ㅎ가-힣a-zA-z\\s]","",review)
    
    a_morphs = okt.nouns(a)

    stop_words = set(['은','는','이','가','하','아','것','저','들','의',
                      '있','되','수','보','주','등','한','을','다','면',
                      '를','하다','있다','으로','되다','이다','에서','보다','호랑이',
                      'the','to','of','in','and','for','is','The','^^','café'])

    # 불용어 제거 
    clean_review = [w for w in a_morphs if not w in stop_words ]
    # 한글자면 제거 
    clean_review2 = [w for w in clean_review if not len(w)==1]

    return clean_review2


okt=Okt()
clean_review_tiger=[]

for w in tqdm(data_tiger['content']):
    #비어있는 데이터에서 멈추지 않도록 문자열인 경우에만 진행 
    if type(w)==str:
        clean_review_tiger.append(
            preprocessing(w,okt)
        )
    else: #str이 아니면 비어있는 값 추가 
        pass

clean_review_tiger2=[]
for w in clean_review_tiger:
    clean_review_tiger2.extend(w)

count_tiger=Counter(clean_review_tiger2).most_common(300)
#%%
"""비중 빼기"""
sum(list(zip(*count_ace))[1]) #904729
sum(list(zip(*count_tiger))[1]) #51725

dic_ace={}
for i in range(300):
    dic_ace[count_ace[i][0]] = count_ace[i][1]/904729

dic_tiger={}
for i in range(300):
    dic_tiger[count_tiger[i][0]] = count_tiger[i][1]/51725

dic_new={}

for ace in dic_ace:
    if ace in dic_tiger.keys():
        dic_new[ace] = round(dic_ace[ace]-dic_tiger[ace],3)*100

sorted(dic_new.items(),key=lambda x:x[1],reverse=True)
len(dic_new)

pd.DataFrame.from_dict(dic_new).to_csv("aa.csv")