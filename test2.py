import pandas as pd
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
import os



import pickle

#load data đã được tiền xử lý và lưu vào file pickle
X_data = pickle.load(open('data/X_data_short.pkl', 'rb'))
doc_names = pickle.load(open('data/y_data_short.pkl', 'rb'))

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer



#Tạo đối tượng vectorizer
vectorizer = TfidfVectorizer()
vectorizer.smooth_idf=False


tfidf_matrix = vectorizer.fit_transform(X_data)
ft_names=vectorizer.get_feature_names_out()

# Chuyển ma trận sang dạng dataframe và lưu dưới dạng csv

data={}
tfidf_a=tfidf_matrix.toarray()

for i in range(len(tfidf_a)):
    data[doc_names[i]]=tfidf_a[i]

df=pd.DataFrame(data,index=ft_names)
print(df.to_string())
print((df.transpose()).to_string())
# df.to_csv("train-tf-idf-news.csv",encoding="utf-8-sig")