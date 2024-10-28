# pip install nltk spacy sklearn textblob gensim
from collections import Counter
import pickle,nltk,ssl, pandas as pd
from idlelib.iomenu import encoding

from nltk.util import ngrams

# Bỏ qua chứng chỉ SSL để tránh lỗi tải xuống
ssl._create_default_https_context = ssl._create_unverified_context

# Tải các công cụ cần thiết cho NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

out_tokens=pickle.load(open("data/filtered_words_en.pkl","rb"))



# BOW: Đếm tần số xuất hiện của các từ và sắp xếp các từ theo tần số xuất hiện
word_counts = Counter(out_tokens)
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
word_per_line=0
BOW=""
for word, count in sorted_words:
    if (word_per_line%4)==0:
        BOW+="\n"
    BOW+=f"{word:}: {count}, "
    word_per_line += 1
word_per_line=0
print(type(word_counts))
print("BOW:",BOW)
with open("BOW.txt","w",encoding="UTF-8") as f:
    f.write(BOW)

#N-gram

print("Uni-gram:")
#in uni-gram
uni_gram=list(ngrams(out_tokens,1))
for e in uni_gram:
    if (word_per_line%10)==0:
        print("")
    print(f"{e}", end="  ")
    word_per_line +=1
word_per_line=0
print("\nBi-gram:")

#in bi-gram
bi_gram=list(ngrams(out_tokens,2))

for e in bi_gram:
    if (word_per_line%10)==0:
        print("")
    print(f"{e}", end="  ")
    word_per_line +=1
word_per_line=0

#in tri-gram
print("\nTri-gram:")
tri_gram=list(ngrams(out_tokens,3))

for e in tri_gram:
    if (word_per_line%10)==0:
        print("")
    print(f"{e}", end="  ")
    word_per_line +=1
word_per_line=0


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
df.to_csv("train-tf-idf-news.csv",encoding="utf-8-sig")


























# print(X_test)

# # import required module
# from sklearn.feature_extraction.text import TfidfVectorizer
#
# d0 = 'Geeks for geeks'
# d1 = 'Geeks'
# d2 = 'r2j'
#
# # merge documents into a single corpus
# string = [d0, d1, d2]
# # create object
#
# vectorizer = TfidfVectorizer()
# vectorizer.smooth_idf=False
#
# data={}
# X = vectorizer.fit_transform(string)
# ft_names=vectorizer.get_feature_names_out()
# print(ft_names)
# a=X.toarray()
# print(len(ft_names))
# print(len(a))
# for i in range(len(ft_names)):
#     data[string[i]]=a[i]
# df=pd.DataFrame(data,index=ft_names)
# print(df.to_string())
# print(df.transpose())


