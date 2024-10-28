import string

import pandas as pd,re
from pyvi import ViTokenizer, ViPosTagger # thư viện NLP tiếng Việt
from tqdm import tqdm
import numpy as np
import gensim # thư viện NLP
import os
from spacy.lang.vi import Vietnamese, STOP_WORDS
# from Text_representation import ft_names

def remove_punctuation(data):
    punctuation_free="".join([i for i in data if i not in string.punctuation])
    return punctuation_free
def process_data(data):
    # Đối tượng "nlp"  được dùng để tạo documents với chú thích ngôn ngữ (linguistic annotations).
    nlp_vi = Vietnamese()
    data = remove_punctuation(data)
    data = re.sub("[-\n]", " ", data)

    my_doc = ViTokenizer.tokenize(data)
    print(my_doc)
    # my_doc=my_doc.spli
    # Tạo danh sách tokens từ
    token_list = []
    # for token in my_doc:
    #     token_list.append(token)
    token_list=my_doc.split()
    # Tạo danh sách tokens sau khi loại bỏ từ dùng
    filtered_words = [word for word in token_list if word.isalpha() and word not in STOP_WORDS]
    processed_text = ' '.join(filtered_words)
    return processed_text


dir_path = os.path.dirname(os.path.realpath(os.getcwd()))
dir_path = os.path.join(dir_path, 'TextMiningCNHD-N1')

def get_data(folder_path):
    X = []
    y = []
    dirs = os.listdir(folder_path)
    for path in tqdm(dirs):
        file_paths = os.listdir(os.path.join(folder_path, path))
        for file_path in tqdm(file_paths):
            with open(os.path.join(folder_path, path, file_path), 'r', encoding="utf-8") as f:
                lines = f.readlines()
                lines = ' '.join(lines)
                lines = gensim.utils.simple_preprocess(lines)
                lines = ' '.join(lines)
                lines = ViTokenizer.tokenize(lines)

                X.append(lines)
                y.append(file_path)
                # print(f"folder_path {folder_path}, path {path}, file_path {file_path}" )
                # text = f.read()
                # X.append(process_data(text))
                # y.append(file_path)

    return X, y

# train_path = os.path.join(dir_path, 'VNTC-master/Data/10Topics/Ver1.1/Train_Short')
train_path = os.path.join(dir_path, 'Vi_sample/News')
X_data, y_data = get_data(train_path)

import pickle

pickle.dump(X_data, open('data/X_data_short.pkl', 'wb'))
pickle.dump(y_data, open('data/y_data_short.pkl', 'wb'))

test_path = os.path.join(dir_path, 'VNTC-master/Data/10Topics/Ver1.1/Test_Short')
X_test, y_test = get_data(test_path)

pickle.dump(X_test, open('data/X_test_short.pkl', 'wb'))
pickle.dump(y_test, open('data/y_test_short.pkl', 'wb'))

import pickle

X_data = pickle.load(open('data/X_data_short.pkl', 'rb'))
y_data = pickle.load(open('data/y_data_short.pkl', 'rb'))
print(X_data, type(X_data))
# X_test = pickle.load(open('data/X_test_short.pkl', 'rb'))
# y_test = pickle.load(open('data/y_test_short.pkl', 'rb'))

from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer


# # create a count vectorizer object
# count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}')
# count_vect.fit(X_data)
#
# # transform the training and validation data using count vectorizer object
# X_data_count = count_vect.transform(X_data)
# X_test_count = count_vect.transform(X_test)
# print("X_data_count",X_data_count)
# print("X_data_count feature",count_vect.get_feature_names_out())
# X_ft_names=count_vect.get_feature_names_out()
# X_data={}
# for i in range(3):
#     print(i)
#     X_data[X_ft_names[i]]=X_data_count[i]
# print(X_data)
# # df=pd.DataFrame(X_data)
# # print("df:\n {df} ")
# # word level - we choose max number of words equal to 30000 except all words (100k+ words)
# tfidf_vect = TfidfVectorizer(analyzer='word', max_features=30000)
# tfidf_vect.fit(X_data) # learn vocabulary and idf from training set
# X_data_tfidf =  tfidf_vect.transform(X_data)
# # assume that we don't have test set before
# X_test_tfidf =  tfidf_vect.transform(X_test)
#
# print("X_data_tfidf:\n",X_data_tfidf)
#
# # ngram level - we choose max number of words equal to 30000 except all words (100k+ words)
# tfidf_vect_ngram = TfidfVectorizer(analyzer='word', max_features=30000, ngram_range=(2, 3))
# tfidf_vect_ngram.fit(X_data)
# X_data_tfidf_ngram =  tfidf_vect_ngram.transform(X_data)
# # assume that we don't have test set before
# X_test_tfidf_ngram =  tfidf_vect_ngram.transform(X_test)
#
# print("X_data_tfidf:\n",X_data_tfidf_ngram)
#
# # ngram-char level - we choose max number of words equal to 30000 except all words (100k+ words)
# tfidf_vect_ngram_char = TfidfVectorizer(analyzer='char', max_features=30000, ngram_range=(2, 3))
# tfidf_vect_ngram_char.fit(X_data)
# X_data_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_data)
# # assume that we don't have test set before
# X_test_tfidf_ngram_char =  tfidf_vect_ngram_char.transform(X_test)
#
# print("X_data_tfidf:\n",X_data_tfidf_ngram_char)