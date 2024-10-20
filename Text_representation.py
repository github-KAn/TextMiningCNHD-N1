# pip install nltk spacy sklearn textblob gensim
from idlelib.iomenu import encoding
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob
import ssl
import Index



tokens=Index.tokens

# Tạo mô hình bag-of-words
word_counts = {}

#Tính tần suất xuất hiện từ
for word in tokens:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

print(type(word_counts))
print("BOW:",word_counts)

with open('Processed Results/bow_text_rep.txt', 'w', encoding='utf-8') as file:
    file.write(str(word_counts))
