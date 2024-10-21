# pip install nltk spacy sklearn textblob gensim
from idlelib.iomenu import encoding
import nltk
import ssl
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.text import TextCollection
import Index
from Index import text_data

out_tokens=Index.tokens
text_data=Index.text_data
#Bag of word
# Tạo mô hình bag-of-words
word_counts = {}

#Tính tần suất xuất hiện từ
for word in out_tokens:
        if word in word_counts:
            word_counts[word] += 1
        else:
            word_counts[word] = 1

print(type(word_counts))
print("BOW:",word_counts)

#N-gram
print("Uni-gram:",list(ngrams(out_tokens,1)))
print("Bi-gram:",list(ngrams(out_tokens,2)))
print("Tri-gram:",list(ngrams(out_tokens,3)))
#TF-IDF
new_list=["a","a", "a","a","b"]
text_collection=TextCollection(out_tokens)
for text in new_list:
    print(text+":"+str(text_collection.tf_idf(text,new_list)))
with open('Processed Results/bow_text_rep.txt', 'w', encoding='utf-8') as file:
    file.write(str(word_counts))

stop_words = set(stopwords.words('english')) # define stopwords
senteces=out_tokens
word_freq = {}
sentence_word_freq = []

for sentence in sentences:
    # Lowercase
    words = word_tokenize(sentence.lower())
    # Remove stopwords and non alphanumeric words
    words = [word for word in words if word.isalnum()
                              and word not in stop_words]
    freq = {}
    for word in words:
        if word not in freq:
            freq[word] = 0
        # Increment the frequency of the word for a sentence
        freq[word] += 1
    sentence_word_freq.append(freq)
    for word in freq:
        if word not in word_freq:
            word_freq[word] = 0
        # Increment the frequency of the word overall
        word_freq[word] += 1

# Frequency of terms in the document
print(word_freq)
# Frequency of terms per sentence
print(sentence_word_freq)
