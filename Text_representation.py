# pip install nltk spacy sklearn textblob gensim
from collections import Counter
from idlelib.iomenu import encoding
from math import log

import Index,spacy_NPL,nltk,ssl
from nltk.corpus import stopwords
from nltk.util import bigrams
from nltk.util import ngrams
from nltk.text import TextCollection
from Index import text_data
from nltk.tokenize import word_tokenize
# Bỏ qua chứng chỉ SSL để tránh lỗi tải xuống
ssl._create_default_https_context = ssl._create_unverified_context

# Tải các công cụ cần thiết cho NLTK
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')

out_tokens=spacy_NPL.filtered_words

def termFrequency(term, doc):
    """
    Input: term: Term in the Document, doc: Document
    Return: Normalized tf: Number of times term occurs
      in document/Total number of terms in the document
    """
    # Splitting the document into individual terms
    normalizeTermFreq = doc.lower().split()

    # Number of times the term occurs in the document
    term_in_document = normalizeTermFreq.count(term.lower())

    # Total number of terms in the document
    len_of_document = float(len(normalizeTermFreq))

    # Normalized Term Frequency
    normalized_tf = term_in_document / len_of_document

    return normalized_tf


def inverseDocumentFrequency(term, allDocs):
    num_docs_with_given_term = 0

    """
    Input: term: Term in the Document,
        allDocs: List of all documents
    Return: Inverse Document Frequency (idf) for term
            = Logarithm ((Total Number of Documents) /
            (Number of documents containing the term))
    """
    # Iterate through all the documents
    for doc in allDocs:

        """
        Putting a check if a term appears in a document.
        If term is present in the document, then 
        increment "num_docs_with_given_term" variable
        """
        if term.lower() in allDocs[doc].lower().split():
            num_docs_with_given_term += 1

    if num_docs_with_given_term > 0:
        # Total number of documents
        total_num_docs = len(allDocs)

        # Calculating the IDF
        idf_val = log(float(total_num_docs) / num_docs_with_given_term)
        return idf_val
    else:
        return 0

#Bag of word
# Tạo mô hình bag-of-words
# word_counts = {}

#Tính tần suất xuất hiện từ
# for word in out_tokens:
#         if word in word_counts:
#             word_counts[word] += 1
#         else:
#             word_counts[word] = 1
# print("BOW:")
# BOW: Đếm tần số xuất hiện của các từ và sắp xếp các từ theo tần số xuất hiện
word_counts = Counter(out_tokens)
sorted_words = sorted(word_counts.items(), key=lambda x: x[1], reverse=True)
word_per_line=0
BOW=""
for word, count in sorted_words:
    if (word_per_line%5)==0:
        BOW+="\n"
    BOW+=f"{word:14}: {count:3}  |"
    word_per_line += 1
    # print(f"{word}: {count}")
print(type(word_counts))
print("BOW:",BOW)

#N-gram
print("Uni-gram:",list(ngrams(out_tokens,1)))
print("Bi-gram:",list(ngrams(out_tokens,2)))
print("Tri-gram:",list(ngrams(out_tokens,3)))
#TF-IDF
# doc1="Ben Studies Computer Lab"
# doc2="Steve teaches Brown University"
# doc3="Data scientists work large datasets"
# # doc4=[doc1,doc2,doc3]
# doc4={"doc1":doc1,"doc2":doc2,"doc3":doc3}
# doc4_str=doc1+doc2+doc3
# doc4_token=word_tokenize(doc4_str.lower())
# doc4_token=[word for word in doc4_token if word.isalpha() and word not in stopwords.words('english')]
# for term in doc4_token:
#     print(f"{term}: {inverseDocumentFrequency(term,doc4)}")

# import required module
from sklearn.feature_extraction.text import TfidfVectorizer

d0 = 'Geeks for geeks'
d1 = 'Geeks'
d2 = 'r2j'

# merge documents into a single corpus
string = [d0, d1, d2]
# create object
tfidf = TfidfVectorizer()

# get tf-df values
result = tfidf.fit_transform(string)

# get idf values
print('\nidf values:')
for ele1, ele2 in zip(tfidf.get_feature_names_out(), tfidf.idf_):
    print(ele1, ':', ele2)

# get indexing
print('\nWord indexes:')
print(tfidf.vocabulary_)

# display tf-idf values
print('\ntf-idf value:')
print(result)

# in matrix form
print('\ntf-idf values in matrix form:')
print(result.toarray())

with open('Processed Results/bow_text_rep.txt', 'w', encoding='utf-8') as file:
    file.write(str(word_counts))


