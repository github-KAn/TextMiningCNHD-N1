# Import NLTK
import nltk,string, numpy as np, matplotlib,matplotlib.pyplot as plt
from collections import Counter
# Scikit-learn has some useful NLP tools, such as a TFIDF vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

# Set an example sentence of 'data' to play with
sentence = "UC San Diego is a great place to study cognitive science."

# If you hit an error downloading things in the cell below, come back to this cell, uncomment it, and run this code.
#   This code gives python permission to write to your disk (if it doesn't already have persmission to do so).
# import ssl
#
# try:
#     _create_unverified_https_context = ssl.create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context

# Download some useful data files from NLTK
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('maxent_ne_chunker')
nltk.download('words')
nltk.download('treebank')
nltk.download('basque_grammars')
nltk.download('knbc')

# Tokenize our sentence, at the word level
tokens = nltk.word_tokenize(sentence)

# Check out the word-tokenized data
print(tokens)

# Apply part-of-speech tagging to our sentence
tags = nltk.pos_tag(tokens)

# Check the POS tags for our data
print(tags)

# Apply named entity recognition to our POS tags
entities = nltk.chunk.ne_chunk(tags)

# Check out the named entities
print(entities)

# Check out the corpus of stop words in English
print(nltk.corpus.stopwords.words('english'))

# Load the data
with open('files/book10k.txt', 'r') as file:
    sents = file.readlines()



# Check out the documentation for describing the abbreviations
# nltk.help.upenn_tagset(tagpattern='NNP')
# nltk.help.upenn_tagset(tagpattern='DT')
# nltk.help.upenn_tagset(tagpattern='JJ')